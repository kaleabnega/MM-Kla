"""Five-fold protein-grouped TransKla ensemble on CLEAR-Lactyl validation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from run_transkla_clear import (
    AUTHOR_COMMIT, AUTHOR_SOURCE_SHA256, load_author, load_lock, make_loader,
    make_model, sha256, write_inputs, write_predictions,
)
from pcbert_kla_clean.metrics import compute_binary_metrics
from pcbert_kla_clean.stacked_development import grouped_splits


def fold_rows(train_rows, validation_rows, outer_train, held_idx, inner_train,
              inner_stop):
    """Build disjoint inner-selection and outer-refit record sets."""
    outer_train = np.asarray(outer_train)
    held_idx = np.asarray(held_idx)
    inner_train = np.asarray(inner_train)
    inner_stop = np.asarray(inner_stop)
    if (set(outer_train) & set(held_idx) or
            set(inner_train) & set(inner_stop) or
            set(inner_train) | set(inner_stop) != set(outer_train)):
        raise ValueError("Fold indices overlap or do not cover outer training rows")
    return (
        [train_rows[int(index)] for index in inner_train],
        [train_rows[int(index)] for index in inner_stop],
        [train_rows[int(index)] for index in outer_train],
        [train_rows[int(index)] for index in held_idx] + validation_rows,
    )


def fit(author, proteins, train_rows, score_rows, output_dir: Path, device,
        seed: int, max_epochs: int, patience: int, batch_size: int, workers: int,
        fixed_epochs: int | None = None):
    author.set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=False)
    data_dir = write_inputs(output_dir, {"train": train_rows, "validation": score_rows},
                            proteins, ("train", "validation"))
    train_loader = make_loader(author, data_dir, "train", batch_size, workers, True)
    score_loader = make_loader(author, data_dir, "validation", batch_size, workers, False)
    model = make_model(author, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    scheduler = author.WarmupCosineLR(optimizer, warmup_epochs=5,
                                      max_epochs=max_epochs, min_lr=1e-6)
    ema = author.ModelEMA(model, decay=0.995, device=device)
    best_epoch, best_ap, stale = 0, -1.0, 0
    limit = fixed_epochs if fixed_epochs is not None else max_epochs
    if limit < 1 or limit > max_epochs:
        raise ValueError("Invalid fixed epoch")
    for epoch in range(1, limit + 1):
        author.run_epoch(model, train_loader, device, optimizer=optimizer,
                         scaler=scaler, ema=ema)
        scheduler.step()
        if fixed_epochs is not None:
            continue
        current = {name: value.detach().clone() for name, value in model.state_dict().items()}
        ema.load_to(model, device=device)
        _, labels, scores = author.run_epoch(model, score_loader, device)
        model.load_state_dict(current)
        ap = compute_binary_metrics(labels, scores)["AUPRC"]
        if ap > best_ap + 0.002:
            best_epoch, best_ap, stale = epoch, ap, 0
        else:
            stale += 1
        print(f"inner epoch={epoch} AUPRC={ap:.5f} best_epoch={best_epoch}", flush=True)
        if stale >= patience:
            break
    if fixed_epochs is None:
        result = (best_epoch, best_ap)
    else:
        ema.load_to(model, device=device)
        torch.save({name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()},
                   output_dir / "transkla_best.pt")
        _, labels, scores = author.run_epoch(model, score_loader, device)
        if (len(scores) != len(score_rows) or
                not np.array_equal(np.asarray(labels).astype(int),
                                   [int(row["label"]) for row in score_rows]) or
                not np.isfinite(scores).all()):
            raise ValueError("Invalid or misordered fold predictions")
        result = np.asarray(scores, dtype=np.float64)
    del model, optimizer, ema
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run(args):
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("A CUDA compute node is required")
    if min(args.epochs, args.patience, args.batch_size) < 1 or args.workers < 0:
        raise ValueError("Invalid training configuration")
    author = load_author(args.author_source)
    partitions, proteins, manifest_hash = load_lock(args.lock, args.proteome)
    train_rows, validation_rows = partitions["train"], partitions["validation"]
    labels = np.asarray([int(row["label"]) for row in train_rows])
    groups = np.asarray([row["homology_group"] for row in train_rows])
    folds = grouped_splits(labels, groups, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    validation_scores = np.zeros(len(validation_rows), dtype=np.float64)
    oof_scores = np.full(len(train_rows), np.nan, dtype=np.float64)
    fold_reports = []
    started = time.monotonic()
    for fold_number, (outer_train, held_idx) in enumerate(folds, 1):
        inner_train_local, inner_stop_local = grouped_splits(
            labels[outer_train], groups[outer_train], args.seed + fold_number,
        )[0]
        inner_train = outer_train[inner_train_local]
        inner_stop = outer_train[inner_stop_local]
        inner_rows, stop_rows, refit_rows, score_rows = fold_rows(
            train_rows, validation_rows, outer_train, held_idx, inner_train, inner_stop,
        )
        fold_dir = args.output_dir / f"fold_{fold_number}"
        training_seed = args.seed + 100 * fold_number + 1
        print(f"Fold {fold_number}/5: selecting TransKla epoch", flush=True)
        epoch, inner_ap = fit(
            author, proteins, inner_rows, stop_rows, fold_dir / "selection", device,
            training_seed, args.epochs, args.patience, args.batch_size, args.workers,
        )
        print(f"Fold {fold_number}/5: refitting TransKla for {epoch} epochs", flush=True)
        scores = fit(
            author, proteins, refit_rows, score_rows, fold_dir / "refit", device,
            training_seed, args.epochs, args.patience, args.batch_size, args.workers,
            fixed_epochs=epoch,
        )
        oof_scores[held_idx] = scores[:len(held_idx)]
        validation_scores += scores[len(held_idx):] / len(folds)
        write_predictions(fold_dir / "validation_predictions.csv", validation_rows,
                          scores[len(held_idx):])
        fold_reports.append({"fold": fold_number, "training_seed": training_seed,
                             "outer_train_rows": len(outer_train),
                             "inner_selection_rows": len(inner_stop),
                             "selected_epoch": epoch, "inner_auprc": inner_ap})
        print(json.dumps(fold_reports[-1], sort_keys=True), flush=True)
    if not np.isfinite(oof_scores).all() or not np.isfinite(validation_scores).all():
        raise FloatingPointError("Incomplete ensemble predictions")
    validation_labels = np.asarray([int(row["label"]) for row in validation_rows])
    metrics = compute_binary_metrics(validation_labels, validation_scores, 0.5)
    write_predictions(args.output_dir / "validation_predictions.csv", validation_rows,
                      validation_scores)
    np.savez_compressed(args.output_dir / "oof_scores.npz", scores=oof_scores,
                        labels=labels, groups=groups)
    report = {
        "model": "TransKla author architecture, five-fold protein-grouped ensemble",
        "author_commit": AUTHOR_COMMIT, "author_source_sha256": AUTHOR_SOURCE_SHA256,
        "lock_sha256": sha256(args.lock), "manifest_sha256": manifest_hash,
        "proteome_sha256": sha256(args.proteome), "seed": args.seed,
        "training_rows": len(train_rows), "validation_rows": len(validation_rows),
        "outer_folds": 5, "ensemble_size": 5, "folds": fold_reports,
        "epoch_selection": "inner grouped training fold AUPRC, min_delta=0.002",
        "training": "author AdamW/EMA/warmup-cosine/augmentation, fixed-epoch refit",
        "epochs_cap": args.epochs, "patience": args.patience,
        "batch_size": args.batch_size, "decision_threshold": 0.5,
        "validation_metrics": metrics, "heldout_test_scored": False,
        "wall_seconds": time.monotonic() - started,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"validation_metrics": metrics, "ensemble_size": 5}, indent=2),
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-source", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--proteome", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
