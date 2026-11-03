"""Retrain HybridKla's eight-branch architecture on a locked source benchmark.

The authors' published row-wise CV is replaced by protein-grouped out-of-fold
stacking, and checkpoints are selected on the locked validation partition.
This is architecture-aligned retraining, not a byte-for-byte code reproduction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from pcbert_kla_clean.hybridkla_features import encode_features, fit_gps_reference, load_resources
from pcbert_kla_clean.hybridkla_models import (
    FeatureDNN, MetaDNN, SequenceLSTM, average_precision,
    encode_lstm_sequences, fit_predictor, predict_scores,
)
from run_hybridkla_adapted import BRANCHES, RESOURCE_SHA256, evaluate_and_save, partition_labels, sha256


class TrainableEsmBranch(nn.Module):
    def __init__(self, model_name: str, revision: str):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(model_name, revision=revision)
        width = int(self.encoder.config.hidden_size)
        self.dense = nn.Linear(width, 512)
        self.norm = nn.LayerNorm(512)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(512, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                special_mask: torch.Tensor) -> torch.Tensor:
        states = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        projected = self.norm(self.dense(states))
        residue_mask = (attention_mask.bool() & ~special_mask.bool()).unsqueeze(-1)
        if not residue_mask.any(dim=1).all():
            raise ValueError("ESM2 received a sequence without residue tokens")
        pooled = (projected * residue_mask).sum(dim=1) / residue_mask.sum(dim=1)
        return self.out(self.dropout(self.activation(pooled))).squeeze(-1)


def tokenize_sequences(sequences: list[str], tokenizer) -> tuple[torch.Tensor, ...]:
    tokenized = tokenizer(sequences, padding=True, truncation=False, return_tensors="pt")
    ids = tokenized["input_ids"]
    if ids.shape[1] > 53:
        raise ValueError("Unexpected ESM2 token length for 51-residue windows")
    special = torch.zeros_like(ids, dtype=torch.bool)
    for token_id in tokenizer.all_special_ids:
        special |= ids == token_id
    if not (tokenized["attention_mask"].bool() & ~special).any(dim=1).all():
        raise ValueError("ESM2 tokenizer removed every residue")
    return ids, tokenized["attention_mask"], special


def esm_scores(model: nn.Module, tokens: tuple[torch.Tensor, ...],
               device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(tokens[0]), batch_size):
            batch = (array[start:start + batch_size].to(device) for array in tokens)
            with (torch.autocast(device_type="cuda", dtype=torch.float16)
                  if device.type == "cuda" else nullcontext()):
                logits = model(*batch)
            result.append(torch.sigmoid(logits.float()).cpu().numpy())
    scores = np.concatenate(result).astype(np.float32)
    if not np.isfinite(scores).all():
        raise FloatingPointError("Nonfinite ESM2 predictions")
    return scores


def train_esm_fold(model_name: str, revision: str, tokenizer,
                   train_rows: list[dict[str, str]], validation_rows: list[dict[str, str]],
                   *, device: torch.device, epochs: int, batch_size: int,
                   patience: int, seed: int, checkpoint: Path) -> dict:
    torch.manual_seed(seed)
    model = TrainableEsmBranch(model_name, revision).to(device)
    train_tokens = tokenize_sequences([row["window_51aa"] for row in train_rows], tokenizer)
    val_tokens = tokenize_sequences([row["window_51aa"] for row in validation_rows], tokenizer)
    labels = torch.tensor(partition_labels([{"Label": row["label"]} for row in train_rows]),
                          dtype=torch.float32)
    data = TensorDataset(*train_tokens, labels)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)
    steps = max(1, epochs * len(loader))
    warmup = min(150, max(0, steps // 10))

    def rate(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    criterion = nn.BCEWithLogitsLoss()
    val_labels = partition_labels([{"Label": row["label"]} for row in validation_rows])
    best_ap, selected_epoch, completed = -1.0, 0, 0
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, epochs + 1):
        model.train()
        for ids, mask, special, y in loader:
            ids, mask, special, y = (tensor.to(device) for tensor in (ids, mask, special, y))
            optimizer.zero_grad(set_to_none=True)
            with (torch.autocast(device_type="cuda", dtype=torch.float16)
                  if device.type == "cuda" else nullcontext()):
                logits = model(ids, mask, special)
                loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite ESM2 training loss")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        completed = epoch
        scores = esm_scores(model, val_tokens, device, batch_size)
        ap = average_precision(val_labels, scores)
        print(f"ESM2 epoch {epoch}/{epochs}: validation AUPRC {ap:.4f}", flush=True)
        if ap > best_ap:
            best_ap, selected_epoch = ap, epoch
            torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()},
                       checkpoint)
        if patience and epoch - selected_epoch >= patience:
            break
    if not checkpoint.is_file():
        raise RuntimeError("No ESM2 checkpoint was selected")
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    report = {"selected_epoch": selected_epoch, "completed_epochs": completed,
              "validation_auprc": best_ap,
              "wall_seconds": round(time.monotonic() - started, 1),
              "peak_cuda_gb": (round(torch.cuda.max_memory_allocated(device) / 1e9, 3)
                               if device.type == "cuda" else None)}
    return {"model": model, "validation_tokens": val_tokens, "report": report}


def grouped_folds(rows: list[dict[str, str]], folds: int, seed: int) -> list[list[int]]:
    if folds < 2:
        raise ValueError("At least two grouped folds are required")
    by_group = {}
    for index, row in enumerate(rows):
        by_group.setdefault(row["group_id"], []).append(index)
    if len(by_group) < folds:
        raise ValueError("Too few protein groups for grouped folds")
    rng = random.Random(seed)
    groups = sorted(by_group)
    rng.shuffle(groups)
    groups.sort(key=lambda group: -len(by_group[group]))
    fold_indices = [[] for _ in range(folds)]
    fold_pos = [0] * folds
    fold_size = [0] * folds
    for group in groups:
        indices = by_group[group]
        positives = sum(rows[index]["label"] == "1" for index in indices)
        choice = min(range(folds), key=lambda fold: (fold_size[fold], fold_pos[fold], fold))
        fold_indices[choice].extend(indices)
        fold_size[choice] += len(indices)
        fold_pos[choice] += positives
    for indices in fold_indices:
        indices.sort()
        labels = {rows[index]["label"] for index in indices}
        if labels != {"0", "1"}:
            raise ValueError("Grouped fold lacks one class")
    if sorted(index for indices in fold_indices for index in indices) != list(range(len(rows))):
        raise AssertionError("Grouped folds do not partition training rows")
    return fold_indices


def read_lock(lock_path: Path) -> tuple[dict[str, list[dict[str, str]]], dict]:
    audit = json.loads(lock_path.read_text())
    manifest = lock_path.with_name("model_manifest.csv")
    study_lock = "output_sha256" in audit
    expected_sha = (audit["output_sha256"]["model_manifest.csv"] if study_lock
                    else audit["model_manifest_sha256"])
    if sha256(manifest) != expected_sha:
        raise ValueError("HybridKla source manifest does not match lock")
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    partitions = {name: [] for name in ("train", "validation", "internal_test", "external_test")}
    accessions, groups = {}, {}
    windows = set()
    for row in rows:
        partition = row["partition"]
        if partition not in partitions:
            raise ValueError("Invalid source partition")
        if study_lock:
            expected = ("S7", "Mus musculus") if partition == "external_test" else (
                "S1", "Homo sapiens")
            if (row["source_table"], row["species"]) != expected:
                raise ValueError("HybridKla study source does not match partition")
            row["group_id"] = row["homology_group"]
            row["source_site_id"] = f'{row["protein_id"]}:{row["position_1based"]}'
            identifier = row["protein_id"]
        else:
            if row["group_id"] != row["accession"]:
                raise ValueError("Invalid source accession group")
            identifier = row["accession"]
        if accessions.setdefault(identifier, partition) != partition:
            raise ValueError("Protein crosses source partitions")
        if groups.setdefault(row["group_id"], partition) != partition:
            raise ValueError("Homology group crosses source partitions")
        if row["window_51aa"] in windows:
            raise ValueError("Exact window repeated in source lock")
        windows.add(row["window_51aa"])
        partitions[partition].append(row)
    for name, group_rows in partitions.items():
        if name == "external_test" and not study_lock:
            continue
        expected = (audit["partitions_before_training_resampling"][name] if study_lock
                    else audit["partitions"][name])
        if ((name != "train" or not study_lock) and
            (len(group_rows) != expected["rows"] or
             sum(r["label"] == "1" for r in group_rows) != expected["positives"])):
            raise ValueError(f"Source lock count mismatch in {name}")
        if not group_rows:
            raise ValueError(f"Empty partition: {name}")
    return partitions, audit


def branch_arrays(name: str, rows: list[dict[str, str]], resources,
                  gps_reference: np.ndarray | None):
    sequences = [row["window_51aa"] for row in rows]
    if name == "LSTM":
        return encode_lstm_sequences(sequences)
    if gps_reference is None:
        raise ValueError("Feature branch lacks its training-only GPS reference")
    return encode_features(sequences, resources, gps_reference)[name]


def run(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device is unavailable")
    if min(args.esm_epochs, args.lstm_epochs, args.feature_epochs, args.meta_epochs,
           args.esm_batch_size, args.lstm_batch_size, args.feature_batch_size) < 1:
        raise ValueError("Epoch and batch settings must be positive")
    for name, expected in RESOURCE_SHA256.items():
        if sha256(args.resource_dir / name) != expected:
            raise ValueError(f"Official HybridKla resource changed: {name}")
    partitions, audit = read_lock(args.benchmark_lock)
    train_rows, validation_rows = partitions["train"], partitions["validation"]
    if args.pilot:
        print("One-fold ESM2 timing pilot; internal test is not scored.", flush=True)
    else:
        print("Full architecture-aligned training; internal test is scored only after selection.", flush=True)
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    revision = HfApi().model_info(args.esm_model).sha
    if not revision or len(revision) != 40:
        raise ValueError("Could not pin ESM2 model revision")
    tokenizer = AutoTokenizer.from_pretrained(args.esm_model, revision=revision)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    folds = grouped_folds(train_rows, 5, args.seed)
    all_indices = set(range(len(train_rows)))
    resources = load_resources(args.resource_dir / "top10.txt", args.resource_dir / "BLOSUM62R.txt")
    val_labels = np.asarray([int(row["label"]) for row in validation_rows])
    fold_reports = {}
    if args.pilot:
        holdout = set(folds[0])
        base_rows = [train_rows[index] for index in sorted(all_indices - holdout)]
        result = train_esm_fold(args.esm_model, revision, tokenizer, base_rows, validation_rows,
                                device=device, epochs=1, batch_size=args.esm_batch_size,
                                patience=0, seed=args.seed,
                                checkpoint=checkpoint_dir / "ESM2_fold_0.pth")
        fold_reports["ESM2_fold_0"] = result["report"]
        del result
        test_metrics = {}
    else:
        oof = np.empty((len(train_rows), len(BRANCHES)), dtype=np.float32)
        val_scores = np.empty((len(validation_rows), len(BRANCHES)), dtype=np.float32)
        for column, name in enumerate(BRANCHES):
            accumulated_val = np.zeros(len(validation_rows), dtype=np.float64)
            for fold, holdout_indices in enumerate(folds):
                holdout = set(holdout_indices)
                training_rows = [train_rows[index] for index in sorted(all_indices - holdout)]
                held_rows = [train_rows[index] for index in holdout_indices]
                checkpoint = checkpoint_dir / f"{name}_fold_{fold}.pth"
                seed = args.seed + column * 100 + fold
                if name == "ESM2":
                    result = train_esm_fold(args.esm_model, revision, tokenizer,
                                            training_rows, validation_rows,
                                            device=device, epochs=args.esm_epochs,
                                            batch_size=args.esm_batch_size,
                                            patience=args.patience, seed=seed,
                                            checkpoint=checkpoint)
                    model = result["model"]
                    scores = esm_scores(model, tokenize_sequences(
                        [row["window_51aa"] for row in held_rows], tokenizer),
                        device, args.esm_batch_size)
                    validation_scores = esm_scores(model, result["validation_tokens"],
                                                   device, args.esm_batch_size)
                    fold_reports[f"{name}_fold_{fold}"] = result["report"]
                    del result, model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                else:
                    gps_reference = fit_gps_reference([row["window_51aa"] for row in training_rows])
                    if name == "GPS":
                        np.save(output / f"GPS_fold_{fold}_reference.npy", gps_reference)
                    train_values = branch_arrays(name, training_rows, resources, gps_reference)
                    held_values = branch_arrays(name, held_rows, resources, gps_reference)
                    validation_values = branch_arrays(name, validation_rows, resources, gps_reference)
                    model = SequenceLSTM() if name == "LSTM" else FeatureDNN(name)
                    epochs = args.lstm_epochs if name == "LSTM" else args.feature_epochs
                    batch_size = args.lstm_batch_size if name == "LSTM" else args.feature_batch_size
                    lr = 1e-4 if name == "LSTM" else 1e-3
                    result = fit_predictor(model, train_values,
                                           np.asarray([int(r["label"]) for r in training_rows]),
                                           validation_values, val_labels,
                                           device=device, epochs=epochs, batch_size=batch_size,
                                           learning_rate=lr, patience=args.patience, seed=seed)
                    model = result.model
                    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()},
                               checkpoint)
                    scores = predict_scores(model, held_values, device,
                                            batch_size=128 if name == "LSTM" else 1024)
                    validation_scores = predict_scores(model, validation_values, device,
                                                       batch_size=128 if name == "LSTM" else 1024)
                    fold_reports[f"{name}_fold_{fold}"] = {
                        "selected_epoch": result.selected_epoch,
                        "validation_auprc": result.validation_auprc,
                    }
                    del model, result
                oof[holdout_indices, column] = scores
                accumulated_val += validation_scores / len(folds)
                print(f"{name} fold {fold + 1}/5 selected", flush=True)
            val_scores[:, column] = accumulated_val.astype(np.float32)
        if not np.isfinite(oof).all() or not np.isfinite(val_scores).all():
            raise FloatingPointError("Nonfinite out-of-fold branch scores")
        meta_result = fit_predictor(MetaDNN(), oof,
                                    np.asarray([int(r["label"]) for r in train_rows]),
                                    val_scores, val_labels, device=device,
                                    epochs=args.meta_epochs, batch_size=args.feature_batch_size,
                                    learning_rate=1e-3, patience=args.patience,
                                    seed=args.seed + 1000)
        torch.save({key: value.detach().cpu() for key, value in meta_result.model.state_dict().items()},
                   checkpoint_dir / "META.pth")
        fold_reports["META"] = {"selected_epoch": meta_result.selected_epoch,
                                "validation_auprc": meta_result.validation_auprc}
        test_metrics = {}
        for test_name in ("internal_test", "external_test"):
            test_rows = partitions[test_name]
            if not test_rows:
                continue
            test_branch_scores = np.empty((len(test_rows), len(BRANCHES)), dtype=np.float32)
            for column, name in enumerate(BRANCHES):
                accumulated = np.zeros(len(test_rows), dtype=np.float64)
                for fold, holdout_indices in enumerate(folds):
                    checkpoint = checkpoint_dir / f"{name}_fold_{fold}.pth"
                    if name == "ESM2":
                        model = TrainableEsmBranch(args.esm_model, revision).to(device)
                        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
                        scores = esm_scores(model, tokenize_sequences(
                            [row["window_51aa"] for row in test_rows], tokenizer),
                            device, args.esm_batch_size)
                        del model
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                    else:
                        gps_reference = fit_gps_reference([
                            train_rows[index]["window_51aa"]
                            for index in sorted(all_indices - set(holdout_indices))])
                        test_values = branch_arrays(name, test_rows, resources, gps_reference)
                        model = (SequenceLSTM() if name == "LSTM" else FeatureDNN(name)).to(device)
                        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
                        scores = predict_scores(model, test_values, device,
                                                batch_size=128 if name == "LSTM" else 1024)
                        del model
                    accumulated += scores / len(folds)
                test_branch_scores[:, column] = accumulated.astype(np.float32)
            final_scores = predict_scores(meta_result.model, test_branch_scores, device)
            score_rows = [{"Site_ID": row["source_site_id"], "Label": row["label"]}
                          for row in test_rows]
            test_metrics[test_name] = evaluate_and_save(test_name, score_rows, final_scores, output)
            print(f"HybridKla architecture {test_name}:", json.dumps(test_metrics[test_name]), flush=True)
    report = {
        "method": "HybridKla architecture-aligned retraining; grouped protocol, not exact authors' CV",
        "mode": "one-fold-training-only-timing" if args.pilot else "locked-test-evaluation",
        "benchmark_status": audit.get("benchmark_status", audit.get("protocol")),
        "benchmark_manifest_sha256": (audit["output_sha256"]["model_manifest.csv"]
                                      if "output_sha256" in audit else audit["model_manifest_sha256"]),
        "benchmark_lock_sha256": sha256(args.benchmark_lock),
        "model_name": args.esm_model, "model_revision": revision,
        "seed": args.seed, "folds": 5, "threshold": 0.5,
        "selection_metric": "validation AUPRC",
        "resource_sha256": RESOURCE_SHA256,
        "hyperparameters": {"esm_epochs": args.esm_epochs,
                            "lstm_epochs": args.lstm_epochs,
                            "feature_epochs": args.feature_epochs,
                            "meta_epochs": args.meta_epochs,
                            "esm_batch_size": args.esm_batch_size,
                            "lstm_batch_size": args.lstm_batch_size,
                            "feature_batch_size": args.feature_batch_size,
                            "patience": args.patience},
        "fold_reports": fold_reports,
        "test_metrics": test_metrics,
        "wall_seconds": round(time.monotonic() - started, 1),
    }
    (output / "run_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-lock", type=Path, required=True)
    parser.add_argument("--resource-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--esm-model", default="facebook/esm2_t30_150M_UR50D")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--esm-epochs", type=int, default=40)
    parser.add_argument("--lstm-epochs", type=int, default=30)
    parser.add_argument("--feature-epochs", type=int, default=100)
    parser.add_argument("--meta-epochs", type=int, default=50)
    parser.add_argument("--esm-batch-size", type=int, default=32)
    parser.add_argument("--lstm-batch-size", type=int, default=32)
    parser.add_argument("--feature-batch-size", type=int, default=2048)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
