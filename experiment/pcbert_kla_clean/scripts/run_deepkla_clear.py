"""Train DeepKla-aligned architecture on locked CLEAR-Lactyl development data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pcbert_kla_clean.clear_lactyl import load_clear_lactyl  # noqa: E402
from pcbert_kla_clean.deepkla import DeepKlaAligned, encode_windows  # noqa: E402
from pcbert_kla_clean.metrics import compute_binary_metrics  # noqa: E402
from pcbert_kla_clean.utils import set_seed  # noqa: E402


def evaluate(model: DeepKlaAligned, loader: DataLoader, device: torch.device) -> tuple[float, np.ndarray]:
    model.eval()
    scores: list[np.ndarray] = []
    sum_loss = 0.0
    count = 0
    with torch.inference_mode():
        for tokens, labels in loader:
            probabilities = model(tokens.to(device))
            labels = labels.to(device)
            sum_loss += torch.nn.functional.mse_loss(
                probabilities, labels, reduction="sum",
            ).item()
            count += len(labels)
            scores.append(probabilities.cpu().numpy())
    return sum_loss / count, np.concatenate(scores)


def run(args: argparse.Namespace) -> dict:
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise ValueError("Epochs, patience, and batch size must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    set_seed(args.seed)

    development, validation_indices, _ = load_clear_lactyl(args.clear_lock)
    train_count = int(validation_indices[0])
    if not np.array_equal(validation_indices, np.arange(train_count, len(development.records))):
        raise ValueError("CLEAR-Lactyl validation rows are not a contiguous suffix")
    windows = [record.sequence for record in development.records]
    labels = torch.tensor([record.label for record in development.records], dtype=torch.float32)
    tokens = encode_windows(windows)
    train_loader = DataLoader(
        TensorDataset(tokens[:train_count], labels[:train_count]),
        batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed),
    )
    validation_loader = DataLoader(
        TensorDataset(tokens[train_count:], labels[train_count:]),
        batch_size=max(args.batch_size, 256), shuffle=False,
    )
    y_validation = labels[train_count:].numpy().astype(int)
    model = DeepKlaAligned().to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9, eps=1e-7)

    args.output_dir.mkdir(parents=True)
    history = []
    best_auprc = -math.inf
    best_epoch = 0
    stale_epochs = 0
    best_scores = None
    best_metrics = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for batch_tokens, batch_labels in train_loader:
            batch_tokens = batch_tokens.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            scores = model(batch_tokens)
            loss = torch.nn.functional.mse_loss(scores, batch_labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_labels)
            total_rows += len(batch_labels)

        validation_loss, validation_scores = evaluate(model, validation_loader, device)
        if not np.isfinite(validation_scores).all():
            raise FloatingPointError(f"Non-finite validation scores at epoch {epoch}")
        metrics = {key: float(value) for key, value in
                   compute_binary_metrics(y_validation, validation_scores).items()}
        record = {"epoch": epoch, "train_loss": total_loss / total_rows,
                  "validation_loss": validation_loss, **metrics}
        history.append(record)
        print(json.dumps(record), flush=True)
        if metrics["AUPRC"] > best_auprc:
            best_auprc = metrics["AUPRC"]
            best_epoch = epoch
            best_scores = validation_scores.copy()
            best_metrics = metrics
            stale_epochs = 0
            torch.save(model.state_dict(), args.output_dir / "best_model.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    assert best_scores is not None and best_metrics is not None
    with (args.output_dir / "validation_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("accession", "position_1based", "label", "score"))
        for record, score in zip(development.records[train_count:], best_scores):
            writer.writerow((record.name, record.position, record.label, float(score)))
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    report = {
        "architecture": "deepkla_aligned_pytorch",
        "source": "linDing-group/DeepKla d9f0f5ef987cd65f1562f1ba312e5608ddd44f9e",
        "split": "CLEAR-Lactyl locked train/validation; internal test unused",
        "seed": args.seed,
        "train_rows": train_count,
        "validation_rows": len(y_validation),
        "selected_epoch": best_epoch,
        "selection_metric": "validation_AUPRC",
        "loss": "MSE",
        "optimizer": "RMSprop(lr=0.001, alpha=0.9, eps=1e-7)",
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "patience": args.patience,
        "validation_metrics": best_metrics,
        "limitations": [
            "PyTorch GRU and RMSprop are not numerically identical to legacy Keras 2.0.6.",
            "The original script selects via validation loss; this benchmark selects via AUPRC.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clear-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
