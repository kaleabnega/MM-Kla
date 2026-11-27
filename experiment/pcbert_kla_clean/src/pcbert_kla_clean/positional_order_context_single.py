"""Single-model development evaluation of the order-context positional predictor."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from pcbert_kla_clean.clear_lactyl import load_clear_lactyl
from pcbert_kla_clean.metrics import compute_binary_metrics
from pcbert_kla_clean.positional_global_development import load_verified_proteins
from pcbert_kla_clean.positional_order_context_development import (
    OrderContextPositionalBranch, fit_model, protein_tokens,
)
from pcbert_kla_clean.source_study import sha256
from pcbert_kla_clean.stacked_development import (
    grouped_splits, read_training_groups, sequence_tokens,
)


def run(lock: Path, proteome: Path, output_dir: Path, device: torch.device,
        seed: int = 42, max_epochs: int = 30, patience: int = 5,
        model_factory=OrderContextPositionalBranch,
        model_name: str = "single order-context positional predictor") -> dict:
    if seed < 0 or max_epochs < 1 or patience < 1:
        raise ValueError("Invalid training configuration")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("A CUDA compute node is required")

    split, validation_idx, _ = load_clear_lactyl(lock, add_cksaap=False)
    train_count = int(validation_idx[0])
    labels = split.labels.astype(int)
    groups = read_training_groups(lock, split, train_count)
    train_idx, select_idx = grouped_splits(labels[:train_count], groups, seed)[0]
    if set(groups[train_idx]) & set(groups[select_idx]):
        raise ValueError("Epoch-selection groups overlap training groups")
    proteins = load_verified_proteins(proteome, split.records)
    bank = {name: protein_tokens(sequence) for name, sequence in proteins.items()}
    local_tokens = sequence_tokens(split.sequences)

    output_dir.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    training_seed = seed + 101
    selected_epoch, selection_auprc = fit_model(
        local_tokens, split.records, bank, labels, train_idx, select_idx,
        device, training_seed, max_epochs, patience, selection_idx=select_idx,
        model_factory=model_factory,
    )
    print(f"Selected epoch {selected_epoch}; refitting on {train_count} training rows",
          flush=True)
    scores = fit_model(
        local_tokens, split.records, bank, labels, np.arange(train_count),
        validation_idx, device, training_seed, max_epochs, patience,
        fixed_epochs=selected_epoch, checkpoint_path=output_dir / "model.pt",
        model_factory=model_factory,
    )
    if len(scores) != len(validation_idx) or not np.isfinite(scores).all():
        raise FloatingPointError("Invalid validation predictions")
    metrics = compute_binary_metrics(labels[validation_idx], scores, 0.5)
    with (output_dir / "validation_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("accession", "position_1based", "label", "score"))
        for index, score in zip(validation_idx, scores):
            record = split.records[int(index)]
            writer.writerow((record.name, record.position, record.label, float(score)))
    report = {
        "model": model_name,
        "seed": seed, "training_seed": training_seed,
        "lock_sha256": sha256(lock), "proteome_sha256": sha256(proteome),
        "training_rows": train_count, "selection_train_rows": len(train_idx),
        "selection_rows": len(select_idx), "validation_rows": len(validation_idx),
        "selected_epoch": selected_epoch, "selection_auprc": selection_auprc,
        "validation_metrics": metrics, "decision_threshold": 0.5,
        "ensemble_size": 1, "heldout_test_scored": False,
        "wall_seconds": time.monotonic() - start,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
