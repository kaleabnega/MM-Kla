"""Validation-only combination of contextual sequence and 27 descriptors."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pcbert_kla_clean.clear_lactyl import load_clear_lactyl
from pcbert_kla_clean.descriptor_ablation_development import load_contextual_reference
from pcbert_kla_clean.features import FEATURE_NAMES
from pcbert_kla_clean.metrics import compute_binary_metrics
from pcbert_kla_clean.source_study import sha256
from pcbert_kla_clean.stacked_development import grouped_splits, read_training_groups
from pcbert_kla_clean.utils import set_seed


class DescriptorPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(27, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != 27:
            raise ValueError("Descriptor predictor expects 27 features per row")
        return self.network(features).squeeze(-1)


def descriptor_features(split) -> np.ndarray:
    if split.feature_names[:27] != list(FEATURE_NAMES) or split.features.shape[1] != 1627:
        raise ValueError("Expected the 27 CLEAR-Lactyl descriptors before CKSAAP")
    return np.asarray(split.features[:, :27], dtype=np.float32)


def predict_scores(model: DescriptorPredictor, features: torch.Tensor,
                   indices: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    scores = []
    with torch.no_grad():
        for (batch,) in DataLoader(TensorDataset(features[indices]), batch_size=256):
            scores.extend(torch.sigmoid(model(batch.to(device))).cpu().tolist())
    return np.asarray(scores, dtype=np.float64)


def fit_descriptors(features: np.ndarray, labels: np.ndarray, train_idx: np.ndarray,
                    score_idx: np.ndarray, device: torch.device, seed: int,
                    max_epochs: int, patience: int,
                    selection_idx: np.ndarray | None = None,
                    fixed_epochs: int | None = None):
    if selection_idx is None and (fixed_epochs is None or not 1 <= fixed_epochs <= max_epochs):
        raise ValueError("Refit needs a selected epoch within the training schedule")
    set_seed(seed)
    scaler = MinMaxScaler()
    scaled = np.empty_like(features, dtype=np.float32)
    scaled[train_idx] = scaler.fit_transform(features[train_idx])
    scaled[score_idx] = scaler.transform(features[score_idx])
    if selection_idx is not None:
        scaled[selection_idx] = scaler.transform(features[selection_idx])
    tensor = torch.from_numpy(scaled)
    targets = torch.as_tensor(labels, dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(tensor[train_idx], targets[train_idx]),
                        batch_size=128, shuffle=True, generator=generator)
    model = DescriptorPredictor().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    best_epoch, best_ap, stale = 0, -1.0, 0
    epochs_to_run = max_epochs if selection_idx is not None else fixed_epochs
    for epoch in range(1, epochs_to_run + 1):
        model.train()
        for batch_features, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_features.to(device)), batch_labels.to(device))
            loss.backward()
            optimizer.step()
        if selection_idx is None:
            continue
        scores = predict_scores(model, tensor, selection_idx, device)
        ap = average_precision_score(labels[selection_idx], scores)
        if ap > best_ap:
            best_epoch, best_ap, stale = epoch, float(ap), 0
        else:
            stale += 1
        print(f"Descriptor inner epoch={epoch} AUPRC={ap:.5f}", flush=True)
        if stale >= patience:
            break
    result = ((best_epoch, best_ap) if selection_idx is not None else
              predict_scores(model, tensor, score_idx, device))
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run(lock_path: Path, reference_dir: Path, output_dir: Path,
        device: torch.device, seed: int, max_epochs: int = 30,
        patience: int = 5) -> dict:
    if seed < 0 or max_epochs < 1 or patience < 1:
        raise ValueError("Invalid seed, epoch count, or patience")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    split, validation_idx, _ = load_clear_lactyl(lock_path, add_cksaap=True)
    train_count = int(validation_idx[0])
    if not np.array_equal(validation_idx, np.arange(train_count, len(split.records))):
        raise ValueError("Expected contiguous training and validation partitions")
    groups = read_training_groups(lock_path, split, train_count)
    folds = grouped_splits(split.labels[:train_count], groups, seed)
    contextual_oof, contextual_validation, full_stack_validation, reference_hashes = (
        load_contextual_reference(reference_dir, lock_path, seed,
                                  split, validation_idx, groups)
    )
    features = descriptor_features(split)
    labels = split.labels.astype(int)
    oof = np.full(train_count, np.nan, dtype=np.float64)
    validation = np.zeros(len(validation_idx), dtype=np.float64)
    fold_reports = []
    output_dir.mkdir(parents=True, exist_ok=False)
    for fold_number, (outer_train, held_idx) in enumerate(folds, 1):
        inner_train_local, inner_stop_local = grouped_splits(
            labels[outer_train], groups[outer_train], seed + fold_number,
        )[0]
        inner_train = outer_train[inner_train_local]
        inner_stop = outer_train[inner_stop_local]
        if set(groups[inner_train]) & set(groups[inner_stop]):
            raise ValueError("Inner checkpoint groups overlap")
        branch_seed = seed + 100 * fold_number + 2
        print(f"Fold {fold_number}/5: selecting descriptor epoch", flush=True)
        selected_epoch, inner_ap = fit_descriptors(
            features, labels, inner_train, inner_stop, device, branch_seed,
            max_epochs, patience, selection_idx=inner_stop,
        )
        score_idx = np.r_[held_idx, validation_idx]
        print(f"Fold {fold_number}/5: refitting descriptor predictor", flush=True)
        scores = fit_descriptors(
            features, labels, outer_train, score_idx, device, branch_seed,
            max_epochs, patience, fixed_epochs=selected_epoch,
        )
        oof[held_idx] = scores[:len(held_idx)]
        validation += scores[len(held_idx):] / len(folds)
        fold_reports.append({"fold": fold_number, "outer_train_rows": len(outer_train),
                             "outer_held_rows": len(held_idx), "inner_stop_rows": len(inner_stop),
                             "selected_epoch": selected_epoch, "inner_auprc": inner_ap})
    if not np.isfinite(oof).all() or not np.isfinite(validation).all():
        raise FloatingPointError("Incomplete or nonfinite descriptor predictions")
    meta = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    meta.fit(np.column_stack((contextual_oof, oof)), labels[:train_count])
    combined = meta.predict_proba(
        np.column_stack((contextual_validation, validation)),
    )[:, 1]
    prediction_sets = {
        "contextual_positional": contextual_validation,
        "physicochemical_27_only": validation,
        "contextual_plus_physicochemical_27": combined,
        "reference_full_two_predictor": full_stack_validation,
    }
    metrics = {name: compute_binary_metrics(labels[validation_idx], values, threshold=0.5)
               for name, values in prediction_sets.items()}
    np.savez_compressed(output_dir / "oof_descriptor_scores.npz", scores=oof,
                        labels=labels[:train_count], groups=groups)
    with (output_dir / "validation_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["accession", "position_1based", "label", *prediction_sets])
        for row_number, index in enumerate(validation_idx):
            record = split.records[int(index)]
            writer.writerow([record.name, record.position, record.label,
                             *(float(values[row_number]) for values in prediction_sets.values())])
    report = {
        "seed": seed, "lock_sha256": sha256(lock_path),
        "training_rows": train_count, "validation_rows": len(validation_idx),
        "descriptor_names": list(FEATURE_NAMES), "descriptor_count": 27,
        "descriptor_model": "27-64-1 MLP with GELU and dropout 0.2",
        "contextual_reference": reference_hashes,
        "folds": fold_reports,
        "combiner": {"type": "logistic_regression", "C": 1.0,
                     "inputs": ["contextual_positional", "physicochemical_27_only"],
                     "coefficients": meta.coef_[0].tolist(),
                     "intercept": float(meta.intercept_[0])},
        "selection": "inner grouped training fold AUPRC only",
        "primary_metric": "validation AUPRC",
        "validation_metrics": metrics,
        "internal_test_scored": False,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
