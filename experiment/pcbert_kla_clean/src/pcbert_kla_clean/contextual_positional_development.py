"""Validation-only positional branch with frozen ProtBERT residue context."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from pcbert_kla_clean.backbones import load_encoder, load_tokenizer
from pcbert_kla_clean.clear_lactyl import load_clear_lactyl
from pcbert_kla_clean.metrics import compute_binary_metrics
from pcbert_kla_clean.source_study import sha256
from pcbert_kla_clean.stacked_development import (
    PositionalSequenceBranch, grouped_splits, read_training_groups, sequence_tokens,
)
from pcbert_kla_clean.utils import set_seed


MODEL_NAME = "Rostlab/prot_bert"


class ContextualPositionalBranch(PositionalSequenceBranch):
    def __init__(self, context_dim: int) -> None:
        super().__init__()
        self.context_projection = nn.Sequential(
            nn.LayerNorm(context_dim), nn.Linear(context_dim, 64), nn.Dropout(0.2),
        )

    def forward(self, tokens: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if context.shape[:2] != tokens.shape or context.shape[-1] != self.context_projection[0].normalized_shape[0]:
            raise ValueError("Context shape must be [batch, 51, context_dim]")
        positions = torch.arange(51, device=tokens.device)
        embedded = (self.residue(tokens) + self.position(positions)[None, :, :]
                    + self.context_projection(context.float()))
        return self.forward_embedded(embedded)


class IndexedWindows(Dataset):
    def __init__(self, tokens: torch.Tensor, context: torch.Tensor,
                 labels: np.ndarray, indices: np.ndarray) -> None:
        self.tokens = tokens
        self.context = context
        self.labels = torch.as_tensor(labels, dtype=torch.float32)
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        row = int(self.indices[index])
        return self.tokens[row], self.context[row], self.labels[row]


def frozen_context(sequences: list[str], device: torch.device,
                   batch_size: int = 16) -> torch.Tensor:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    tokenizer = load_tokenizer(MODEL_NAME, None)
    encoder = load_encoder(MODEL_NAME, None).to(device)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    width = int(encoder.config.hidden_size)
    features = torch.empty((len(sequences), 51, width), dtype=torch.float16)
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            chunk = sequences[start:start + batch_size]
            inputs = tokenizer([" ".join(sequence) for sequence in chunk],
                               return_tensors="pt", padding=True)
            if inputs["input_ids"].shape != (len(chunk), 53) or not torch.all(inputs["attention_mask"] == 1):
                raise ValueError("Expected 51 ProtBERT residue tokens plus CLS and SEP")
            outputs = encoder(**{key: value.to(device) for key, value in inputs.items()})
            residue_states = outputs.last_hidden_state[:, 1:52, :]
            if not torch.isfinite(residue_states).all():
                raise FloatingPointError("Nonfinite frozen ProtBERT embeddings")
            features[start:start + len(chunk)] = residue_states.to(
                device="cpu", dtype=torch.float16,
            )
            if start % (batch_size * 100) == 0:
                print(f"Frozen ProtBERT context: {start + len(chunk)}/{len(sequences)}", flush=True)
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return features


def predict_scores(model: ContextualPositionalBranch, tokens: torch.Tensor,
                   context: torch.Tensor, labels: np.ndarray,
                   indices: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    scores = []
    loader = DataLoader(IndexedWindows(tokens, context, labels, indices), batch_size=128)
    with torch.no_grad():
        for batch_tokens, batch_context, _ in loader:
            logits = model(batch_tokens.to(device), batch_context.to(device))
            scores.extend(torch.sigmoid(logits).cpu().tolist())
    return np.asarray(scores, dtype=np.float64)


def fit_contextual(tokens: torch.Tensor, context: torch.Tensor, labels: np.ndarray,
                   train_idx: np.ndarray, score_idx: np.ndarray,
                   device: torch.device, seed: int, max_epochs: int,
                   patience: int, selection_idx: np.ndarray | None = None,
                   fixed_epochs: int | None = None):
    if selection_idx is None and (fixed_epochs is None or not 1 <= fixed_epochs <= max_epochs):
        raise ValueError("Refit needs a selected epoch within the 30-epoch schedule")
    set_seed(seed)
    model = ContextualPositionalBranch(context.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(IndexedWindows(tokens, context, labels, train_idx),
                        batch_size=128, shuffle=True, generator=generator)
    best_epoch, best_ap, stale = 0, -1.0, 0
    epochs_to_run = max_epochs if selection_idx is not None else fixed_epochs
    for epoch in range(1, epochs_to_run + 1):
        model.train()
        for batch_tokens, batch_context, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_tokens.to(device), batch_context.to(device))
            loss = criterion(logits, batch_labels.to(device))
            loss.backward()
            optimizer.step()
        if selection_idx is None:
            continue
        scores = predict_scores(model, tokens, context, labels, selection_idx, device)
        ap = average_precision_score(labels[selection_idx], scores)
        if ap > best_ap:
            best_epoch, best_ap, stale = epoch, ap, 0
        else:
            stale += 1
        print(f"Contextual positional inner epoch={epoch} AUPRC={ap:.5f}", flush=True)
        if stale >= patience:
            break
    result = ((best_epoch, best_ap) if selection_idx is not None else
              predict_scores(model, tokens, context, labels, score_idx, device))
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def load_reference(reference_dir: Path, lock_path: Path, seed: int,
                   labels: np.ndarray, groups: np.ndarray,
                   split, validation_idx: np.ndarray):
    report_path = reference_dir / "report.json"
    prediction_path = reference_dir / "validation_predictions.csv"
    oof_path = reference_dir / "oof_branch_scores.npz"
    report = json.loads(report_path.read_text())
    if (report["seed"] != seed or report["lock_sha256"] != sha256(lock_path) or
            report["training_rows"] != len(groups) or
            report["validation_rows"] != len(validation_idx) or
            report["internal_test_scored"]):
        raise ValueError("Reference stack does not match seed, lock, or partition")
    with np.load(oof_path) as saved:
        reference_oof = saved["scores"]
        if (reference_oof.shape != (len(groups), 2) or
                not np.array_equal(saved["labels"], labels[:len(groups)]) or
                not np.array_equal(saved["groups"], groups)):
            raise ValueError("Reference OOF rows do not match the locked training data")
    with prediction_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(validation_idx):
        raise ValueError("Reference validation prediction count differs")
    for row, index in zip(rows, validation_idx):
        record = split.records[int(index)]
        if (row["accession"] != record.name or
                int(row["position_1based"]) != record.position or
                int(row["label"]) != record.label):
            raise ValueError("Reference validation row differs from locked data")
    reference_validation = np.asarray(
        [[float(row["protbert_cksaap"]), float(row["positional_sequence"]),
          float(row["oof_logistic_combination"])] for row in rows], dtype=np.float64,
    )
    if (not np.isfinite(reference_oof).all() or
            not np.isfinite(reference_validation).all()):
        raise FloatingPointError("Reference predictions contain nonfinite scores")
    return report_path, oof_path, prediction_path, reference_oof, reference_validation


def run(lock_path: Path, reference_dir: Path, output_dir: Path,
        device: torch.device, seed: int, max_epochs: int = 30,
        patience: int = 5) -> dict:
    if seed < 0 or max_epochs < 1 or patience < 1:
        raise ValueError("Invalid seed, epoch count, or patience")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    split, validation_idx, _ = load_clear_lactyl(lock_path, add_cksaap=True)
    train_count = int(validation_idx[0])
    if not np.array_equal(validation_idx, np.arange(train_count, train_count + len(validation_idx))):
        raise ValueError("Expected contiguous train and validation partitions")
    labels = split.labels.astype(int)
    groups = read_training_groups(lock_path, split, train_count)
    folds = grouped_splits(labels[:train_count], groups, seed)
    reference_paths = load_reference(reference_dir, lock_path, seed, labels,
                                     groups, split, validation_idx)
    report_path, oof_path, prediction_path, reference_oof, reference_validation = reference_paths
    tokens = sequence_tokens(split.sequences[:train_count + len(validation_idx)])
    output_dir.mkdir(parents=True, exist_ok=False)
    context = frozen_context(split.sequences[:len(tokens)], device)
    oof = np.full(train_count, np.nan, dtype=np.float64)
    validation = np.zeros(len(validation_idx), dtype=np.float64)
    fold_reports = []
    for fold_number, (outer_train, held_idx) in enumerate(folds, 1):
        inner_train_local, inner_stop_local = grouped_splits(
            labels[outer_train], groups[outer_train], seed + fold_number,
        )[0]
        inner_train = outer_train[inner_train_local]
        inner_stop = outer_train[inner_stop_local]
        if set(groups[inner_train]) & set(groups[inner_stop]):
            raise ValueError("Inner checkpoint groups overlap")
        branch_seed = seed + 100 * fold_number + 1
        print(f"Fold {fold_number}/5: selecting contextual positional epoch", flush=True)
        selected_epoch, inner_ap = fit_contextual(
            tokens, context, labels, inner_train, inner_stop, device,
            branch_seed, max_epochs, patience, selection_idx=inner_stop,
        )
        print(f"Fold {fold_number}/5: refitting contextual positional branch", flush=True)
        score_idx = np.r_[held_idx, validation_idx]
        scores = fit_contextual(
            tokens, context, labels, outer_train, score_idx, device,
            branch_seed, max_epochs, patience, fixed_epochs=selected_epoch,
        )
        oof[held_idx] = scores[:len(held_idx)]
        validation += scores[len(held_idx):] / len(folds)
        fold_reports.append({"fold": fold_number, "outer_train_rows": len(outer_train),
                             "outer_held_rows": len(held_idx), "inner_stop_rows": len(inner_stop),
                             "selected_epoch": selected_epoch, "inner_auprc": inner_ap})
    if not np.isfinite(oof).all() or not np.isfinite(validation).all():
        raise FloatingPointError("Incomplete or nonfinite contextual predictions")
    meta = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    meta.fit(np.column_stack((reference_oof[:, 0], oof)), labels[:train_count])
    np.savez_compressed(output_dir / "oof_contextual_scores.npz", scores=oof,
                        labels=labels[:train_count], groups=groups)
    scores = {
        "reference_positional": reference_validation[:, 1],
        "reference_two_predictor": reference_validation[:, 2],
        "contextual_positional": validation,
        "contextual_two_predictor": meta.predict_proba(
            np.column_stack((reference_validation[:, 0], validation)),
        )[:, 1],
    }
    results = {name: compute_binary_metrics(labels[validation_idx], values, threshold=0.5)
               for name, values in scores.items()}
    with (output_dir / "validation_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["accession", "position_1based", "label", *scores])
        for row_number, index in enumerate(validation_idx):
            record = split.records[int(index)]
            writer.writerow([record.name, record.position, record.label,
                             *(float(values[row_number]) for values in scores.values())])
    report = {
        "seed": seed, "lock_sha256": sha256(lock_path),
        "training_rows": train_count, "validation_rows": len(validation_idx),
        "model_name": MODEL_NAME, "encoder_layers": 30,
        "encoder_frozen": True, "context_cache_dtype": "float16",
        "context_projection_dim": 64,
        "feature_dim": split.features.shape[1], "folds": fold_reports,
        "reference": {"report": str(report_path), "report_sha256": sha256(report_path),
                      "oof_sha256": sha256(oof_path),
                      "validation_predictions_sha256": sha256(prediction_path)},
        "combiner": {"type": "logistic_regression", "C": 1.0,
                     "coefficients": meta.coef_[0].tolist(),
                     "intercept": float(meta.intercept_[0])},
        "selection": "inner grouped training fold AUPRC only",
        "primary_metric": "validation AUPRC",
        "validation_metrics": results, "internal_test_scored": False,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
