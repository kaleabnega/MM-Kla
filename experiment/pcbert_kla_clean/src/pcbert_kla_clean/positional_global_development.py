"""Validation-only positional predictor with coarse full-protein context."""

from __future__ import annotations

import csv
import gzip
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pcbert_kla_clean.clear_lactyl import load_clear_lactyl
from pcbert_kla_clean.metrics import compute_binary_metrics
from pcbert_kla_clean.source_study import sha256
from pcbert_kla_clean.stacked_development import (
    AA, AA_TO_ID, PositionalSequenceBranch, grouped_splits,
    read_training_groups, sequence_tokens,
)
from pcbert_kla_clean.utils import set_seed


CHUNKS = 8
GLOBAL_CHANNELS = len(AA) + 2


def load_verified_proteins(path: Path, records) -> dict[str, str]:
    needed = {record.name for record in records}
    proteins: dict[str, str] = {}
    opener = gzip.open if path.suffix == ".gz" else open
    accession = None
    sequence_parts: list[str] = []

    def finish() -> None:
        if accession in needed:
            if accession in proteins:
                raise ValueError(f"Duplicate protein accession: {accession}")
            proteins[accession] = "".join(sequence_parts).upper()

    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                finish()
                identifier = line[1:].split()[0]
                accession = identifier.split("|")[1] if "|" in identifier else identifier
                sequence_parts = []
            else:
                sequence_parts.append(line.strip())
    finish()
    if needed - proteins.keys():
        raise ValueError(f"Reference proteome lacks {len(needed - proteins.keys())} accessions")
    for record in records:
        protein = proteins[record.name]
        if protein[record.position - 26:record.position + 25] != record.sequence:
            raise ValueError(f"Reference sequence mismatch: {record.name}:{record.position}")
    return proteins


def protein_chunks(sequence: str) -> np.ndarray:
    """Eight ordered amino-acid frequency summaries with an unknown-residue channel."""
    values = np.zeros((CHUNKS, GLOBAL_CHANNELS), dtype=np.float32)
    for chunk_number, chunk in enumerate(np.array_split(list(sequence), CHUNKS)):
        for residue in chunk:
            values[chunk_number, AA_TO_ID.get(residue, len(AA))] += 1
        if len(chunk):
            values[chunk_number] /= len(chunk)
    return values


def mark_site_chunk(features: np.ndarray, protein_length: int, position: int) -> np.ndarray:
    if not 1 <= position <= protein_length:
        raise ValueError("Candidate position falls outside the protein")
    marked = features.copy()
    base, extra = divmod(protein_length, CHUNKS)
    lengths = [base + (index < extra) for index in range(CHUNKS)]
    chunk_index = int(np.searchsorted(np.cumsum(lengths), position - 1, side="right"))
    marked[chunk_index, -1] = 1
    return marked


class GlobalPositionalSequenceBranch(PositionalSequenceBranch):
    """Original local pathway plus one center-to-protein-context attention."""

    def __init__(self) -> None:
        super().__init__()
        self.global_projection = nn.Linear(GLOBAL_CHANNELS, 64)
        self.global_position = nn.Embedding(CHUNKS, 64)
        self.global_query = nn.Linear(128, 64)
        self.global_attention = nn.MultiheadAttention(64, 4, batch_first=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(320), nn.Linear(320, 64), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(64, 1),
        )

    def forward(self, tokens: torch.Tensor, global_chunks: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(51, device=tokens.device)
        embedded = self.residue(tokens) + self.position(positions)[None, :, :]
        states, _ = self.encoder(embedded)
        center = states[:, 25, :]
        query = center[:, None, :].expand_as(states)
        weights = torch.softmax(self.attention(torch.cat((states, query), dim=-1)), dim=1)
        local_context = (weights * states).sum(dim=1)

        global_positions = torch.arange(CHUNKS, device=tokens.device)
        global_tokens = self.global_projection(global_chunks) + self.global_position(global_positions)
        global_context, _ = self.global_attention(
            self.global_query(center)[:, None, :], global_tokens, global_tokens,
            need_weights=False,
        )
        combined = torch.cat((center, local_context, global_context[:, 0, :]), dim=-1)
        return self.classifier(combined).squeeze(-1)


def predict_scores(model, tokens, global_features, indices, device, global_mode):
    model.eval()
    scores = []
    dataset = TensorDataset(tokens[indices], global_features[indices])
    with torch.no_grad():
        for local, protein in DataLoader(dataset, batch_size=256):
            local, protein = local.to(device), protein.to(device)
            logits = model(local, protein) if global_mode else model(local)
            scores.extend(torch.sigmoid(logits).cpu().tolist())
    return np.asarray(scores, dtype=np.float64)


def fit_model(tokens, global_features, labels, train_idx, score_idx, device,
              seed, max_epochs, patience, global_mode, selection_idx=None,
              fixed_epochs=None):
    if selection_idx is None and (fixed_epochs is None or not 1 <= fixed_epochs <= max_epochs):
        raise ValueError("Refit requires a selected epoch")
    set_seed(seed)
    model = (GlobalPositionalSequenceBranch() if global_mode else PositionalSequenceBranch()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(
        tokens[train_idx], global_features[train_idx],
        torch.tensor(labels[train_idx], dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=128, shuffle=True, generator=generator)
    best_epoch, best_ap, stale = 0, -1.0, 0
    epoch_limit = max_epochs if selection_idx is not None else fixed_epochs
    for epoch in range(1, epoch_limit + 1):
        model.train()
        for local, protein, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            local, protein = local.to(device), protein.to(device)
            logits = model(local, protein) if global_mode else model(local)
            loss = criterion(logits, batch_labels.to(device))
            loss.backward()
            optimizer.step()
        if selection_idx is None:
            continue
        scores = predict_scores(model, tokens, global_features, selection_idx, device, global_mode)
        ap = average_precision_score(labels[selection_idx], scores)
        if ap > best_ap:
            best_epoch, best_ap, stale = epoch, ap, 0
        else:
            stale += 1
        print(f"{'global' if global_mode else 'control'} inner epoch={epoch} AUPRC={ap:.5f}", flush=True)
        if stale >= patience:
            break
    result = ((best_epoch, best_ap) if selection_idx is not None else
              predict_scores(model, tokens, global_features, score_idx, device, global_mode))
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run(lock: Path, proteome: Path, output_dir: Path, device: torch.device,
        seed: int = 42, max_epochs: int = 30, patience: int = 5) -> dict:
    if seed < 0 or max_epochs < 1 or patience < 1:
        raise ValueError("Invalid run configuration")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    split, validation_idx, _ = load_clear_lactyl(lock, add_cksaap=False)
    train_count = int(validation_idx[0])
    train_idx = np.arange(train_count)
    labels = split.labels.astype(int)
    groups = read_training_groups(lock, split, train_count)
    folds = grouped_splits(labels[:train_count], groups, seed)
    proteins = load_verified_proteins(proteome, split.records)
    chunks_by_accession = {name: protein_chunks(sequence) for name, sequence in proteins.items()}
    global_features = torch.from_numpy(np.stack([
        mark_site_chunk(chunks_by_accession[record.name],
                        len(proteins[record.name]), record.position)
        for record in split.records
    ], axis=0))
    tokens = sequence_tokens(split.sequences)
    oof = np.full((train_count, 2), np.nan, dtype=np.float64)
    validation = np.zeros((len(validation_idx), 2), dtype=np.float64)
    fold_reports = []
    start = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=False)
    for fold_number, (outer_train, held_idx) in enumerate(folds, 1):
        inner_train_local, inner_stop_local = grouped_splits(
            labels[outer_train], groups[outer_train], seed + fold_number,
        )[0]
        inner_train = outer_train[inner_train_local]
        inner_stop = outer_train[inner_stop_local]
        fold_report = {"fold": fold_number}
        for column, (name, global_mode) in enumerate((
            ("positional_control", False), ("positional_global", True),
        )):
            print(f"Fold {fold_number}/5: selecting {name} epoch", flush=True)
            selected_epoch, inner_ap = fit_model(
                tokens, global_features, labels, inner_train, inner_stop,
                device, seed + 100 * fold_number + 1, max_epochs, patience,
                global_mode, selection_idx=inner_stop,
            )
            print(f"Fold {fold_number}/5: refitting {name}", flush=True)
            score_idx = np.r_[held_idx, validation_idx]
            predictions = fit_model(
                tokens, global_features, labels, outer_train, score_idx,
                device, seed + 100 * fold_number + 1, max_epochs, patience,
                global_mode, fixed_epochs=selected_epoch,
            )
            oof[held_idx, column] = predictions[:len(held_idx)]
            validation[:, column] += predictions[len(held_idx):] / len(folds)
            fold_report[name] = {"selected_epoch": selected_epoch, "inner_auprc": inner_ap}
        fold_reports.append(fold_report)
        print(json.dumps(fold_report, sort_keys=True), flush=True)
    if not np.isfinite(oof).all() or not np.isfinite(validation).all():
        raise FloatingPointError("Incomplete or nonfinite predictions")
    names = ("positional_control", "positional_global")
    metrics = {name: compute_binary_metrics(labels[validation_idx], validation[:, column], 0.5)
               for column, name in enumerate(names)}
    with (output_dir / "validation_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("accession", "position_1based", "label", *names))
        for row_number, index in enumerate(validation_idx):
            record = split.records[int(index)]
            writer.writerow((record.name, record.position, record.label,
                             *validation[row_number].tolist()))
    np.savez_compressed(output_dir / "oof_scores.npz", scores=oof,
                        labels=labels[train_idx], groups=groups)
    report = {
        "seed": seed, "lock_sha256": sha256(lock), "proteome_sha256": sha256(proteome),
        "training_rows": train_count, "validation_rows": len(validation_idx),
        "internal_test_scored": False, "selection": "inner grouped fold AUPRC",
        "global_features": "eight ordered full-protein amino-acid frequency chunks with candidate-site marker",
        "global_attention": "central lysine state queries eight protein chunks",
        "control_parameters": sum(p.numel() for p in PositionalSequenceBranch().parameters()),
        "global_parameters": sum(p.numel() for p in GlobalPositionalSequenceBranch().parameters()),
        "epochs_cap": max_epochs, "patience": patience,
        "folds": fold_reports, "validation_metrics": metrics,
        "delta_global_minus_control_auprc": metrics[names[1]]["AUPRC"] - metrics[names[0]]["AUPRC"],
        "wall_seconds": time.monotonic() - start,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
