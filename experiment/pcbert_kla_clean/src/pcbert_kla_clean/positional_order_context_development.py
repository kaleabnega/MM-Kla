"""Validation-only test of an order-sensitive full-protein context encoder."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset

from pcbert_kla_clean.clear_lactyl import load_clear_lactyl
from pcbert_kla_clean.metrics import compute_binary_metrics
from pcbert_kla_clean.positional_global_development import CHUNKS, load_verified_proteins
from pcbert_kla_clean.source_study import sha256
from pcbert_kla_clean.stacked_development import (
    AA, AA_TO_ID, PositionalSequenceBranch, grouped_splits,
    read_training_groups, sequence_tokens,
)
from pcbert_kla_clean.utils import set_seed


MICROBATCH = 32
EFFECTIVE_BATCH = 128


def protein_tokens(sequence: str) -> torch.Tensor:
    # Zero is reserved for padding; unknown residues get their own embedding.
    return torch.tensor([AA_TO_ID.get(residue, len(AA)) + 1 for residue in sequence],
                        dtype=torch.long)


def make_protein_batch(indices: torch.Tensor, records, bank: dict[str, torch.Tensor],
                       device: torch.device):
    proteins = [bank[records[int(index)].name] for index in indices]
    lengths = torch.tensor([len(protein) for protein in proteins], device=device)
    positions = torch.tensor([records[int(index)].position - 1 for index in indices],
                             device=device)
    return pad_sequence(proteins, batch_first=True).to(device), lengths, positions


def batch_targets(labels: np.ndarray, indices: torch.Tensor, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(labels[indices.tolist()], dtype=torch.float32, device=device)


class OrderContextPositionalBranch(PositionalSequenceBranch):
    """Local positional pathway with an order-sensitive full-protein context."""

    def __init__(self) -> None:
        super().__init__()
        self.protein_residue = nn.Embedding(len(AA) + 2, 32, padding_idx=0)
        self.protein_depthwise = nn.Conv1d(32, 32, 5, padding=2, groups=32)
        self.protein_pointwise = nn.Conv1d(32, 64, 1)
        self.global_position = nn.Embedding(CHUNKS, 64)
        self.site_marker = nn.Embedding(2, 64)
        self.global_query = nn.Linear(128, 64)
        self.global_attention = nn.MultiheadAttention(64, 4, batch_first=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(320), nn.Linear(320, 64), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(64, 1),
        )

    def encode_protein(self, proteins: torch.Tensor, lengths: torch.Tensor,
                       site_positions: torch.Tensor) -> torch.Tensor:
        if torch.any(lengths < 1) or torch.any(site_positions < 0) or torch.any(site_positions >= lengths):
            raise ValueError("Invalid protein length or site position")
        embedded = self.protein_residue(proteins).transpose(1, 2)
        residue_states = torch.nn.functional.gelu(
            self.protein_pointwise(self.protein_depthwise(embedded))
        ).transpose(1, 2)
        positions = torch.arange(proteins.shape[1], device=proteins.device)[None, :]
        valid = positions < lengths[:, None]
        bins = torch.clamp(positions * CHUNKS // lengths[:, None], max=CHUNKS - 1)
        bins = bins.expand_as(proteins)
        pooled = residue_states.new_zeros((len(proteins), CHUNKS, 64))
        pooled.scatter_add_(1, bins[:, :, None].expand(-1, -1, 64),
                            residue_states * valid[:, :, None])
        counts = residue_states.new_zeros((len(proteins), CHUNKS, 1))
        counts.scatter_add_(1, bins[:, :, None], valid[:, :, None].to(residue_states.dtype))
        pooled = pooled / counts.clamp_min(1)
        chunk_ids = torch.arange(CHUNKS, device=proteins.device)
        site_bin = torch.clamp(site_positions * CHUNKS // lengths, max=CHUNKS - 1)
        marker = (chunk_ids[None, :] == site_bin[:, None]).long()
        return pooled + self.global_position(chunk_ids) + self.site_marker(marker)

    def global_padding_mask(self, proteins: torch.Tensor, lengths: torch.Tensor,
                            site_positions: torch.Tensor) -> torch.Tensor | None:
        return None

    def forward(self, local_tokens: torch.Tensor, proteins: torch.Tensor,
                lengths: torch.Tensor, site_positions: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(51, device=local_tokens.device)
        embedded = self.residue(local_tokens) + self.position(positions)[None, :, :]
        states, _ = self.encoder(embedded)
        center = states[:, 25, :]
        query = center[:, None, :].expand_as(states)
        weights = torch.softmax(self.attention(torch.cat((states, query), dim=-1)), dim=1)
        local_context = (weights * states).sum(dim=1)
        global_tokens = self.encode_protein(proteins, lengths, site_positions)
        global_context, _ = self.global_attention(
            self.global_query(center)[:, None, :], global_tokens, global_tokens,
            key_padding_mask=self.global_padding_mask(proteins, lengths, site_positions),
            need_weights=False,
        )
        return self.classifier(torch.cat((center, local_context, global_context[:, 0]), dim=-1)).squeeze(-1)


def predict_scores(model, local_tokens, records, bank, indices, device):
    model.eval()
    scores = []
    with torch.no_grad():
        for (batch_indices,) in DataLoader(TensorDataset(torch.as_tensor(indices)), batch_size=MICROBATCH):
            proteins, lengths, positions = make_protein_batch(batch_indices, records, bank, device)
            logits = model(local_tokens[batch_indices].to(device), proteins, lengths, positions)
            scores.extend(torch.sigmoid(logits).cpu().tolist())
    return np.asarray(scores, dtype=np.float64)


def fit_model(local_tokens, records, bank, labels, train_idx, score_idx, device,
              seed, max_epochs, patience, selection_idx=None, fixed_epochs=None,
              checkpoint_path=None, model_factory=OrderContextPositionalBranch):
    if selection_idx is None and (fixed_epochs is None or not 1 <= fixed_epochs <= max_epochs):
        raise ValueError("Refit requires a selected epoch")
    set_seed(seed)
    model = model_factory().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.as_tensor(train_idx)), batch_size=EFFECTIVE_BATCH,
                        shuffle=True, generator=generator)
    best_epoch, best_ap, stale = 0, -1.0, 0
    epoch_limit = max_epochs if selection_idx is not None else fixed_epochs
    for epoch in range(1, epoch_limit + 1):
        model.train()
        for (batch_indices,) in loader:
            optimizer.zero_grad(set_to_none=True)
            for micro in batch_indices.split(MICROBATCH):
                proteins, lengths, positions = make_protein_batch(micro, records, bank, device)
                logits = model(local_tokens[micro].to(device), proteins, lengths, positions)
                loss = criterion(logits, batch_targets(labels, micro, device))
                (loss / len(batch_indices)).backward()
            optimizer.step()
        if selection_idx is None:
            continue
        scores = predict_scores(model, local_tokens, records, bank, selection_idx, device)
        ap = average_precision_score(labels[selection_idx], scores)
        if ap > best_ap:
            best_epoch, best_ap, stale = epoch, ap, 0
        else:
            stale += 1
        print(f"order_context inner epoch={epoch} AUPRC={ap:.5f}", flush=True)
        if stale >= patience:
            break
    result = ((best_epoch, best_ap) if selection_idx is not None else
              predict_scores(model, local_tokens, records, bank, score_idx, device))
    if checkpoint_path is not None:
        if selection_idx is not None:
            raise ValueError("Checkpoint saving requires a fixed-epoch refit")
        torch.save({"model_state_dict": {name: value.detach().cpu()
                                         for name, value in model.state_dict().items()},
                    "selected_epoch": fixed_epochs}, checkpoint_path)
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run(lock: Path, proteome: Path, reference_report: Path, output_dir: Path,
        device: torch.device, seed: int = 42, max_epochs: int = 30,
        patience: int = 5) -> dict:
    if seed < 0 or max_epochs < 1 or patience < 1:
        raise ValueError("Invalid run configuration")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    reference = json.loads(reference_report.read_text())
    if (reference["seed"] != seed or reference["lock_sha256"] != sha256(lock) or
            reference["proteome_sha256"] != sha256(proteome) or
            reference["epochs_cap"] != max_epochs or reference["patience"] != patience or
            reference["internal_test_scored"]):
        raise ValueError("Reference pilot does not match the locked inputs and protocol")
    split, validation_idx, _ = load_clear_lactyl(lock, add_cksaap=False)
    train_count = int(validation_idx[0])
    labels = split.labels.astype(int)
    groups = read_training_groups(lock, split, train_count)
    folds = grouped_splits(labels[:train_count], groups, seed)
    if (reference["training_rows"] != train_count or
            reference["validation_rows"] != len(validation_idx)):
        raise ValueError("Reference split size differs")
    proteins = load_verified_proteins(proteome, split.records)
    bank = {name: protein_tokens(sequence) for name, sequence in proteins.items()}
    local_tokens = sequence_tokens(split.sequences)
    oof = np.full(train_count, np.nan, dtype=np.float64)
    validation = np.zeros(len(validation_idx), dtype=np.float64)
    fold_reports = []
    start = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=False)
    for fold_number, (outer_train, held_idx) in enumerate(folds, 1):
        inner_train_local, inner_stop_local = grouped_splits(
            labels[outer_train], groups[outer_train], seed + fold_number,
        )[0]
        inner_train, inner_stop = outer_train[inner_train_local], outer_train[inner_stop_local]
        print(f"Fold {fold_number}/5: selecting order-context epoch", flush=True)
        selected_epoch, inner_ap = fit_model(
            local_tokens, split.records, bank, labels, inner_train, inner_stop,
            device, seed + 100 * fold_number + 1, max_epochs, patience,
            selection_idx=inner_stop,
        )
        print(f"Fold {fold_number}/5: refitting order-context model", flush=True)
        score_idx = np.r_[held_idx, validation_idx]
        predictions = fit_model(
            local_tokens, split.records, bank, labels, outer_train, score_idx,
            device, seed + 100 * fold_number + 1, max_epochs, patience,
            fixed_epochs=selected_epoch,
        )
        oof[held_idx] = predictions[:len(held_idx)]
        validation += predictions[len(held_idx):] / len(folds)
        fold_reports.append({"fold": fold_number, "selected_epoch": selected_epoch,
                             "inner_auprc": inner_ap})
        print(json.dumps(fold_reports[-1], sort_keys=True), flush=True)
    if not np.isfinite(oof).all() or not np.isfinite(validation).all():
        raise FloatingPointError("Incomplete or nonfinite predictions")
    metrics = compute_binary_metrics(labels[validation_idx], validation, 0.5)
    with (output_dir / "validation_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("accession", "position_1based", "label", "order_context_score"))
        for row_number, index in enumerate(validation_idx):
            record = split.records[int(index)]
            writer.writerow((record.name, record.position, record.label, validation[row_number]))
    np.savez_compressed(output_dir / "oof_scores.npz", scores=oof,
                        labels=labels[:train_count], groups=groups)
    reference_metrics = reference["validation_metrics"]["positional_global"]
    report = {
        "seed": seed, "lock_sha256": sha256(lock), "proteome_sha256": sha256(proteome),
        "reference_report_sha256": sha256(reference_report),
        "training_rows": train_count, "validation_rows": len(validation_idx),
        "internal_test_scored": False, "selection": "inner grouped fold AUPRC",
        "global_features": "all full-protein residues encoded by a depthwise-separable 1D convolution and pooled into eight ordered tokens",
        "effective_batch": EFFECTIVE_BATCH, "physical_microbatch": MICROBATCH,
        "parameters": sum(p.numel() for p in OrderContextPositionalBranch().parameters()),
        "epochs_cap": max_epochs, "patience": patience, "folds": fold_reports,
        "validation_metrics": metrics,
        "reference_frequency_context_metrics": reference_metrics,
        "delta_auprc_vs_frequency_context": metrics["AUPRC"] - reference_metrics["AUPRC"],
        "wall_seconds": time.monotonic() - start,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
