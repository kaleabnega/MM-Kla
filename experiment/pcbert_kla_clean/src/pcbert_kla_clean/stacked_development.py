"""Leakage-screened two-branch development experiment for CLEAR-Lactyl."""

from __future__ import annotations

import csv
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pcbert_kla_clean.backbones import load_tokenizer
from pcbert_kla_clean.clear_lactyl import load_clear_lactyl
from pcbert_kla_clean.data import KlaSplit
from pcbert_kla_clean.datasets import make_loader
from pcbert_kla_clean.features import FEATURE_NAMES
from pcbert_kla_clean.metrics import compute_binary_metrics
from pcbert_kla_clean.source_study import sha256
from pcbert_kla_clean.training import (
    build_model, build_optimizer, build_scheduler, make_training_criterion,
    predict, train_one_epoch,
)
from pcbert_kla_clean.utils import set_seed


AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_ID = {residue: index for index, residue in enumerate(AA)}


POSITIONAL_ABLATIONS = ("none", "no_position_embedding", "one_lstm_layer", "mean_pool")
FEATURE_ABLATIONS = ("none", "no_descriptors", "no_cksaap")


class PositionalSequenceBranch(nn.Module):
    """Independent, site-focused sequence-order predictor without a PLM."""

    def __init__(self, ablation: str = "none") -> None:
        super().__init__()
        if ablation not in POSITIONAL_ABLATIONS:
            raise ValueError(f"Unsupported positional ablation: {ablation}")
        self.ablation = ablation
        self.residue = nn.Embedding(len(AA), 64)
        self.position = nn.Embedding(51, 64)
        layers = 1 if ablation == "one_lstm_layer" else 2
        self.encoder = nn.LSTM(
            64, 64, num_layers=layers, batch_first=True, bidirectional=True,
            dropout=0.2 if layers == 2 else 0,
        )
        self.attention = nn.Linear(256, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(256), nn.Linear(256, 64), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(64, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.residue(tokens)
        if self.ablation != "no_position_embedding":
            positions = torch.arange(51, device=tokens.device)
            embedded = embedded + self.position(positions)[None, :, :]
        return self.forward_embedded(embedded)

    def forward_embedded(self, embedded: torch.Tensor) -> torch.Tensor:
        states, _ = self.encoder(embedded)
        center = states[:, 25, :]
        if self.ablation == "mean_pool":
            context = states.mean(dim=1)
        else:
            query = center[:, None, :].expand_as(states)
            weights = torch.softmax(self.attention(torch.cat((states, query), dim=-1)), dim=1)
            context = (weights * states).sum(dim=1)
        return self.classifier(torch.cat((center, context), dim=-1)).squeeze(-1)


def grouped_splits(labels: np.ndarray, groups: np.ndarray, seed: int, folds: int = 5):
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = list(splitter.split(np.zeros(len(labels)), labels, groups))
    seen = np.zeros(len(labels), dtype=np.int8)
    for train_idx, held_idx in splits:
        if (set(groups[train_idx]) & set(groups[held_idx]) or
                len(np.unique(labels[train_idx])) != 2 or
                len(np.unique(labels[held_idx])) != 2):
            raise ValueError("Grouped fold has overlapping groups or lacks a class")
        seen[held_idx] += 1
    if not np.all(seen == 1):
        raise ValueError("Grouped folds do not cover each training row exactly once")
    return splits


def protbert_args(epochs: int, ablation: str = "none", freeze_encoder: bool = False) -> Namespace:
    return Namespace(
        architecture="token_gated", model_name="Rostlab/prot_bert",
        encoder_layers=4, fusion_dim=256, attention_dim=256, arch_dropout=0.2,
        site_token_index=26, freeze_encoder=freeze_encoder, ablation=ablation, cache_dir=None,
        encoder_adaptation="full", lora_rank=8, lora_alpha=32, lora_dropout=0.3,
        train_last_encoder_layers=None, optimizer="adamw", learning_rate=2e-5,
        head_learning_rate=1e-4, weight_decay=0.01, adam_beta1=0.9,
        adam_beta2=0.999, adam_epsilon=1e-8, scheduler="linear",
        warmup_ratio=0.1, epochs=epochs,
    )


def feature_columns(feature_names: list[str], ablation: str) -> np.ndarray:
    if ablation not in FEATURE_ABLATIONS:
        raise ValueError(f"Unsupported feature ablation: {ablation}")
    descriptor_count = len(FEATURE_NAMES)
    if (len(feature_names) != descriptor_count + 1600 or
            feature_names[:descriptor_count] != list(FEATURE_NAMES)):
        raise ValueError("Expected 27 descriptors followed by 1,600 CKSAAP features")
    if ablation == "no_descriptors":
        return np.arange(descriptor_count, len(feature_names))
    if ablation == "no_cksaap":
        return np.arange(descriptor_count)
    return np.arange(len(feature_names))


def fit_protbert(split, train_idx, score_idx, tokenizer, device, seed,
                 max_epochs, patience, selection_idx=None, fixed_epochs=None,
                 ablation="none", freeze_encoder=False, feature_ablation="none"):
    if selection_idx is None and (fixed_epochs is None or not 1 <= fixed_epochs <= max_epochs):
        raise ValueError("Refit needs a selected epoch within the 30-epoch schedule")
    set_seed(seed)
    args = protbert_args(max_epochs, ablation=ablation, freeze_encoder=freeze_encoder)
    scaler = MinMaxScaler()
    columns = feature_columns(split.feature_names, feature_ablation)
    features = split.features[:, columns]
    scaled = np.empty_like(features, dtype=np.float32)
    scaled[train_idx] = scaler.fit_transform(features[train_idx])
    scaled[score_idx] = scaler.transform(features[score_idx])
    if selection_idx is not None:
        scaled[selection_idx] = scaler.transform(features[selection_idx])
    train_loader = make_loader(split, train_idx, scaled, tokenizer, 4, True, 64,
                               "protbert_spaced")
    score_loader = make_loader(split, score_idx, scaled, tokenizer, 4, False, 64,
                               "protbert_spaced")
    selection_loader = (
        make_loader(split, selection_idx, scaled, tokenizer, 4, False, 64,
                    "protbert_spaced") if selection_idx is not None else None
    )
    model = build_model(args, len(columns), device)
    criterion, _ = make_training_criterion(split.labels, train_idx, "none")
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer, len(train_loader))
    best_epoch, best_ap, stale = 0, -1.0, 0
    epochs_to_run = max_epochs if selection_idx is not None else fixed_epochs
    for epoch in range(1, epochs_to_run + 1):
        train_one_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        if selection_loader is None:
            continue
        _, labels, scores = predict(model, selection_loader, criterion, device)
        ap = average_precision_score(labels, scores)
        if ap > best_ap:
            best_epoch, best_ap, stale = epoch, ap, 0
        else:
            stale += 1
        print(f"ProtBERT inner epoch={epoch} AUPRC={ap:.5f}", flush=True)
        if stale >= patience:
            break
    result = (best_epoch, best_ap) if selection_loader is not None else predict(
        model, score_loader, criterion, device,
    )[2]
    del model, optimizer, scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def sequence_tokens(sequences: list[str]) -> torch.Tensor:
    if any(len(sequence) != 51 or sequence[25] != "K" or
           set(sequence) - set(AA) for sequence in sequences):
        raise ValueError("Positional branch needs canonical 51-residue K-centered windows")
    return torch.tensor([[AA_TO_ID[residue] for residue in sequence]
                         for sequence in sequences], dtype=torch.long)


def position_scores(model, tokens, indices, device):
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in DataLoader(TensorDataset(tokens[indices]), batch_size=256):
            scores.extend(torch.sigmoid(model(batch[0].to(device))).cpu().tolist())
    return np.asarray(scores, dtype=np.float64)


def fit_positional(tokens, labels, train_idx, score_idx, device, seed,
                   max_epochs, patience, selection_idx=None, ablation="none"):
    set_seed(seed)
    model = PositionalSequenceBranch(ablation=ablation).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(tokens[train_idx], torch.tensor(
        labels[train_idx], dtype=torch.float32)), batch_size=128,
        shuffle=True, generator=generator)
    best_epoch, best_ap, stale = 0, -1.0, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_tokens, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_tokens.to(device))
            loss = criterion(logits, batch_labels.to(device))
            loss.backward()
            optimizer.step()
        if selection_idx is None:
            continue
        scores = position_scores(model, tokens, selection_idx, device)
        ap = average_precision_score(labels[selection_idx], scores)
        if ap > best_ap:
            best_epoch, best_ap, stale = epoch, ap, 0
        else:
            stale += 1
        print(f"Positional inner epoch={epoch} AUPRC={ap:.5f}", flush=True)
        if stale >= patience:
            break
    result = (best_epoch, best_ap) if selection_idx is not None else position_scores(
        model, tokens, score_idx, device,
    )
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def read_training_groups(lock_path: Path, split, train_count: int) -> np.ndarray:
    with lock_path.with_name("model_manifest.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    train_rows = [row for row in rows if row["partition"] == "train"]
    if len(train_rows) != train_count:
        raise ValueError("Manifest training count differs from CLEAR-Lactyl loader")
    for row, record in zip(train_rows, split.records[:train_count]):
        if (row["accession"] != record.name or row["window_51aa"] != record.sequence or
                int(row["label"]) != record.label):
            raise ValueError("Manifest order differs from CLEAR-Lactyl loader")
    validation_groups = {row["homology_group"] for row in rows
                         if row["partition"] == "validation"}
    groups = np.asarray([row["homology_group"] for row in train_rows])
    if set(groups) & validation_groups:
        raise ValueError("Training homology group appears in validation")
    return groups


def read_branch_reference(reference_dir: Path, lock_path: Path, split,
                          train_count: int, validation_idx: np.ndarray,
                          groups: np.ndarray, seed: int, branch: str) -> tuple[np.ndarray, np.ndarray, dict]:
    branch_columns = {"positional": (1, "positional_sequence"),
                      "protbert": (0, "protbert_cksaap")}
    column_index, column_name = branch_columns[branch]
    report = json.loads((reference_dir / "report.json").read_text())
    if (report["seed"] != seed or report["lock_sha256"] != sha256(lock_path) or
            report["training_rows"] != train_count or
            report["validation_rows"] != len(validation_idx) or
            report["feature_dim"] != split.features.shape[1] or
            report["internal_test_scored"] or len(report["folds"]) != 5 or
            report.get("protbert_ablation", "none") != "none" or
            report.get("positional_ablation", "none") != "none" or
            report.get("feature_ablation", "none") != "none" or
            report.get("freeze_protbert", False)):
        raise ValueError("Branch reference does not match the unchanged locked validation protocol")
    with np.load(reference_dir / "oof_branch_scores.npz") as saved:
        if (saved["scores"].shape != (train_count, 2) or
                not np.array_equal(saved["labels"], split.labels[:train_count]) or
                not np.array_equal(saved["groups"], groups)):
            raise ValueError("Positional OOF rows, labels, or groups do not match")
        oof_scores = saved["scores"][:, column_index].copy()
    with (reference_dir / "validation_predictions.csv").open(newline="") as handle:
        validation_rows = list(csv.DictReader(handle))
    if len(validation_rows) != len(validation_idx):
        raise ValueError("Positional validation rows do not match")
    for row, index in zip(validation_rows, validation_idx):
        record = split.records[int(index)]
        if (row["accession"] != record.name or
                int(row["position_1based"]) != record.position or
                int(row["label"]) != record.label):
            raise ValueError("Positional validation identities or labels do not match")
    validation_scores = np.asarray(
        [float(row[column_name]) for row in validation_rows], dtype=np.float64,
    )
    if (not np.isfinite(oof_scores).all() or
            not np.isfinite(validation_scores).all() or
            ((oof_scores < 0) | (oof_scores > 1)).any() or
            ((validation_scores < 0) | (validation_scores > 1)).any()):
        raise ValueError(f"{branch} reference contains invalid probabilities")
    return oof_scores, validation_scores, report


def read_positional_reference(reference_dir: Path, lock_path: Path, split,
                              train_count: int, validation_idx: np.ndarray,
                              groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    return read_branch_reference(reference_dir, lock_path, split, train_count,
                                 validation_idx, groups, seed, "positional")


def run(lock_path: Path, output_dir: Path, device: torch.device, seed: int,
        max_epochs: int = 30, patience: int = 5,
        score_internal_test: bool = False,
        reference_report: Path | None = None,
        protbert_ablation: str = "none",
        positional_reference_dir: Path | None = None,
        freeze_protbert: bool = False,
        protbert_reference_dir: Path | None = None,
        positional_ablation: str = "none",
        feature_ablation: str = "none") -> dict:
    if max_epochs < 1 or patience < 1 or seed < 0:
        raise ValueError("Invalid seed, epoch count, or patience")
    if positional_ablation not in POSITIONAL_ABLATIONS or feature_ablation not in FEATURE_ABLATIONS:
        raise ValueError("Unknown branch ablation")
    if score_internal_test and reference_report is None:
        raise ValueError("Locked test scoring requires the prior validation report")
    if (positional_reference_dir is not None or protbert_reference_dir is not None) and score_internal_test:
        raise ValueError("Saved branch references support validation-only ablations")
    if positional_reference_dir is not None and protbert_reference_dir is not None:
        raise ValueError("Only one branch may be loaded from a reference")
    if (protbert_ablation != "none" or freeze_protbert or feature_ablation != "none") and positional_reference_dir is None:
        raise ValueError("ProtBERT ablations require a fixed positional reference")
    if positional_ablation != "none" and protbert_reference_dir is None:
        raise ValueError("Positional ablations require a fixed ProtBERT reference")
    if protbert_reference_dir is not None and (protbert_ablation != "none" or freeze_protbert or feature_ablation != "none"):
        raise ValueError("ProtBERT cannot be ablated and reused simultaneously")
    if positional_reference_dir is not None and positional_ablation != "none":
        raise ValueError("Positional branch cannot be ablated and reused simultaneously")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    split, validation_idx, internal_test = load_clear_lactyl(lock_path, add_cksaap=True)
    test_idx = np.arange(len(split.records), len(split.records) + len(internal_test.records))
    if score_internal_test:
        split = KlaSplit(
            records=split.records + internal_test.records,
            feature_names=split.feature_names,
            features=np.concatenate((split.features, internal_test.features)),
        )
    train_count = int(validation_idx[0])
    train_idx = np.arange(train_count)
    labels = split.labels.astype(int)
    groups = read_training_groups(lock_path, split, train_count)
    folds = grouped_splits(labels[:train_count], groups, seed)
    reference = None
    if positional_reference_dir is not None:
        positional_oof, positional_validation, reference = read_positional_reference(
            positional_reference_dir, lock_path, split, train_count,
            validation_idx, groups, seed,
        )
    if protbert_reference_dir is not None:
        protbert_oof, protbert_validation, reference = read_branch_reference(
            protbert_reference_dir, lock_path, split, train_count,
            validation_idx, groups, seed, "protbert",
        )
    if positional_reference_dir is None:
        tokens = sequence_tokens(split.sequences)
    tokenizer = load_tokenizer("Rostlab/prot_bert", None) if protbert_reference_dir is None else None
    oof = np.full((train_count, 2), np.nan, dtype=np.float64)
    validation = np.zeros((len(validation_idx), 2), dtype=np.float64)
    test_predictions = np.zeros((len(test_idx), 2), dtype=np.float64) if score_internal_test else None
    if positional_reference_dir is not None:
        oof[:, 1] = positional_oof
        validation[:, 1] = positional_validation
    if protbert_reference_dir is not None:
        oof[:, 0] = protbert_oof
        validation[:, 0] = protbert_validation
    fold_reports = []
    output_dir.mkdir(parents=True, exist_ok=False)
    for fold_number, (outer_train, held_idx) in enumerate(folds, 1):
        inner_train_local, inner_stop_local = grouped_splits(
            labels[outer_train], groups[outer_train], seed + fold_number,
        )[0]
        inner_train, inner_stop = outer_train[inner_train_local], outer_train[inner_stop_local]
        if set(groups[inner_train]) & set(groups[inner_stop]):
            raise ValueError("Inner checkpoint groups overlap")
        report = {"fold": fold_number, "outer_train_rows": len(outer_train),
                  "outer_held_rows": len(held_idx), "inner_stop_rows": len(inner_stop)}
        score_indices = np.r_[held_idx, validation_idx, test_idx] if score_internal_test else np.r_[held_idx, validation_idx]
        val_end = len(held_idx) + len(validation_idx)
        if protbert_reference_dir is None:
            print(f"Fold {fold_number}/5: selecting ProtBERT epoch", flush=True)
            prot_epoch, prot_ap = fit_protbert(
                split, inner_train, inner_stop, tokenizer, device, seed + 100 * fold_number,
                max_epochs, patience, selection_idx=inner_stop,
                ablation=protbert_ablation, freeze_encoder=freeze_protbert,
                feature_ablation=feature_ablation,
            )
            report["protbert_selected_epoch"] = prot_epoch
            report["protbert_inner_auprc"] = prot_ap
            print(f"Fold {fold_number}/5: refitting ProtBERT", flush=True)
            prot_scores = fit_protbert(
                split, outer_train, score_indices, tokenizer, device,
                seed + 100 * fold_number, max_epochs, patience, fixed_epochs=prot_epoch,
                ablation=protbert_ablation, freeze_encoder=freeze_protbert,
                feature_ablation=feature_ablation,
            )
            oof[held_idx, 0] = prot_scores[:len(held_idx)]
            validation[:, 0] += prot_scores[len(held_idx):val_end] / len(folds)
            if test_predictions is not None:
                test_predictions[:, 0] += prot_scores[val_end:] / len(folds)
        else:
            report["protbert_selected_epoch"] = reference["folds"][fold_number - 1]["protbert_selected_epoch"]
            report["protbert_inner_auprc"] = reference["folds"][fold_number - 1]["protbert_inner_auprc"]

        if positional_reference_dir is None:
            print(f"Fold {fold_number}/5: selecting positional epoch", flush=True)
            pos_epoch, pos_ap = fit_positional(
                tokens, labels, inner_train, inner_stop, device, seed + 100 * fold_number + 1,
                max_epochs, patience, selection_idx=inner_stop, ablation=positional_ablation,
            )
            report["positional_selected_epoch"] = pos_epoch
            report["positional_inner_auprc"] = pos_ap
            print(f"Fold {fold_number}/5: refitting positional branch", flush=True)
            pos_scores = fit_positional(
                tokens, labels, outer_train, score_indices, device,
                seed + 100 * fold_number + 1, pos_epoch, patience,
                ablation=positional_ablation,
            )
            oof[held_idx, 1] = pos_scores[:len(held_idx)]
            validation[:, 1] += pos_scores[len(held_idx):val_end] / len(folds)
            if test_predictions is not None:
                test_predictions[:, 1] += pos_scores[val_end:] / len(folds)
        else:
            report["positional_selected_epoch"] = reference["folds"][fold_number - 1]["positional_selected_epoch"]
            report["positional_inner_auprc"] = reference["folds"][fold_number - 1]["positional_inner_auprc"]
        fold_reports.append(report)
        print(json.dumps(report, sort_keys=True), flush=True)
    if (not np.isfinite(oof).all() or not np.isfinite(validation).all() or
            test_predictions is not None and not np.isfinite(test_predictions).all()):
        raise FloatingPointError("Incomplete or nonfinite branch scores")
    meta = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    meta.fit(oof, labels[:train_count])
    scores = {
        "protbert_cksaap": validation[:, 0],
        "positional_sequence": validation[:, 1],
        "simple_average": validation.mean(axis=1),
        "oof_logistic_combination": meta.predict_proba(validation)[:, 1],
    }
    results = {name: compute_binary_metrics(labels[validation_idx], values, threshold=0.5)
               for name, values in scores.items()}
    if score_internal_test:
        reference = json.loads(reference_report.read_text())
        if (reference["seed"] != seed or reference["lock_sha256"] != sha256(lock_path) or
                reference["internal_test_scored"]):
            raise ValueError("Prior two-branch report does not match this validation run")
        for name in ("protbert_cksaap", "positional_sequence", "oof_logistic_combination"):
            previous = reference["validation_metrics"][name]
            for key in ("AUPRC", "AUC"):
                if abs(results[name][key] - previous[key]) > 1e-3:
                    raise ValueError(f"Two-branch validation {name} {key} did not reproduce")
    test_results = None
    if test_predictions is not None:
        test_scores = {
            "protbert_cksaap": test_predictions[:, 0],
            "positional_sequence": test_predictions[:, 1],
            "simple_average": test_predictions.mean(axis=1),
            "oof_logistic_combination": meta.predict_proba(test_predictions)[:, 1],
        }
        test_results = {name: compute_binary_metrics(labels[test_idx], values, threshold=0.5)
                        for name, values in test_scores.items()}
        with (output_dir / "internal_test_predictions.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["accession", "position_1based", "label", *test_scores])
            for row_number, index in enumerate(test_idx):
                record = split.records[int(index)]
                writer.writerow([record.name, record.position, record.label,
                                 *(float(values[row_number]) for values in test_scores.values())])
    np.savez_compressed(output_dir / "oof_branch_scores.npz", scores=oof,
                        labels=labels[:train_count], groups=groups)
    with (output_dir / "validation_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["accession", "position_1based", "label", *scores])
        for row_number, index in enumerate(validation_idx):
            record = split.records[int(index)]
            writer.writerow([record.name, record.position, record.label,
                             *(float(values[row_number]) for values in scores.values())])
    report = {
        "seed": seed, "lock_sha256": sha256(lock_path),
        "protbert_ablation": protbert_ablation,
        "freeze_protbert": freeze_protbert,
        "positional_ablation": positional_ablation,
        "feature_ablation": feature_ablation,
        "used_feature_dim": len(feature_columns(split.feature_names, feature_ablation)),
        "positional_reference_dir": str(positional_reference_dir) if positional_reference_dir else None,
        "protbert_reference_dir": str(protbert_reference_dir) if protbert_reference_dir else None,
        "training_rows": train_count, "validation_rows": len(validation_idx),
        "feature_dim": split.features.shape[1], "folds": fold_reports,
        "combiner": {"type": "logistic_regression", "C": 1.0,
                     "coefficients": meta.coef_[0].tolist(),
                     "intercept": float(meta.intercept_[0])},
        "selection": "inner grouped training fold AUPRC only",
        "primary_metric": "validation AUPRC",
        "validation_metrics": results,
        "internal_test_scored": score_internal_test,
    }
    if test_results is not None:
        report["internal_test_rows"] = len(test_idx)
        report["internal_test_metrics"] = test_results
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
