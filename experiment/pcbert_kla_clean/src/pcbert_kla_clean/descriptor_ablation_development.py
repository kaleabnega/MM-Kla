"""Validation-only ablation of the 27 descriptors from the two-predictor stack."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from pcbert_kla_clean.backbones import load_tokenizer
from pcbert_kla_clean.clear_lactyl import load_clear_lactyl
from pcbert_kla_clean.data import KlaSplit
from pcbert_kla_clean.features import FEATURE_NAMES, cksaap_feature_names
from pcbert_kla_clean.metrics import compute_binary_metrics
from pcbert_kla_clean.source_study import sha256
from pcbert_kla_clean.stacked_development import (
    fit_protbert, grouped_splits, read_training_groups,
)


def without_descriptors(split: KlaSplit) -> KlaSplit:
    descriptor_count = len(FEATURE_NAMES)
    cksaap_names = cksaap_feature_names()
    if (descriptor_count != 27 or
            split.feature_names[:descriptor_count] != list(FEATURE_NAMES) or
            split.feature_names[descriptor_count:] != cksaap_names or
            split.features.shape != (len(split.records), descriptor_count + len(cksaap_names))):
        raise ValueError("Expected 27 descriptors followed by the CKSAAP feature family")
    return KlaSplit(records=split.records, feature_names=cksaap_names,
                    features=split.features[:, descriptor_count:])


def load_contextual_reference(reference_dir: Path, lock_path: Path, seed: int,
                              split: KlaSplit, validation_idx: np.ndarray,
                              groups: np.ndarray):
    report_path = reference_dir / "report.json"
    oof_path = reference_dir / "oof_contextual_scores.npz"
    prediction_path = reference_dir / "validation_predictions.csv"
    report = json.loads(report_path.read_text())
    if (report["seed"] != seed or report["lock_sha256"] != sha256(lock_path) or
            report["training_rows"] != len(groups) or
            report["validation_rows"] != len(validation_idx) or
            report["model_name"] != "Rostlab/prot_bert" or
            report["encoder_layers"] != 30 or not report["encoder_frozen"] or
            report["feature_dim"] != split.features.shape[1] or
            report["internal_test_scored"]):
        raise ValueError("Contextual reference differs from the locked experiment")
    with np.load(oof_path) as saved:
        oof = saved["scores"]
        if (oof.shape != (len(groups),) or
                not np.array_equal(saved["labels"], split.labels[:len(groups)]) or
                not np.array_equal(saved["groups"], groups)):
            raise ValueError("Contextual out-of-fold scores do not match training rows")
    with prediction_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(validation_idx):
        raise ValueError("Contextual validation prediction count differs")
    for row, index in zip(rows, validation_idx):
        record = split.records[int(index)]
        if (row["accession"] != record.name or
                int(row["position_1based"]) != record.position or
                int(row["label"]) != record.label):
            raise ValueError("Contextual validation row differs from locked data")
    validation = np.asarray([float(row["contextual_positional"]) for row in rows])
    full_stack = np.asarray([float(row["contextual_two_predictor"]) for row in rows])
    if (not np.isfinite(oof).all() or not np.isfinite(validation).all() or
            not np.isfinite(full_stack).all()):
        raise FloatingPointError("Nonfinite contextual reference score")
    return (oof, validation, full_stack,
            {"report_sha256": sha256(report_path), "oof_sha256": sha256(oof_path),
             "validation_predictions_sha256": sha256(prediction_path)})


def run(lock_path: Path, reference_dir: Path, output_dir: Path,
        device: torch.device, seed: int, max_epochs: int = 30,
        patience: int = 5) -> dict:
    if seed < 0 or max_epochs < 1 or patience < 1:
        raise ValueError("Invalid seed, epoch count, or patience")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    full_split, validation_idx, _ = load_clear_lactyl(lock_path, add_cksaap=True)
    train_count = int(validation_idx[0])
    if not np.array_equal(validation_idx, np.arange(train_count, train_count + len(validation_idx))):
        raise ValueError("Expected contiguous training and validation partitions")
    groups = read_training_groups(lock_path, full_split, train_count)
    folds = grouped_splits(full_split.labels[:train_count], groups, seed)
    contextual_oof, contextual_validation, full_stack_validation, reference_hashes = (
        load_contextual_reference(reference_dir, lock_path, seed,
                                  full_split, validation_idx, groups)
    )
    split = without_descriptors(full_split)
    tokenizer = load_tokenizer("Rostlab/prot_bert", None)
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
        branch_seed = seed + 100 * fold_number
        print(f"Fold {fold_number}/5: selecting no-descriptor ProtBERT epoch", flush=True)
        selected_epoch, inner_ap = fit_protbert(
            split, inner_train, inner_stop, tokenizer, device, branch_seed,
            max_epochs, patience, selection_idx=inner_stop,
        )
        print(f"Fold {fold_number}/5: refitting no-descriptor ProtBERT", flush=True)
        score_idx = np.r_[held_idx, validation_idx]
        scores = fit_protbert(
            split, outer_train, score_idx, tokenizer, device, branch_seed,
            max_epochs, patience, fixed_epochs=selected_epoch,
        )
        oof[held_idx] = scores[:len(held_idx)]
        validation += scores[len(held_idx):] / len(folds)
        fold_reports.append({"fold": fold_number, "outer_train_rows": len(outer_train),
                             "outer_held_rows": len(held_idx), "inner_stop_rows": len(inner_stop),
                             "selected_epoch": selected_epoch, "inner_auprc": inner_ap})
    if not np.isfinite(oof).all() or not np.isfinite(validation).all():
        raise FloatingPointError("Incomplete or nonfinite ablation predictions")
    meta = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    meta.fit(np.column_stack((oof, contextual_oof)), labels[:train_count])
    ablated_stack = meta.predict_proba(
        np.column_stack((validation, contextual_validation)),
    )[:, 1]
    prediction_sets = {
        "reference_contextual_positional": contextual_validation,
        "reference_full_two_predictor": full_stack_validation,
        "no_27_descriptors_protbert_cksaap": validation,
        "no_27_descriptors_two_predictor": ablated_stack,
    }
    metrics = {name: compute_binary_metrics(labels[validation_idx], values, threshold=0.5)
               for name, values in prediction_sets.items()}
    np.savez_compressed(output_dir / "oof_ablation_scores.npz", scores=oof,
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
        "removed_feature_names": list(FEATURE_NAMES),
        "removed_feature_count": len(FEATURE_NAMES),
        "retained_feature_count": len(split.feature_names),
        "retained_feature_family": "CKSAAP gaps 0-3",
        "contextual_reference": reference_hashes,
        "folds": fold_reports,
        "combiner": {"type": "logistic_regression", "C": 1.0,
                     "coefficients": meta.coef_[0].tolist(),
                     "intercept": float(meta.intercept_[0])},
        "selection": "inner grouped training fold AUPRC only",
        "primary_metric": "validation AUPRC",
        "validation_metrics": metrics,
        "internal_test_scored": False,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
