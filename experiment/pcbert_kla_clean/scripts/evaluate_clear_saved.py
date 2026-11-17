"""Evaluate preselected CLEAR-Lactyl checkpoints on the internal held-out set."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from argparse import Namespace
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from pcbert_kla_clean.backbones import load_tokenizer  # noqa: E402
from pcbert_kla_clean.clear_lactyl import load_clear_lactyl  # noqa: E402
from pcbert_kla_clean.datasets import make_loader  # noqa: E402
from pcbert_kla_clean.deepkla import DeepKlaAligned, encode_windows  # noqa: E402
from pcbert_kla_clean.hybridkla_features import load_resources  # noqa: E402
from pcbert_kla_clean.hybridkla_models import FeatureDNN, MetaDNN, SequenceLSTM, predict_scores  # noqa: E402
from pcbert_kla_clean.metrics import compute_binary_metrics  # noqa: E402
from pcbert_kla_clean.source_study import sha256  # noqa: E402
from pcbert_kla_clean.training import build_model, predict  # noqa: E402
from run_hybridkla_adapted import BRANCHES, RESOURCE_SHA256  # noqa: E402
from run_hybridkla_architecture import (  # noqa: E402
    TrainableEsmBranch, branch_arrays, esm_scores, read_lock, tokenize_sequences,
)


def checked_metrics(labels: np.ndarray, scores: np.ndarray, prior: dict,
                    name: str) -> dict[str, float]:
    if len(labels) != len(scores) or not np.isfinite(scores).all():
        raise ValueError(f"Invalid {name} scores")
    metrics = {key: float(value) for key, value in compute_binary_metrics(labels, scores).items()}
    for key in ("AUPRC", "AUC", "MCC"):
        if abs(metrics[key] - prior[key]) > 1e-4:
            raise ValueError(f"{name} saved validation {key} did not reproduce: "
                             f"{metrics[key]} versus {prior[key]}")
    return metrics


def save_result(output_dir: Path, model_name: str, rows, scores: np.ndarray,
                validation_metrics: dict, provenance: dict) -> dict:
    labels = np.asarray([int(row.label if hasattr(row, "label") else row["label"])
                         for row in rows])
    if len(labels) != len(scores) or not np.isfinite(scores).all():
        raise ValueError(f"Invalid {model_name} test scores")
    metrics = {key: float(value) for key, value in compute_binary_metrics(labels, scores).items()}
    result = {"model": model_name, "partition": "CLEAR-Lactyl internal held-out test",
              "rows": len(rows), "threshold": 0.5, "validation_metrics": validation_metrics,
              "test_metrics": metrics, "provenance": provenance}
    with (output_dir / f"{model_name}_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("accession", "position_1based", "label", "score"))
        for row, score in zip(rows, scores):
            if hasattr(row, "name"):
                writer.writerow((row.name, row.position, row.label, float(score)))
            else:
                writer.writerow((row["accession"], row["position_1based"], row["label"],
                                 float(score)))
    (output_dir / f"{model_name}_report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"model": model_name, "test_metrics": metrics}), flush=True)
    return result


def evaluate_pcbert(name: str, run_dir: Path, lock: Path, output_dir: Path,
                    device: torch.device) -> dict:
    runs = list((run_dir / "results/clear_lactyl/runs").glob("*/run_config.json"))
    if len(runs) != 1:
        raise ValueError(f"Expected exactly one saved {name} run")
    saved = runs[0].parent
    config = json.loads(runs[0].read_text())
    expected_architecture = "baseline" if name == "pcbert_kla" else "token_gated"
    if (config["architecture"] != expected_architecture or config["seed"] != 42 or
            config["run"] != "clear-lactyl-development" or
            config["training_rows"] != 20599 or config["validation_rows"] != 2944 or
            not config["fixed_validation_partition"] or config["decision_threshold_used"] != 0.5 or
            config["selected_by"] != "AUPRC" or config.get("add_cksaap", False)):
        raise ValueError(f"Unexpected saved {name} configuration")
    development, val_idx, test = load_clear_lactyl(lock)
    train_count = int(val_idx[0])
    scaler = joblib.load(saved / "feature_scaler_development.joblib")
    tokenizer = load_tokenizer(config["model_name"], None)
    args = Namespace(**config)
    model = build_model(args, development.features.shape[1], device)
    model.load_state_dict(torch.load(saved / f"{expected_architecture}_development.pt",
                                     map_location=device, weights_only=True), strict=True)
    sequence_format = "protbert_spaced"
    val_loader = make_loader(development, val_idx,
                             scaler.transform(development.features).astype(np.float32),
                             tokenizer, 4, False, config["max_length"], sequence_format)
    test_loader = make_loader(test, np.arange(len(test.records)),
                              scaler.transform(test.features).astype(np.float32),
                              tokenizer, 4, False, config["max_length"], sequence_format)
    _, val_labels, val_scores = predict(model, val_loader, nn.BCELoss(), device)
    prior = json.loads((saved / "validation_metrics.json").read_text())
    validation_metrics = checked_metrics(val_labels, val_scores, prior, name)
    _, test_labels, test_scores = predict(model, test_loader, nn.BCELoss(), device)
    if not np.array_equal(test_labels.astype(int), test.labels):
        raise ValueError(f"{name} test row order changed")
    return save_result(output_dir, name, test.records, test_scores, validation_metrics,
                       {"saved_run": str(saved), "selected_epoch": config["selected_epoch"]})


def evaluate_deepkla(run_dir: Path, lock: Path, output_dir: Path,
                     device: torch.device) -> dict:
    saved = run_dir / "results"
    prior = json.loads((saved / "report.json").read_text())
    if (prior["architecture"] != "deepkla_aligned_pytorch" or prior["seed"] != 42 or
            prior["train_rows"] != 20599 or prior["validation_rows"] != 2944):
        raise ValueError("Unexpected saved DeepKla configuration")
    development, val_idx, test = load_clear_lactyl(lock)
    model = DeepKlaAligned().to(device)
    model.load_state_dict(torch.load(saved / "best_model.pt", map_location=device,
                                     weights_only=True), strict=True)

    def scores(rows) -> np.ndarray:
        tokens = encode_windows([row.sequence for row in rows])
        batches = DataLoader(TensorDataset(tokens), batch_size=256, shuffle=False)
        model.eval()
        chunks = []
        with torch.inference_mode():
            for (batch,) in batches:
                chunks.append(model(batch.to(device)).cpu().numpy())
        return np.concatenate(chunks)

    val_rows = [development.records[int(index)] for index in val_idx]
    val_scores = scores(val_rows)
    validation_metrics = checked_metrics(development.labels[val_idx], val_scores,
                                         prior["validation_metrics"], "deepkla_aligned")
    test_scores = scores(test.records)
    return save_result(output_dir, "deepkla_aligned", test.records, test_scores,
                       validation_metrics,
                       {"saved_run": str(saved), "selected_epoch": prior["selected_epoch"]})


def evaluate_hybrid(run_dir: Path, lock: Path, output_dir: Path,
                    device: torch.device) -> dict:
    saved = run_dir / "results"
    prior = json.loads((saved / "run_report.json").read_text())
    if (prior["mode"] != "development-only" or prior["seed"] != 42 or
            prior["benchmark_lock_sha256"] != sha256(lock) or prior["folds"] != 5 or
            prior["threshold"] != 0.5):
        raise ValueError("Unexpected saved HybridKla configuration")
    partitions, _ = read_lock(lock)
    rows = partitions["validation"] + partitions["internal_test"]
    validation_count = len(partitions["validation"])
    resources_dir = run_dir / "hybridkla_resources"
    for name, expected in RESOURCE_SHA256.items():
        if sha256(resources_dir / name) != expected:
            raise ValueError("HybridKla feature resources changed")
    resources = load_resources(resources_dir / "top10.txt", resources_dir / "BLOSUM62R.txt")
    branch_scores = np.empty((len(rows), len(BRANCHES)), dtype=np.float32)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(prior["model_name"], revision=prior["model_revision"])
    for column, branch_name in enumerate(BRANCHES):
        accumulated = np.zeros(len(rows), dtype=np.float64)
        for fold in range(5):
            checkpoint = saved / "checkpoints" / f"{branch_name}_fold_{fold}.pth"
            if branch_name == "ESM2":
                model = TrainableEsmBranch(prior["model_name"], prior["model_revision"]).to(device)
                model.load_state_dict(torch.load(checkpoint, map_location=device,
                                                 weights_only=True), strict=True)
                fold_scores = esm_scores(model, tokenize_sequences(
                    [row["window_51aa"] for row in rows], tokenizer), device,
                    prior["hyperparameters"]["esm_batch_size"])
            else:
                gps_reference = np.load(saved / f"GPS_fold_{fold}_reference.npy")
                values = branch_arrays(branch_name, rows, resources, gps_reference)
                model = (SequenceLSTM() if branch_name == "LSTM" else FeatureDNN(branch_name)).to(device)
                model.load_state_dict(torch.load(checkpoint, map_location=device,
                                                 weights_only=True), strict=True)
                fold_scores = predict_scores(model, values, device,
                                             batch_size=128 if branch_name == "LSTM" else 1024)
            accumulated += fold_scores / 5
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        branch_scores[:, column] = accumulated.astype(np.float32)
    meta = MetaDNN().to(device)
    meta.load_state_dict(torch.load(saved / "checkpoints/META.pth", map_location=device,
                                    weights_only=True), strict=True)
    scores = predict_scores(meta, branch_scores, device)
    val_labels = np.array([int(row["label"]) for row in rows[:validation_count]])
    validation_metrics = checked_metrics(val_labels, scores[:validation_count],
                                         prior["validation_metrics"], "hybridkla_aligned")
    return save_result(output_dir, "hybridkla_aligned", rows[validation_count:],
                       scores[validation_count:], validation_metrics,
                       {"saved_run": str(saved), "model_revision": prior["model_revision"]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clear-lock", type=Path, required=True)
    parser.add_argument("--saved-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    results = {}
    paths = {
        "pcbert_kla": args.saved_root / "baseline/seed_42",
        "original_mm_kla": args.saved_root / "token_gated_protbert_head_lr/seed_42",
        "hybridkla_aligned": args.saved_root / "hybridkla_architecture_aligned/seed_42",
        "deepkla_aligned": args.saved_root / "deepkla_aligned/seed_42",
    }
    results["pcbert_kla"] = evaluate_pcbert(
        "pcbert_kla", paths["pcbert_kla"], args.clear_lock, args.output_dir, device)
    results["original_mm_kla"] = evaluate_pcbert(
        "original_mm_kla", paths["original_mm_kla"], args.clear_lock, args.output_dir, device)
    results["deepkla_aligned"] = evaluate_deepkla(
        paths["deepkla_aligned"], args.clear_lock, args.output_dir, device)
    results["hybridkla_aligned"] = evaluate_hybrid(
        paths["hybridkla_aligned"], args.clear_lock, args.output_dir, device)
    (args.output_dir / "summary.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
