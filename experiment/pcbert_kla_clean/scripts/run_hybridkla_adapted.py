"""Train a compute-bounded eight-branch HybridKla adaptation on the human lock.

This is not an exact reproduction of the published HybridKla training recipe:
ESM2 is frozen and a group-disjoint holdout replaces five-fold base stacking.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from export_hybridkla_splits import export_splits
from pcbert_kla_clean.hybridkla_features import (
    FEATURE_DIMS, encode_features, fit_gps_reference, load_resources,
)
from pcbert_kla_clean.hybridkla_models import (
    EsmEmbeddingHead, FeatureDNN, MetaDNN, SequenceLSTM,
    encode_lstm_sequences, fit_predictor, predict_scores,
)


BRANCHES = ("ESM2", "LSTM", "ACF", "AAINDEX", "CKSAAP", "GPS", "OBC", "PSEAAC")
RESOURCE_SHA256 = {
    "BLOSUM62R.txt": "7f6f4a1ba4c9df90b402898e9ed7baaf9e515037b4236f063d9fcf991e044526",
    "top10.txt": "6ec0c0a43535c1718ed0f80972c759e03121627a4b129cb4112f00d4adacd2fe",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_partition(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(row["Label"] not in {"0", "1"} for row in rows):
        raise ValueError(f"Invalid partition: {path}")
    return rows


def partition_labels(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray([int(row["Label"]) for row in rows], dtype=np.int64)


def split_training_groups(rows: list[dict[str, str]], fraction: float,
                          seed: int) -> tuple[list[int], list[int]]:
    if not 0.05 <= fraction <= 0.4:
        raise ValueError("Meta-training fraction must be between 0.05 and 0.4")
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault(row["Homology_group"], []).append(index)
    if len(groups) < 5:
        raise ValueError("Too few homology groups for a held-out meta-training partition")
    rng = random.Random(seed)
    group_names = list(groups)
    rng.shuffle(group_names)
    selected = set()
    count = 0
    target = round(len(rows) * fraction)
    for name in group_names:
        if count >= target:
            break
        selected.add(name)
        count += len(groups[name])
    base = [index for index, row in enumerate(rows) if row["Homology_group"] not in selected]
    meta = [index for index, row in enumerate(rows) if row["Homology_group"] in selected]
    if not base or not meta or set(rows[index]["Homology_group"] for index in base) & selected:
        raise ValueError("Invalid group-disjoint meta-training split")
    if {rows[index]["ACC_ID"] for index in base} & {rows[index]["ACC_ID"] for index in meta}:
        raise ValueError("Protein crosses meta-training split")
    for name, indices in (("base", base), ("meta", meta)):
        if set(partition_labels([rows[index] for index in indices])) != {0, 1}:
            raise ValueError(f"{name} partition lacks a class")
    return base, meta


def extract_frozen_esm_embeddings(
    partitions: dict[str, list[dict[str, str]]], model_name: str,
    revision: str, device: torch.device, batch_size: int,
) -> tuple[dict[str, np.ndarray], int]:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    encoder = AutoModel.from_pretrained(model_name, revision=revision).to(device).eval()
    hidden_size = int(encoder.config.hidden_size)
    special_ids = set(tokenizer.all_special_ids)
    result = {}
    for name, rows in partitions.items():
        embeddings = []
        sequences = [row["Sequence"] for row in rows]
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start:start + batch_size]
            tokens = tokenizer(batch, return_tensors="pt", padding=True,
                               truncation=False)
            if tokens["input_ids"].shape[1] > 53:
                raise ValueError("Unexpected ESM2 token length for 51-residue windows")
            tokens = {key: value.to(device) for key, value in tokens.items()}
            with torch.inference_mode():
                if device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        states = encoder(**tokens).last_hidden_state
                else:
                    states = encoder(**tokens).last_hidden_state
                mask = tokens["attention_mask"].bool()
                for token_id in special_ids:
                    mask &= tokens["input_ids"] != token_id
                if not mask.any(dim=1).all():
                    raise ValueError("ESM2 embedding mask excludes every residue")
                mask_float = mask.unsqueeze(-1)
                mean = (states.float() * mask_float).sum(dim=1) / mask_float.sum(dim=1)
            embeddings.append(mean.cpu().numpy().astype(np.float32))
        result[name] = np.concatenate(embeddings)
        if result[name].shape != (len(rows), hidden_size) or not np.isfinite(result[name]).all():
            raise ValueError(f"Invalid ESM2 embeddings for {name}")
        print(f"Frozen ESM2 embeddings: {name} {result[name].shape}", flush=True)
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, hidden_size


def evaluate_and_save(
    name: str, rows: list[dict[str, str]], scores: np.ndarray, output_dir: Path,
) -> dict[str, float]:
    from pcbert_kla_clean.metrics import compute_binary_metrics

    labels = partition_labels(rows)
    if scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError(f"Invalid predictions for {name}")
    metrics = {key: float(value) for key, value in
               compute_binary_metrics(labels, scores, threshold=0.5).items()}
    (output_dir / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    with (output_dir / f"{name}_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("Site_ID", "Label", "Score"))
        writer.writerows((row["Site_ID"], row["Label"], float(score))
                         for row, score in zip(rows, scores))
    return metrics


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    device = torch.device(args.device)
    esm_device = torch.device(args.esm_device)
    if (device.type == "cuda" or esm_device.type == "cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device is unavailable")
    if args.esm_batch_size < 1 or args.epochs < 1 or args.patience < 0:
        raise ValueError("Invalid ESM batch size, epochs, or patience")
    if args.feature_batch_size < 2 or args.lstm_batch_size < 2 or args.meta_epochs < 1:
        raise ValueError("Invalid training batch size or meta epochs")
    for name, expected in RESOURCE_SHA256.items():
        path = args.resource_dir / name
        if sha256(path) != expected:
            raise ValueError(f"HybridKla feature resource differs from official archive: {name}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    from huggingface_hub import HfApi

    esm_revision = HfApi().model_info(args.esm_model).sha
    if not esm_revision or len(esm_revision) != 40:
        raise ValueError("Could not pin the ESM2 model revision")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    export_dir = output / "locked_export"
    export_report = export_splits(args.human_lock, export_dir)
    training = read_partition(export_dir / "train.csv")
    validation = read_partition(export_dir / "validation.csv")
    full_training_count = len(training)
    full_validation_count = len(validation)
    if args.pilot:
        if args.pilot_train_rows < 100 or args.pilot_validation_rows < 20:
            raise ValueError("Pilot row caps are too small to test training")
        rng = np.random.default_rng(args.seed)
        train_indices = sorted(rng.choice(len(training),
                                          min(args.pilot_train_rows, len(training)),
                                          replace=False))
        validation_indices = sorted(rng.choice(len(validation),
                                               min(args.pilot_validation_rows, len(validation)),
                                               replace=False))
        training = [training[index] for index in train_indices]
        validation = [validation[index] for index in validation_indices]
    base_indices, meta_indices = split_training_groups(training, args.meta_fraction, args.seed)
    base_rows = [training[index] for index in base_indices]
    meta_rows = [training[index] for index in meta_indices]
    partitions = {"base": base_rows, "meta": meta_rows, "validation": validation}
    print("Rows: base", len(base_rows), "meta", len(meta_rows),
          "validation", len(validation), flush=True)

    resources = load_resources(args.resource_dir / "top10.txt",
                               args.resource_dir / "BLOSUM62R.txt")
    gps_reference = fit_gps_reference([row["Sequence"] for row in base_rows])
    np.save(output / "gps_reference.npy", gps_reference)
    features = {name: encode_features([row["Sequence"] for row in rows],
                                      resources, gps_reference)
                for name, rows in partitions.items()}
    tokens = {name: encode_lstm_sequences([row["Sequence"] for row in rows])
              for name, rows in partitions.items()}
    embeddings, esm_hidden = extract_frozen_esm_embeddings(
        partitions, args.esm_model, esm_revision, esm_device, args.esm_batch_size)

    base_labels = partition_labels(base_rows)
    meta_labels = partition_labels(meta_rows)
    val_labels = partition_labels(validation)
    meta_scores = np.empty((len(meta_rows), len(BRANCHES)), dtype=np.float32)
    validation_scores = np.empty((len(validation), len(BRANCHES)), dtype=np.float32)
    trained = {}
    selected_epochs = {}
    branch_val_auprc = {}
    for column, name in enumerate(BRANCHES):
        torch.manual_seed(args.seed + column)
        if name == "ESM2":
            model = EsmEmbeddingHead(esm_hidden)
            arrays = embeddings
            learning_rate = 1e-3
            batch_size = args.feature_batch_size
        elif name == "LSTM":
            model = SequenceLSTM()
            arrays = tokens
            learning_rate = 1e-4
            batch_size = args.lstm_batch_size
        else:
            model = FeatureDNN(name)
            arrays = {part: encoded[name] for part, encoded in features.items()}
            learning_rate = 1e-3
            batch_size = args.feature_batch_size
        result = fit_predictor(model, arrays["base"], base_labels,
                               arrays["validation"], val_labels,
                               device=device, epochs=args.epochs, batch_size=batch_size,
                               learning_rate=learning_rate, patience=args.patience,
                               seed=args.seed + column)
        trained[name] = result.model
        selected_epochs[name] = result.selected_epoch
        branch_val_auprc[name] = result.validation_auprc
        infer_batch = 128 if name == "LSTM" else 1024
        meta_scores[:, column] = predict_scores(result.model, arrays["meta"], device,
                                                batch_size=infer_batch)
        validation_scores[:, column] = predict_scores(result.model, arrays["validation"],
                                                      device, batch_size=infer_batch)
        print(f"{name}: selected epoch {result.selected_epoch}, "
              f"validation AUPRC {result.validation_auprc:.4f}", flush=True)

    torch.manual_seed(args.seed + 100)
    meta_result = fit_predictor(MetaDNN(), meta_scores, meta_labels,
                                validation_scores, val_labels,
                                device=device, epochs=args.meta_epochs,
                                batch_size=args.feature_batch_size,
                                learning_rate=1e-3, patience=args.patience,
                                seed=args.seed + 100)
    trained["META"] = meta_result.model
    selected_epochs["META"] = meta_result.selected_epoch
    branch_val_auprc["META"] = meta_result.validation_auprc
    test_metrics = {}
    if args.pilot:
        print("Pilot finished; locked tests were not loaded for scoring.", flush=True)
    else:
        print("All model selection finished; loading locked tests for final scoring.", flush=True)
        tests = {name: read_partition(export_dir / f"{name}.csv")
                 for name in ("internal_test", "external_test")}
        test_features = {name: encode_features([row["Sequence"] for row in rows],
                                               resources, gps_reference)
                         for name, rows in tests.items()}
        test_tokens = {name: encode_lstm_sequences([row["Sequence"] for row in rows])
                       for name, rows in tests.items()}
        test_embeddings, test_hidden = extract_frozen_esm_embeddings(
            tests, args.esm_model, esm_revision, esm_device, args.esm_batch_size)
        if test_hidden != esm_hidden:
            raise ValueError("ESM2 hidden size changed between development and test")
        for split, rows in tests.items():
            scores = np.empty((len(rows), len(BRANCHES)), dtype=np.float32)
            for column, name in enumerate(BRANCHES):
                values = (test_embeddings[split] if name == "ESM2" else
                          test_tokens[split] if name == "LSTM" else test_features[split][name])
                scores[:, column] = predict_scores(trained[name], values, device,
                                                   batch_size=128 if name == "LSTM" else 1024)
            final_scores = predict_scores(trained["META"], scores, device)
            test_metrics[split] = evaluate_and_save(split, rows, final_scores, output)
            print(split, json.dumps(test_metrics[split], sort_keys=True), flush=True)

    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    for name, model in trained.items():
        torch.save({key: tensor.cpu() for key, tensor in model.state_dict().items()},
                   checkpoint_dir / f"{name}.pth")
    report = {
        "method": "HybridKla-adapted, not exact published reproduction",
        "mode": "training-only-pilot" if args.pilot else "locked-test-evaluation",
        "differences": ["frozen ESM2 mean embeddings with trained head",
                        "homology-group-disjoint 20% meta holdout instead of five-fold base stacking",
                        "validation AUPRC checkpoint selection and compute-bounded epochs"],
        "seed": args.seed,
        "model_name": args.esm_model,
        "model_revision": esm_revision,
        "threshold": 0.5,
        "hyperparameters": {
            "epochs": args.epochs,
            "meta_epochs": args.meta_epochs,
            "patience": args.patience,
            "esm_batch_size": args.esm_batch_size,
            "feature_batch_size": args.feature_batch_size,
            "lstm_batch_size": args.lstm_batch_size,
            "meta_fraction": args.meta_fraction,
            "esm_head_learning_rate": 1e-3,
            "lstm_learning_rate": 1e-4,
            "feature_head_learning_rate": 1e-3,
            "meta_learning_rate": 1e-3,
        },
        "devices": {"training": str(device), "esm_extraction": str(esm_device)},
        "training_rows": len(training),
        "full_locked_training_rows": full_training_count,
        "base_training_rows": len(base_rows),
        "meta_training_rows": len(meta_rows),
        "validation_rows": len(validation),
        "full_locked_validation_rows": full_validation_count,
        "base_training_accessions": len({r["ACC_ID"] for r in base_rows}),
        "meta_training_accessions": len({r["ACC_ID"] for r in meta_rows}),
        "base_training_homology_groups": len({r["Homology_group"] for r in base_rows}),
        "meta_training_homology_groups": len({r["Homology_group"] for r in meta_rows}),
        "selected_epochs": selected_epochs,
        "validation_auprc": branch_val_auprc,
        "test_metrics": test_metrics,
        "export_source_manifest_sha256": export_report["source_manifest_sha256"],
        "resources_sha256": RESOURCE_SHA256,
        "gps_reference_sha256": sha256(output / "gps_reference.npy"),
        "wall_seconds": round(time.time() - started, 1),
    }
    (output / "run_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-lock", type=Path, required=True)
    parser.add_argument("--resource-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--esm-device", default="cuda:0")
    parser.add_argument("--esm-model", default="facebook/esm2_t30_150M_UR50D")
    parser.add_argument("--esm-batch-size", type=int, default=32)
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--lstm-batch-size", type=int, default=32)
    parser.add_argument("--meta-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--meta-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--pilot-train-rows", type=int, default=1000)
    parser.add_argument("--pilot-validation-rows", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
