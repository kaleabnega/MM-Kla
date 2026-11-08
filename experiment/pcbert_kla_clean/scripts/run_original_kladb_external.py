"""Grouped model selection on original PCBert-Kla training, then external inference."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT / "src"))

TRAIN_SHA = "5a2f2f220c4761b963b9c4487573f4d6fad516d820ef29837c8553075fa05824"
FEATURE_SHA = "dc1f19a95bf420804203ab2db1010e15cb925f1bb050b5cdd4066a23c15a942b"
EXTERNAL_SOURCE_SHA = "2a46fd62b1d4d67ded1c31d8a9956f48eab6b7c33aba146f9d9f24c267b34748"
EXTERNAL_CSV_SHA = "0165ebe6d79e52bd74a848aa76e788e27804bf78ba2e8c98126d119cb449eae9"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_external(csv_path: Path, manifest_path: Path):
    import numpy as np

    from pcbert_kla_clean.data import KlaRecord, KlaSplit
    from pcbert_kla_clean.features import FEATURE_NAMES, calculate_features

    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("file") != csv_path.name or manifest.get("sha256") != EXTERNAL_CSV_SHA
            or sha256(csv_path) != EXTERNAL_CSV_SHA
            or manifest.get("source_full_manifest_sha256") != EXTERNAL_SOURCE_SHA):
        raise ValueError("External dataset/manifest checksum or source lock mismatch")
    records, vectors, sites, proteins = [], [], set(), set()
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"accession", "position_1based", "window_51aa", "label", "evidence", "source_pmids", "partition"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("External CSV is missing required audited columns")
        for row in reader:
            accession = row["accession"].strip()
            position = int(row["position_1based"])
            sequence = row["window_51aa"].strip().upper()
            label = int(row["label"])
            site = (accession, position)
            if (row["partition"] != "external_test" or row["source_pmids"] not in {"", "41192556"}
                    or len(sequence) != 51 or sequence[25] != "K" or label not in {0, 1}
                    or site in sites or not accession):
                raise ValueError(f"Invalid or duplicate external site {site}")
            if label == 1 and (row["evidence"] != "reported_Kla_site" or row["source_pmids"] != "41192556"):
                raise ValueError(f"Positive site lacks expected held-out study evidence: {site}")
            if label == 0 and row["evidence"] != "unreported_lysine_putative_negative":
                raise ValueError(f"Negative site lacks putative-negative designation: {site}")
            sites.add(site)
            proteins.add(accession)
            records.append(KlaRecord(name=f"{accession}:{position}", position=26, label=label, sequence=sequence))
            vectors.append(calculate_features(sequence))
    counts = {"rows": len(records), "positives": sum(item.label for item in records),
              "putative_negatives": sum(item.label == 0 for item in records), "proteins": len(proteins)}
    if counts != {"rows": 5095, "positives": 254, "putative_negatives": 4841, "proteins": 81} or counts != manifest.get("counts"):
        raise ValueError(f"External-study counts differ from the lock: {counts}")
    return KlaSplit(records=records, feature_names=FEATURE_NAMES, features=np.asarray(vectors, dtype=np.float32)), manifest


def grouped_validation_indices(train, mapping_path: Path, clusters_path: Path, seed: int):
    import numpy as np
    from sklearn.model_selection import GroupShuffleSplit

    from pcbert_kla_clean.candidate_groups import _candidate_components, load_homology_clusters, load_window_candidates

    candidates = load_window_candidates(mapping_path)
    clusters = load_homology_clusters(clusters_path)
    unique_sequences = list(dict.fromkeys(train.sequences))
    labels_by_sequence = {}
    for record in train.records:
        labels_by_sequence.setdefault(record.sequence, set()).add(record.label)
    if any(len(labels) != 1 for labels in labels_by_sequence.values()):
        raise ValueError("Original training data contains contradictory labels for one window")
    groups_by_sequence, _ = _candidate_components(unique_sequences, candidates, clusters)
    groups = np.asarray([groups_by_sequence[sequence] for sequence in unique_sequences])
    unique_labels = np.asarray([next(iter(labels_by_sequence[sequence])) for sequence in unique_sequences])
    splitter = GroupShuffleSplit(n_splits=256, test_size=0.2, random_state=seed)
    best = None
    for train_unique, val_unique in splitter.split(unique_sequences, unique_labels, groups):
        val_positive = int(unique_labels[val_unique].sum())
        train_positive = int(unique_labels[train_unique].sum())
        if val_positive < 5 or train_positive < 5:
            continue
        score = abs(len(val_unique) / len(unique_sequences) - 0.2) + abs(val_positive / int(unique_labels.sum()) - 0.2)
        if best is None or score < best[0]:
            best = (score, val_unique)
    if best is None:
        raise ValueError("No candidate-group validation split retained both classes")
    val_sequences = {unique_sequences[index] for index in best[1]}
    train_indices = np.asarray([i for i, sequence in enumerate(train.sequences) if sequence not in val_sequences], dtype=int)
    val_indices = np.asarray([next(i for i, sequence in enumerate(train.sequences) if sequence == item) for item in unique_sequences if item in val_sequences], dtype=int)
    if not len(train_indices) or not len(val_indices):
        raise ValueError("Empty grouped training or validation partition")
    if {groups_by_sequence[train.sequences[i]] for i in train_indices} & {groups_by_sequence[train.sequences[i]] for i in val_indices}:
        raise ValueError("Candidate-homology groups overlap training and validation")
    return train_indices, val_indices, {"training_rows": len(train_indices), "validation_unique_windows": len(val_indices),
                                         "validation_positive_windows": int(train.labels[val_indices].sum()),
                                         "training_positive_rows": int(train.labels[train_indices].sum()),
                                         "training_unique_windows": len({train.sequences[i] for i in train_indices}),
                                         "validation_group_count": len({groups_by_sequence[train.sequences[i]] for i in val_indices})}


def save_external_predictions(path: Path, split, scores) -> None:
    if len(split.records) != len(scores):
        raise ValueError("External prediction count does not match the locked sites")
    fields = ["name", "accession", "position_1based", "window_51aa", "true_label", "score", "threshold", "predicted_label"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record, score in zip(split.records, scores):
            accession, position = record.name.rsplit(":", 1)
            writer.writerow({"name": record.name, "accession": accession,
                             "position_1based": int(position), "window_51aa": record.sequence,
                             "true_label": record.label, "score": float(score),
                             "threshold": 0.5, "predicted_label": int(float(score) >= 0.5)})


def args_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-csv", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--architecture", choices=["baseline", "token_gated"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--mapping-csv", type=Path, default=REPO / "experiment-1/structure_aware_kla/results/botrytis_b0510_window_mapping/all_window_matches.csv")
    parser.add_argument("--clusters-csv", type=Path, default=PROJECT / "review_data/homology_clusters_mmseqs2.csv")
    return parser.parse_args()


def main() -> None:
    import numpy as np
    import torch
    from sklearn.preprocessing import MinMaxScaler

    from pcbert_kla_clean.backbones import load_tokenizer
    from pcbert_kla_clean.data import load_split
    from pcbert_kla_clean.datasets import make_loader
    from pcbert_kla_clean.metrics import compute_binary_metrics
    from pcbert_kla_clean.training import build_model, build_optimizer, make_training_criterion, predict, train_one_epoch
    from pcbert_kla_clean.utils import set_seed

    args = args_parser()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    train_file = REPO / "baselines/PCBert-Kla-original/data/train.csv"
    feature_file = train_file.with_name("feature_train.csv")
    if sha256(train_file) != TRAIN_SHA or sha256(feature_file) != FEATURE_SHA:
        raise ValueError("Original PCBert-Kla training files differ from the pinned source")
    train = load_split(train_file, feature_file)
    external, external_manifest = load_external(args.external_csv, args.external_manifest)
    if len(train.records) != 3487 or int(train.labels.sum()) != 1720 or train.feature_names != external.feature_names:
        raise ValueError("Original training size, labels, or feature schema changed")
    if set(train.sequences) & set(external.sequences):
        raise ValueError("Exact sequence-window overlap between original training and external test")
    train_idx, val_idx, validation = grouped_validation_indices(train, args.mapping_csv, args.clusters_csv, args.seed)
    preflight = {"original_training_rows": len(train.records), "original_training_positives": int(train.labels.sum()),
                 "original_training_unique_windows": len(set(train.sequences)), "external_test": external_manifest["counts"],
                 "external_sha256": external_manifest["sha256"], "validation": validation,
                 "original_train_sha256": TRAIN_SHA, "original_features_sha256": FEATURE_SHA,
                 "candidate_mapping_sha256": sha256(args.mapping_csv),
                 "homology_clusters_sha256": sha256(args.clusters_csv),
                 "selection": "highest grouped-validation AUPRC; earliest epoch on ties",
                 "external_use": "one final inference after full-data refit", "threshold": 0.5}
    print(json.dumps(preflight, indent=2), flush=True)
    if args.check_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this Kaggle training run")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    device = torch.device(args.device)
    set_seed(args.seed)
    args.model_name = "Rostlab/prot_bert"
    args.encoder_layers = 4
    args.freeze_encoder = False
    args.cache_dir = None
    args.fusion_dim = 256
    args.attention_dim = 256
    args.arch_dropout = 0.2
    args.site_token_index = 26
    args.ablation = "none"
    args.optimizer = "sgd"
    args.learning_rate = 0.003
    args.sgd_momentum = 0.0
    args.weight_decay = 0.0
    args.scheduler = "none"
    tokenizer = load_tokenizer(args.model_name, args.cache_dir)
    criterion, _ = make_training_criterion(train.labels, train_idx, "none")
    scaler_selection = MinMaxScaler().fit(train.features[train_idx])
    scaled_train = scaler_selection.transform(train.features).astype(np.float32)
    train_loader = make_loader(train, train_idx, scaled_train, tokenizer, args.batch_size, True, 64, "protbert_spaced")
    val_loader = make_loader(train, val_idx, scaled_train, tokenizer, args.batch_size, False, 64, "protbert_spaced")
    model = build_model(args, feature_dim=27, device=device)
    optimizer = build_optimizer(args, model)
    history = []
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, labels, scores = predict(model, val_loader, criterion, device)
        if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise ValueError(f"Invalid validation probabilities in epoch {epoch}")
        metrics = compute_binary_metrics(labels, scores, threshold=0.5)
        history.append({"epoch": epoch, "train_loss": loss, "validation_loss": val_loss, **metrics})
        print(f"{args.architecture} seed={args.seed} validation epoch={epoch} AUPRC={metrics['AUPRC']:.4f}", flush=True)
    selected = max(history, key=lambda item: item["AUPRC"])
    (args.output_dir / "validation_history.json").write_text(json.dumps(history, indent=2) + "\n")
    del model, optimizer, train_loader, val_loader, scaler_selection
    gc.collect()
    torch.cuda.empty_cache()

    set_seed(args.seed)
    all_idx = np.arange(len(train.records))
    criterion, weights = make_training_criterion(train.labels, all_idx, "none")
    scaler = MinMaxScaler().fit(train.features)
    scaled_full = scaler.transform(train.features).astype(np.float32)
    full_loader = make_loader(train, all_idx, scaled_full, tokenizer, args.batch_size, True, 64, "protbert_spaced")
    model = build_model(args, feature_dim=27, device=device)
    optimizer = build_optimizer(args, model)
    refit_losses = []
    for epoch in range(1, selected["epoch"] + 1):
        loss = train_one_epoch(model, full_loader, optimizer, criterion, device)
        refit_losses.append(loss)
        print(f"{args.architecture} seed={args.seed} full refit epoch={epoch}/{selected['epoch']} loss={loss:.4f}", flush=True)
    torch.save({"model_state_dict": model.state_dict(), "architecture": args.architecture,
                "seed": args.seed, "selected_epoch": selected["epoch"]}, args.output_dir / "model.pt")
    import joblib
    joblib.dump(scaler, args.output_dir / "feature_scaler.joblib")
    (args.output_dir / "refit_losses.json").write_text(json.dumps(refit_losses, indent=2) + "\n")

    external_scaled = scaler.transform(external.features).astype(np.float32)
    external_idx = np.arange(len(external.records))
    external_loader = make_loader(external, external_idx, external_scaled, tokenizer, args.batch_size, False, 64, "protbert_spaced")
    test_loss, labels, scores = predict(model, external_loader, criterion, device)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Invalid external-test probabilities")
    metrics = compute_binary_metrics(labels, scores, threshold=0.5)
    metrics["loss"] = test_loss
    metrics["selected_epoch"] = int(selected["epoch"])
    save_external_predictions(args.output_dir / "external_predictions.csv", external, scores)
    (args.output_dir / "external_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    report = {"preflight": preflight, "architecture": args.architecture, "seed": args.seed,
              "device": str(device), "git_commit": subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
              "model_name": args.model_name, "encoder_layers": args.encoder_layers,
              "optimizer": args.optimizer, "learning_rate": args.learning_rate,
              "batch_size": args.batch_size, "max_epochs": args.epochs,
              "selected_epoch": int(selected["epoch"]),
              "validation_at_selection": selected, "refit_training": weights,
              "refit_epochs": len(refit_losses), "external_metrics": metrics,
              "wall_seconds": round(time.time() - started, 1)}
    (args.output_dir / "run_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
