from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from pcbert_kla_clean.data import KlaRecord, KlaSplit


def load_pcbert_revised(lock_path: Path) -> tuple[KlaSplit, np.ndarray, KlaSplit]:
    audit = json.loads(lock_path.read_text())
    manifests = {}
    for filename in ("full_manifest.csv", "model_manifest.csv"):
        path = lock_path.with_name(filename)
        if hashlib.sha256(path.read_bytes()).hexdigest() != audit["output_sha256"][filename]:
            raise ValueError(f"Revised PCBert-Kla {filename} does not match its lock")
        with path.open(newline="") as handle:
            manifests[filename] = list(csv.DictReader(handle))
    full = manifests["full_manifest.csv"]
    rows = manifests["model_manifest.csv"]
    feature_columns = [f"feature_{index}" for index in range(len(audit["feature_names"]))]
    full_by_id = {row["window_id"]: row for row in full}
    if len(full_by_id) != len(full) or not rows:
        raise ValueError("Missing or repeated window IDs in revised benchmark")
    partitions: dict[str, list[dict[str, str]]] = defaultdict(list)
    groups, accessions, windows = {}, {}, set()
    for row in rows:
        name = row["partition"]
        if name not in {"train", "validation", "internal_test"}:
            raise ValueError("Unexpected partition in revised benchmark")
        if full_by_id.get(row["window_id"]) != row:
            raise ValueError("Model row differs from full audited inventory")
        sequence = row["sequence"]
        position = int(row["position_1based"])
        if (len(sequence) != 51 or position != 26 or sequence[position - 1] != "K"
                or sequence in windows or row["label"] not in {"0", "1"}):
            raise ValueError("Invalid or repeated revised benchmark window")
        windows.add(sequence)
        group = row["candidate_group"]
        if not group or groups.setdefault(group, name) != name:
            raise ValueError("Candidate homology group crosses partitions")
        candidates = row["candidate_accessions"].split(";")
        if not candidates or any(not candidate for candidate in candidates):
            raise ValueError("Missing candidate accession")
        for candidate in candidates:
            if accessions.setdefault(candidate, name) != name:
                raise ValueError("Candidate accession crosses partitions")
        values = np.asarray([float(row[column]) for column in feature_columns], dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite feature value")
        partitions[name].append(row)
    for name in ("train", "validation", "internal_test"):
        expected = audit["partitions_before_training_resampling"][name]
        current = partitions[name]
        if not current or sum(row["label"] == "1" for row in current) != expected["positives"]:
            raise ValueError(f"Revised partition has unexpected positive count: {name}")
        if name != "train" and len(current) != expected["rows"]:
            raise ValueError(f"Revised partition has unexpected row count: {name}")
    if len(partitions["train"]) != audit["sampled_training_rows"]:
        raise ValueError("Training sample count differs from lock")
    if sum(row["label"] == "0" for row in partitions["train"]) > (
        audit["training_negative_ratio_cap"] *
        sum(row["label"] == "1" for row in partitions["train"])
    ):
        raise ValueError("Training negative sample exceeds declared cap")

    def make_split(group_rows: list[dict[str, str]]) -> KlaSplit:
        return KlaSplit(
            records=[KlaRecord(name=row["window_id"], position=int(row["position_1based"]),
                               label=int(row["label"]), sequence=row["sequence"])
                     for row in group_rows],
            feature_names=audit["feature_names"],
            features=np.asarray([[float(row[column]) for column in feature_columns]
                                 for row in group_rows], dtype=np.float32),
        )

    development = make_split(partitions["train"] + partitions["validation"])
    validation_indices = np.arange(len(partitions["train"]), len(development.records))
    return development, validation_indices, make_split(partitions["internal_test"])
