from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from pcbert_kla_clean.data import KlaRecord, KlaSplit
from pcbert_kla_clean.features import AMINO_ACIDS, FEATURE_NAMES, calculate_features


def load_hybridkla_study(lock_path: Path) -> tuple[KlaSplit, np.ndarray, KlaSplit, KlaSplit]:
    lock = json.loads(lock_path.read_text())
    manifest = lock_path.with_name("model_manifest.csv")
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != lock["output_sha256"]["model_manifest.csv"]:
        raise ValueError("HybridKla study manifest does not match split audit")
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    partitions: dict[str, list[dict[str, str]]] = defaultdict(list)
    proteins, groups, windows = {}, {}, set()
    for row in rows:
        part = row["partition"]
        if part not in {"train", "validation", "internal_test", "external_test"}:
            raise ValueError("Invalid HybridKla study partition")
        protein, group, window = row["protein_id"], row["homology_group"], row["window_51aa"]
        if not protein or not group or proteins.setdefault(protein, part) != part:
            raise ValueError("Protein crosses partitions")
        if groups.setdefault(group, part) != part:
            raise ValueError("Homology group crosses partitions")
        if len(window) != 51 or window[25] != "K" or set(window) - set(AMINO_ACIDS):
            raise ValueError("Invalid sequence window")
        if window in windows:
            raise ValueError("Exact sequence window repeats")
        windows.add(window)
        if row["label"] not in {"0", "1"}:
            raise ValueError("Invalid label")
        expected = ("S7", "Mus musculus", "external_candidate") if part == "external_test" else (
            "S1", "Homo sapiens", "development")
        if (row["source_table"], row["species"], row["partition_eligibility"]) != expected:
            raise ValueError("Source species does not match partition")
        partitions[part].append(row)
    if any(not partitions[name] for name in ("train", "validation", "internal_test", "external_test")):
        raise ValueError("HybridKla study requires four nonempty partitions")
    for name, group_rows in partitions.items():
        expected = lock["partitions_before_training_resampling"][name]
        if name != "train" and (len(group_rows) != expected["rows"] or
                                 sum(row["label"] == "1" for row in group_rows) != expected["positives"]):
            raise ValueError(f"Partition count mismatch: {name}")

    def make_split(group_rows: list[dict[str, str]]) -> KlaSplit:
        return KlaSplit(
            records=[KlaRecord(name=row["protein_id"], position=int(row["position_1based"]),
                               label=int(row["label"]), sequence=row["window_51aa"])
                     for row in group_rows],
            feature_names=FEATURE_NAMES,
            features=np.stack([calculate_features(row["window_51aa"]) for row in group_rows]),
        )

    development = make_split(partitions["train"] + partitions["validation"])
    validation_indices = np.arange(len(partitions["train"]), len(development.records))
    return (development, validation_indices, make_split(partitions["internal_test"]),
            make_split(partitions["external_test"]))
