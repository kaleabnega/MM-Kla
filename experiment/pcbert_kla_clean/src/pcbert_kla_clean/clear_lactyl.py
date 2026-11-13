from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from pcbert_kla_clean.data import KlaRecord, KlaSplit
from pcbert_kla_clean.features import AMINO_ACIDS, FEATURE_NAMES, calculate_features


PARTITIONS = ("train", "validation", "internal_test")


def load_clear_lactyl(lock_path: Path) -> tuple[KlaSplit, np.ndarray, KlaSplit]:
    audit = json.loads(lock_path.read_text())
    if not audit.get("protocol", "").startswith("CLEAR-Lactyl protein-"):
        raise ValueError("Not a CLEAR-Lactyl split audit")
    manifest = lock_path.with_name("model_manifest.csv")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if digest != audit["output_sha256"]["model_manifest.csv"]:
        raise ValueError("CLEAR-Lactyl manifest does not match its split audit")
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    partitions: dict[str, list[dict[str, str]]] = defaultdict(list)
    proteins: dict[str, str] = {}
    groups: dict[str, str] = {}
    windows: set[str] = set()
    for row in rows:
        partition = row["partition"]
        if partition not in PARTITIONS:
            raise ValueError(f"Invalid CLEAR-Lactyl partition: {partition}")
        accession, group, window = row["accession"], row["homology_group"], row["window_51aa"]
        if not accession or not group or proteins.setdefault(accession, partition) != partition:
            raise ValueError("CLEAR-Lactyl protein crosses partitions or lacks identity")
        if groups.setdefault(group, partition) != partition:
            raise ValueError("CLEAR-Lactyl homology group crosses partitions")
        if len(window) != 51 or window[25] != "K" or set(window) - set(AMINO_ACIDS):
            raise ValueError("Invalid CLEAR-Lactyl sequence window")
        if window in windows:
            raise ValueError("Duplicate CLEAR-Lactyl sequence window")
        windows.add(window)
        if row["label"] not in {"0", "1"}:
            raise ValueError("Invalid CLEAR-Lactyl label")
        partitions[partition].append(row)
    for name in PARTITIONS:
        expected = audit["partitions"][name]
        actual = partitions[name]
        if (len(actual) != expected["rows"] or
                sum(row["label"] == "1" for row in actual) != expected["positives"] or
                not actual):
            raise ValueError(f"CLEAR-Lactyl {name} counts differ from split audit")

    def make_split(group_rows: list[dict[str, str]]) -> KlaSplit:
        return KlaSplit(
            records=[KlaRecord(
                name=row["accession"], position=int(row["position_1based"]),
                label=int(row["label"]), sequence=row["window_51aa"],
            ) for row in group_rows],
            feature_names=FEATURE_NAMES,
            features=np.stack([calculate_features(row["window_51aa"]) for row in group_rows]),
        )

    development = make_split(partitions["train"] + partitions["validation"])
    validation_indices = np.arange(len(partitions["train"]), len(development.records))
    return development, validation_indices, make_split(partitions["internal_test"])
