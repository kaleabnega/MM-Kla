from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from pcbert_kla_clean.data import KlaRecord, KlaSplit
from pcbert_kla_clean.features import AMINO_ACIDS, FEATURE_NAMES, calculate_features


def load_human_study(lock_path: Path) -> tuple[KlaSplit, np.ndarray, KlaSplit, KlaSplit]:
    lock = json.loads(lock_path.read_text())
    manifest_path = lock_path.parent / "model_manifest.csv"
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if digest != lock["output_sha256"]["model_manifest.csv"]:
        raise ValueError("Human model manifest does not match its split audit")
    with manifest_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Human model manifest is empty")
    partitions: dict[str, list[dict[str, str]]] = defaultdict(list)
    proteins: dict[str, str] = {}
    groups: dict[str, str] = {}
    windows: dict[str, str] = {}
    for row in rows:
        partition = row["partition"]
        if partition not in {"train", "validation", "internal_test", "external_test"}:
            raise ValueError(f"Unknown human partition: {partition}")
        accession, group = row["accession"], row["homology_group"]
        if not accession or not group:
            raise ValueError("Human manifest lacks accession or homology group")
        if proteins.setdefault(accession, partition) != partition:
            raise ValueError("Human accession crosses partitions")
        if groups.setdefault(group, partition) != partition:
            raise ValueError("Human homology group crosses partitions")
        sequence = row["window_51aa"]
        if len(sequence) != 51 or sequence[25] != "K" or set(sequence) - set(AMINO_ACIDS):
            raise ValueError("Human manifest contains invalid 51-residue window")
        if sequence in windows:
            raise ValueError("Human manifest repeats an exact window")
        windows[sequence] = partition
        if row["label"] not in {"0", "1"}:
            raise ValueError("Human manifest contains invalid label")
        if partition == "external_test":
            if row["label"] == "1" and row["source_pmids"] != lock["holdout_pmid"]:
                raise ValueError("External positive is not exclusive to the held-out study")
        elif lock["holdout_pmid"] in row["source_pmids"].split(";"):
            raise ValueError("Held-out study evidence leaked into development")
        partitions[partition].append(row)
    if any(not partitions[name] for name in ("train", "validation", "internal_test", "external_test")):
        raise ValueError("Human study requires all four partitions")

    def make_split(group_rows: list[dict[str, str]]) -> KlaSplit:
        return KlaSplit(
            records=[KlaRecord(
                name=row["accession"],
                position=int(row["position_1based"]),
                label=int(row["label"]),
                sequence=row["window_51aa"],
            ) for row in group_rows],
            feature_names=FEATURE_NAMES,
            features=np.stack([calculate_features(row["window_51aa"]) for row in group_rows]),
        )

    development = make_split(partitions["train"] + partitions["validation"])
    validation_indices = np.arange(len(partitions["train"]), len(development.records))
    return (
        development, validation_indices,
        make_split(partitions["internal_test"]),
        make_split(partitions["external_test"]),
    )
