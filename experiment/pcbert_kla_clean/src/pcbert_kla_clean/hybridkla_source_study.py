"""Load the provisional HybridKla-source internal benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from pcbert_kla_clean.data import KlaRecord, KlaSplit
from pcbert_kla_clean.features import AMINO_ACIDS, FEATURE_NAMES, calculate_features


PARTITIONS = ("train", "validation", "internal_test")


def load_hybridkla_source_study(lock_path: Path) -> tuple[KlaSplit, np.ndarray, KlaSplit]:
    audit = json.loads(lock_path.read_text())
    if audit.get("benchmark_status") != "provisional internal benchmark; source accessions and study provenance unverified":
        raise ValueError("Unexpected HybridKla source benchmark status")
    manifest = lock_path.with_name("model_manifest.csv")
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != audit["model_manifest_sha256"]:
        raise ValueError("HybridKla source manifest does not match its lock")
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    partitions = {name: [] for name in PARTITIONS}
    accession_partition = {}
    site_ids = set()
    windows = set()
    for row in rows:
        partition = row["partition"]
        if partition not in partitions:
            raise ValueError(f"Unexpected HybridKla partition: {partition}")
        accession, window = row["accession"], row["window_51aa"]
        if not accession or row["group_id"] != accession:
            raise ValueError("HybridKla source accession/group mismatch")
        if accession_partition.setdefault(accession, partition) != partition:
            raise ValueError("HybridKla source accession crosses partitions")
        if len(window) != 51 or window[25] != "K" or set(window) - set(AMINO_ACIDS):
            raise ValueError("HybridKla source contains a noncanonical window")
        if window in windows or row["source_site_id"] in site_ids:
            raise ValueError("HybridKla source has a repeated site or window")
        windows.add(window)
        site_ids.add(row["source_site_id"])
        if row["source_site_id"] != f"{accession}_{row['position_1based']}":
            raise ValueError("HybridKla source site ID does not match accession and position")
        if row["label"] not in {"0", "1"}:
            raise ValueError("HybridKla source label is not binary")
        partitions[partition].append(row)
    for name in PARTITIONS:
        expected = audit["partitions"][name]
        actual = {"rows": len(partitions[name]),
                  "positives": sum(row["label"] == "1" for row in partitions[name]),
                  "accessions": len({row["accession"] for row in partitions[name]})}
        if actual != expected or not 0 < actual["positives"] < actual["rows"]:
            raise ValueError(f"HybridKla source {name} counts disagree with lock")

    def make_split(group_rows: list[dict[str, str]]) -> KlaSplit:
        return KlaSplit(
            records=[KlaRecord(name=row["accession"],
                               position=int(row["position_1based"]),
                               label=int(row["label"]),
                               sequence=row["window_51aa"]) for row in group_rows],
            feature_names=FEATURE_NAMES,
            features=np.stack([calculate_features(row["window_51aa"]) for row in group_rows]),
        )

    development = make_split(partitions["train"] + partitions["validation"])
    validation_indices = np.arange(len(partitions["train"]), len(development.records))
    return development, validation_indices, make_split(partitions["internal_test"])
