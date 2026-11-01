from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from pcbert_kla_clean.data import KlaRecord, KlaSplit
from pcbert_kla_clean.features import AMINO_ACIDS, FEATURE_NAMES, calculate_features


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def validate_rows(rows: list[dict[str, str]], name: str) -> None:
    if not rows:
        raise ValueError(f"{name} is empty")
    sites: set[tuple[str, int]] = set()
    windows: dict[str, int] = {}
    for row in rows:
        accession = row["accession"]
        position = int(row["position_1based"])
        sequence = row["window_51aa"]
        label = int(row["label"])
        if not accession or position < 1 or label not in (0, 1):
            raise ValueError(f"Invalid site or label in {name}: {row}")
        if len(sequence) != 51 or sequence[25] != "K" or any(aa not in AMINO_ACIDS for aa in sequence):
            raise ValueError(f"Invalid 51-aa lysine window in {name}: {row}")
        site = (accession, position)
        if site in sites:
            raise ValueError(f"Repeated accession-position in {name}: {site}")
        sites.add(site)
        if sequence in windows:
            raise ValueError(f"Repeated or conflicting window in {name}: {sequence}")
        windows[sequence] = label


def lock_external_manifest(gao_path: Path, rice_path: Path, output_dir: Path) -> dict[str, object]:
    gao_audit_path = gao_path.with_name("source_split_audit.json")
    rice_audit_path = rice_path.with_name("cross_study_homology_audit.json")
    gao_audit = json.loads(gao_audit_path.read_text())
    rice_audit = json.loads(rice_audit_path.read_text())
    if gao_audit["manifest_sha256"] != sha256(gao_path):
        raise ValueError("Gao manifest does not match source split audit")
    if rice_audit["output_sha256"]["filtered_candidates"] != sha256(rice_path):
        raise ValueError("Rice candidates do not match homology audit")
    if gao_audit["homology_search"]["cross_partition_hits"]:
        raise ValueError("Gao source split has cross-partition homology hits")
    gao = read_rows(gao_path)
    rice = read_rows(rice_path)
    validate_rows(gao, "Gao manifest")
    gao_windows = {row["window_51aa"] for row in gao}
    labels_by_window: dict[str, set[int]] = {}
    for row in rice:
        labels_by_window.setdefault(row["window_51aa"], set()).add(int(row["label"]))
    conflicts = [window for window, labels in labels_by_window.items() if len(labels) != 1]
    if conflicts:
        raise ValueError(f"Rice windows with contradictory labels: {len(conflicts)}")
    selected: dict[str, dict[str, str]] = {}
    for row in sorted(rice, key=lambda r: (r["window_51aa"], r["accession"], int(r["position_1based"]))):
        selected.setdefault(row["window_51aa"], row)
    external = sorted(selected.values(), key=lambda r: (r["accession"], int(r["position_1based"])))
    validate_rows(external, "rice external manifest")
    if gao_windows.intersection(selected):
        raise ValueError("Gao and rice share exact windows")
    output_dir.mkdir(parents=True, exist_ok=True)
    external_path = output_dir / "external_test_manifest.csv"
    with external_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rice[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(external)
    report = {
        "protocol": "Gao protein/homology-cluster split; rice study-held-out external test",
        "gao_manifest_sha256": sha256(gao_path),
        "gao_source_audit_sha256": sha256(gao_audit_path),
        "rice_screened_candidates_sha256": sha256(rice_path),
        "rice_homology_audit_sha256": sha256(rice_audit_path),
        "external_manifest_sha256": sha256(external_path),
        "rice_duplicate_window_policy": "retain lexicographically first accession, then lowest position",
        "rice_repeated_window_rows_removed": len(rice) - len(external),
        "external_rows": len(external),
        "external_label_counts": dict(sorted(Counter(row["label"] for row in external).items())),
        "negative_label_definition": "unannotated lysine; putative negative, not experimentally confirmed non-Kla",
        "model_selection": "Gao validation only; rice external test never used for selection",
    }
    (output_dir / "protocol_lock.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def load_locked_partitions(
    gao_path: Path, rice_path: Path, lock_path: Path
) -> tuple[KlaSplit, np.ndarray, KlaSplit, KlaSplit]:
    lock = json.loads(lock_path.read_text())
    external_path = lock_path.parent / "external_test_manifest.csv"
    for path, key in (
        (gao_path, "gao_manifest_sha256"),
        (gao_path.with_name("source_split_audit.json"), "gao_source_audit_sha256"),
        (rice_path, "rice_screened_candidates_sha256"),
        (rice_path.with_name("cross_study_homology_audit.json"), "rice_homology_audit_sha256"),
        (external_path, "external_manifest_sha256"),
    ):
        if sha256(path) != lock[key]:
            raise ValueError(f"Locked source file changed: {path}")
    gao = read_rows(gao_path)
    rice = read_rows(external_path)
    validate_rows(gao, "Gao manifest")
    validate_rows(rice, "rice external manifest")
    groups: dict[str, list[dict[str, str]]] = {name: [] for name in ("train", "validation", "test")}
    proteins: dict[str, str] = {}
    clusters: dict[str, str] = {}
    for row in gao:
        partition = row["partition"]
        if partition not in groups:
            raise ValueError(f"Unknown Gao partition: {partition}")
        if proteins.setdefault(row["accession"], partition) != partition:
            raise ValueError("Gao protein crosses partitions")
        if clusters.setdefault(row["homology_cluster"], partition) != partition:
            raise ValueError("Gao homology cluster crosses partitions")
        groups[partition].append(row)
    if not all(groups.values()):
        raise ValueError("All Gao partitions must be populated")
    if {row["accession"] for row in rice}.intersection(proteins):
        raise ValueError("Rice and Gao share protein accessions")
    if {row["window_51aa"] for row in rice}.intersection(row["window_51aa"] for row in gao):
        raise ValueError("Rice and Gao share exact windows")

    def make_split(rows: list[dict[str, str]]) -> KlaSplit:
        return KlaSplit(
            records=[KlaRecord(
                name=row["accession"],
                position=int(row["position_1based"]),
                label=int(row["label"]),
                sequence=row["window_51aa"],
            ) for row in rows],
            feature_names=FEATURE_NAMES,
            features=np.stack([calculate_features(row["window_51aa"]) for row in rows]),
        )

    development = make_split(groups["train"] + groups["validation"])
    validation_indices = np.arange(len(groups["train"]), len(development.records))
    return development, validation_indices, make_split(groups["test"]), make_split(rice)
