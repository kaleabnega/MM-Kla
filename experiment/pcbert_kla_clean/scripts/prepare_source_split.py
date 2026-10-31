from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
SOURCE_DOI = "10.3389/fmicb.2020.594743"

from pcbert_kla_clean.candidate_groups import (  # noqa: E402
    _choose_group_partitions,
    audit_homology_hits,
    load_homology_clusters,
)


def load_candidates(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"accession", "position_1based", "window_51aa", "label", "evidence", "source_doi"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Candidate file must contain {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Candidate file is empty")
    seen_sites: set[tuple[str, int]] = set()
    labels_by_window: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        accession = row["accession"].strip()
        position = int(row["position_1based"])
        window = row["window_51aa"].strip().upper()
        label = int(row["label"])
        evidence = row["evidence"]
        if row["source_doi"] != SOURCE_DOI:
            raise ValueError("Candidate contains a site from a different source study")
        if not accession or position < 1 or len(window) != 51 or window[25] != "K":
            raise ValueError("Candidate has an invalid accession, position, or 51-aa window")
        if (label, evidence) not in {
            (1, "reported_Kla_site"), (0, "unannotated_lysine_putative_negative")
        }:
            raise ValueError("Candidate label and evidence disagree")
        if (accession, position) in seen_sites:
            raise ValueError("Candidate accession-position appears more than once")
        seen_sites.add((accession, position))
        labels_by_window[window].add(label)
    if any(len(labels) > 1 for labels in labels_by_window.values()):
        raise ValueError("Identical windows have contradictory labels")
    return rows


def verify_reconstruction(candidate_csv: Path, audit_path: Path) -> dict[str, str]:
    reconstruction = json.loads(audit_path.read_text())
    candidate_sha256 = hashlib.sha256(candidate_csv.read_bytes()).hexdigest()
    if reconstruction.get("output_sha256", {}).get("candidate_sites.csv") != candidate_sha256:
        raise ValueError("Candidate CSV does not match its reconstruction audit")
    return reconstruction["input_sha256"]


def prepare_split(
    rows: list[dict[str, str]],
    clusters: dict[str, str],
    hits_path: Path,
    *,
    seed: int = 42,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.1,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    accessions = {row["accession"] for row in rows}
    missing = accessions - set(clusters)
    if missing:
        raise ValueError(f"Homology clusters omit {len(missing)} source accessions")
    if not 0 < test_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Holdout fractions must be between zero and one")
    if test_fraction + validation_fraction >= 1:
        raise ValueError("Holdout fractions leave no training partition")

    labels = np.array([int(row["label"]) for row in rows], dtype=int)
    group_ids = [clusters[row["accession"]] for row in rows]
    selected = _choose_group_partitions(
        labels, group_ids, seed, test_fraction, validation_fraction
    )
    assignments = {
        int(index): name for name, indices in selected.items() for index in indices
    }
    if set(assignments) != set(range(len(rows))):
        raise AssertionError("Partitioning omitted or duplicated candidate rows")
    split_rows = [
        {**row, "homology_cluster": group_ids[index], "partition": assignments[index]}
        for index, row in enumerate(rows)
    ]
    group_partitions: dict[str, set[str]] = defaultdict(set)
    accession_partitions: dict[str, set[str]] = defaultdict(set)
    for row in split_rows:
        group_partitions[row["homology_cluster"]].add(row["partition"])
        accession_partitions[row["accession"]].add(row["partition"])
    if any(len(names) != 1 for names in group_partitions.values()):
        raise AssertionError("A homology cluster crosses partitions")
    if any(len(names) != 1 for names in accession_partitions.values()):
        raise AssertionError("A protein crosses partitions")

    hit_audit = audit_homology_hits(
        hits_path,
        set(clusters),
        {accession: next(iter(names)) for accession, names in accession_partitions.items()},
    )
    by_partition = {}
    for name in ("train", "validation", "test"):
        part_rows = [row for row in split_rows if row["partition"] == name]
        counts = Counter(int(row["label"]) for row in part_rows)
        by_partition[name] = {
            "sites": len(part_rows),
            "reported_positive_sites": counts[1],
            "putative_negative_sites": counts[0],
            "proteins": len({row["accession"] for row in part_rows}),
            "homology_clusters": len({row["homology_cluster"] for row in part_rows}),
        }
    report = {
        "protocol": "source-derived accession and homology-cluster-disjoint internal split",
        "seed": seed,
        "requested_test_fraction": test_fraction,
        "requested_validation_fraction": validation_fraction,
        "partitions": by_partition,
        "homology_search": hit_audit,
        "limitations": [
            "Negative sites are unannotated lysines, not experimentally verified non-lactylated sites.",
            "Homology screen uses the supplied MMseqs2 threshold; weaker or local similarities may remain.",
            "This is an internal split of Gao 2020 data, not an independent experimental study.",
            "The candidate negatives have not yet been screened against an independent Kla site database.",
            "No model training or test-set evaluation was performed by this script.",
        ],
    }
    return split_rows, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Lock a source-derived, protein-group-disjoint split")
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--reconstruction-audit", type=Path, required=True)
    parser.add_argument("--homology-csv", type=Path, required=True)
    parser.add_argument("--homology-hits-tsv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    args = parser.parse_args()

    raw_source_sha256 = verify_reconstruction(args.candidate_csv, args.reconstruction_audit)
    rows = load_candidates(args.candidate_csv)
    clusters = load_homology_clusters(args.homology_csv)
    split_rows, report = prepare_split(
        rows, clusters, args.homology_hits_tsv,
        seed=args.seed,
        test_fraction=args.test_fraction,
        validation_fraction=args.validation_fraction,
    )
    report["input_sha256"] = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in (
            ("candidate_csv", args.candidate_csv),
            ("reconstruction_audit", args.reconstruction_audit),
            ("homology_csv", args.homology_csv),
            ("homology_hits_tsv", args.homology_hits_tsv),
        )
    }
    report["raw_source_sha256"] = raw_source_sha256
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "source_split_manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(split_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(split_rows)
    report["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (args.output_dir / "source_split_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["partitions"], indent=2))


if __name__ == "__main__":
    main()
