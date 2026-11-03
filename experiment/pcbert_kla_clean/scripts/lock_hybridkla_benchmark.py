"""Prepare a provisional protein-disjoint benchmark from HybridKla's workbook.

The workbook provides accession-position strings but no study evidence or
independent mouse-test records. This lock therefore has an internal test only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from audit_hybridkla_overlap import read_workbook_rows


WORKBOOK_SHA256 = "e813ed5e495b63539a409bf66d70f5279d1308dfda65a1da181c39eabb4aa449"
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
COLUMNS = ("accession", "position_1based", "window_51aa", "label",
           "source_site_id", "group_id", "partition")
PARTITIONS = ("train", "validation", "internal_test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_records(source: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict]:
    sites = {}
    windows = {}
    excluded = Counter()
    cleaned = []
    for row in source:
        site_id, window, label = row["UniProt_ID"], row["Sequence"], row["Label"]
        match = re.fullmatch(r"([^_]+)_([1-9]\d*)", site_id)
        if match is None or label not in {"0", "1"} or len(window) != 51 or window[25] != "K":
            raise ValueError(f"Malformed source record: {site_id}")
        accession, position = match.group(1), int(match.group(2))
        if site_id in sites:
            if sites[site_id] != (window, label):
                raise ValueError(f"Conflicting source labels or windows for site {site_id}")
            excluded[f"repeated_site_label_{label}"] += 1
            continue
        sites[site_id] = (window, label)
        if window in windows:
            if windows[window] != label:
                raise ValueError(f"Conflicting labels for sequence window at {site_id}")
            excluded[f"repeated_window_label_{label}"] += 1
            continue
        windows[window] = label
        if set(window) - AMINO_ACIDS:
            excluded[f"noncanonical_label_{label}"] += 1
            continue
        cleaned.append({"accession": accession, "position_1based": str(position),
                        "window_51aa": window, "label": label,
                        "source_site_id": site_id})
    if not cleaned or sum(excluded.values()) + len(cleaned) != len(source):
        raise ValueError("Source attrition accounting failed")
    return cleaned, {"source_rows": len(source), "eligible_rows": len(cleaned),
                     "excluded": dict(sorted(excluded.items())),
                     "eligible_labels": dict(sorted(Counter(r["label"] for r in cleaned).items()))}


def assign_protein_groups(rows: list[dict[str, str]], seed: int = 42) -> tuple[list[dict[str, str]], dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["accession"]].append(row)
    if len(groups) < 10:
        raise ValueError("At least ten accession groups are needed")
    names = sorted(groups)
    totals = {group: len(members) for group, members in groups.items()}
    positives = {group: sum(row["label"] == "1" for row in members)
                 for group, members in groups.items()}
    total_rows = len(rows)
    total_positives = sum(positives.values())
    generator = random.Random(seed)
    best = None
    for _ in range(200):
        order = names.copy()
        generator.shuffle(order)
        selections = {name: [] for name in PARTITIONS}
        index = 0
        for partition, fraction in (("internal_test", 0.2), ("validation", 0.1)):
            count = 0
            while index < len(order) and count < total_rows * fraction:
                group = order[index]
                selections[partition].append(group)
                count += totals[group]
                index += 1
        selections["train"] = order[index:]
        if any(not selections[name] for name in PARTITIONS):
            continue
        counts = {name: (sum(totals[group] for group in selected),
                         sum(positives[group] for group in selected))
                  for name, selected in selections.items()}
        if any(pos == 0 or pos == size for size, pos in counts.values()):
            continue
        objective = sum(abs(counts[name][0] / total_rows - target)
                        + abs(counts[name][1] / total_positives - target)
                        for name, target in (("internal_test", 0.2), ("validation", 0.1)))
        if best is None or objective < best[0]:
            best = (objective, selections)
    if best is None:
        raise ValueError("Could not assign accession groups with both classes")
    group_partition = {group: partition for partition, selection in best[1].items()
                       for group in selection}
    locked = sorted(({**row, "group_id": row["accession"],
                      "partition": group_partition[row["accession"]]} for row in rows),
                    key=lambda row: (PARTITIONS.index(row["partition"]), row["accession"],
                                     int(row["position_1based"])))
    if len({row["window_51aa"] for row in locked}) != len(locked):
        raise AssertionError("Exact window repeated across locked partitions")
    counts = {name: {"rows": sum(row["partition"] == name for row in locked),
                     "positives": sum(row["partition"] == name and row["label"] == "1"
                                      for row in locked),
                     "accessions": len({row["accession"] for row in locked
                                        if row["partition"] == name})}
              for name in PARTITIONS}
    return locked, {"split_seed": seed, "split_method": "accession-grouped 70/10/20; exact-window-disjoint",
                    "partitions": counts, "selection_objective": best[0]}


def lock_benchmark(workbook: Path, output_dir: Path, seed: int = 42,
                   expected_sha256: str = WORKBOOK_SHA256) -> dict:
    workbook_hash = sha256(workbook)
    if workbook_hash != expected_sha256:
        raise ValueError("HybridKla workbook differs from the pinned official archive")
    eligible, source_audit = clean_records(read_workbook_rows(workbook))
    locked, split_audit = assign_protein_groups(eligible, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "model_manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(locked)
    report = {
        "benchmark_status": "provisional internal benchmark; source accessions and study provenance unverified",
        "source": "HybridKla official code archive, train/train_data.xlsx",
        "workbook_sha256": workbook_hash,
        "model_manifest_sha256": sha256(manifest),
        "source_audit": source_audit,
        **split_audit,
        "negative_definition": "HybridKla workbook label 0; putative non-Kla, not experimentally verified",
        "external_test": "not available in official code archive; no external score can be claimed",
        "homology_scope": "accession grouping and exact-window disjointness only; full-protein homology not verified",
        "model_selection": "train/validation only; internal_test must not guide tuning",
    }
    (output_dir / "split_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(lock_benchmark(args.workbook, args.output_dir, args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
