"""Lock a human S1 development / mouse S7 external Kla benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from lock_kladb_human import connected_proteins, training_sample
from prepare_hybridkla_study import COLUMNS
from prepare_kladb_human import sha256


PARTITIONS = ("train", "validation", "internal_test", "external_test")


def assign_partitions(rows: list[dict[str, str]], components: dict[str, str], seed: int):
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[components[row["protein_id"]]].append(row)
    types = {group: {row["partition_eligibility"] for row in records}
             for group, records in grouped.items()}
    dev_groups = sorted(group for group, kinds in types.items() if kinds == {"development"})
    ext_groups = sorted(group for group, kinds in types.items() if kinds == {"external_candidate"})
    mixed_groups = {group for group, kinds in types.items() if len(kinds) > 1}
    if not dev_groups or not ext_groups:
        raise ValueError("Homology screening leaves no development or external-only proteins")

    group_sizes = {group: len(grouped[group]) for group in dev_groups}
    group_positives = {group: sum(row["label"] == "1" for row in grouped[group])
                       for group in dev_groups}
    total_size = sum(group_sizes.values())
    total_positives = sum(group_positives.values())
    rng = random.Random(seed)
    best = None
    for _ in range(200):
        order = dev_groups.copy()
        rng.shuffle(order)
        allocation = {name: [] for name in PARTITIONS[:3]}
        cursor = 0
        for name, target in (("internal_test", .2), ("validation", .1)):
            current = 0
            while cursor < len(order) and current < total_size * target:
                group = order[cursor]
                allocation[name].append(group)
                current += group_sizes[group]
                cursor += 1
        allocation["train"] = order[cursor:]
        if any(not allocation[name] for name in PARTITIONS[:3]):
            continue
        counts = {name: (
            sum(group_sizes[group] for group in groups),
            sum(group_positives[group] for group in groups),
        ) for name, groups in allocation.items()}
        if any(positives == 0 or positives == size for size, positives in counts.values()):
            continue
        score = sum(
            abs(counts[name][0] / total_size - target)
            + abs(counts[name][1] / total_positives - target)
            for name, target in (("internal_test", .2), ("validation", .1))
        )
        if best is None or score < best[0]:
            best = (score, allocation)
    if best is None:
        raise ValueError("Cannot form class-complete development partitions")
    by_group = {group: name for name, groups in best[1].items() for group in groups}
    by_group.update({group: "external_test" for group in ext_groups})
    retained = [
        {**row, "homology_group": components[row["protein_id"]],
         "partition": by_group[components[row["protein_id"]]]}
        for row in rows if components[row["protein_id"]] in by_group
    ]
    unique: dict[str, dict[str, str]] = {}
    for row in sorted(retained, key=lambda item: (
        item["window_51aa"], item["protein_id"], int(item["position_1based"])
    )):
        previous = unique.setdefault(row["window_51aa"], row)
        if previous["partition"] != row["partition"] or previous["label"] != row["label"]:
            raise AssertionError("Duplicate window crosses partition or label")
    locked = sorted(unique.values(), key=lambda row: (
        row["partition"], row["protein_id"], int(row["position_1based"])
    ))
    counts = {
        partition: {
            "rows": len(partition_rows := [row for row in locked if row["partition"] == partition]),
            "positives": sum(row["label"] == "1" for row in partition_rows),
            "putative_negatives": sum(row["label"] == "0" for row in partition_rows),
            "proteins": len({row["protein_id"] for row in partition_rows}),
            "homology_groups": len({row["homology_group"] for row in partition_rows}),
        } for partition in PARTITIONS
    }
    if any(item["positives"] == 0 or item["putative_negatives"] == 0
           for item in counts.values()):
        raise ValueError("A locked partition has only one class")
    report = {
        "protocol": "Human S1 development, mouse S7 external; full-protein homology and exact-window disjointness",
        "study_provenance": "S7 designated external by HybridKla paper; no per-site PMID in S1/S7",
        "split_seed": seed,
        "homology_overlap_excluded": {
            "groups": len(mixed_groups),
            "human_rows": sum(row["partition_eligibility"] == "development"
                              for group in mixed_groups for row in grouped[group]),
            "mouse_rows": sum(row["partition_eligibility"] == "external_candidate"
                              for group in mixed_groups for row in grouped[group]),
        },
        "same_partition_duplicate_windows_removed": len(retained) - len(locked),
        "partitions_before_training_resampling": counts,
        "negative_definition": "Unreported lysines in positive-bearing proteins; putative, not verified negatives",
        "model_selection": "Use train and validation only; internal and external tests are untouched until final evaluation",
    }
    return locked, report


def lock(candidate_csv: Path, candidate_fasta: Path, preparation_audit: Path, hits_path: Path,
         output_dir: Path, seed: int = 42, negative_ratio: int = 3) -> dict:
    preparation = json.loads(preparation_audit.read_text())
    if sha256(candidate_csv) != preparation["output_sha256"]["candidate_sites.csv"]:
        raise ValueError("Candidate CSV does not match preparation audit")
    if sha256(candidate_fasta) != preparation["output_sha256"]["candidate_proteins.fasta"]:
        raise ValueError("Candidate FASTA does not match preparation audit")
    with candidate_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or set(rows[0]) != set(COLUMNS):
        raise ValueError("Invalid HybridKla candidate inventory")
    if any(row["partition_eligibility"] not in {"development", "external_candidate"}
           for row in rows):
        raise ValueError("Unknown source eligibility")
    if any((row["source_table"], row["species"]) not in {
        ("S1", "Homo sapiens"), ("S7", "Mus musculus")
    } for row in rows):
        raise ValueError("HybridKla source and species mismatch")
    mapped = [{**row, "accession": row["protein_id"]} for row in rows]
    components, homology = connected_proteins(mapped, hits_path)
    locked, report = assign_partitions(rows, components, seed)
    sampled = training_sample(locked, negative_ratio, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, records in (("full_manifest.csv", locked), ("model_manifest.csv", sampled)):
        path = output_dir / name
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=(*COLUMNS, "homology_group", "partition"),
                                    lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
    report.update({
        "homology": homology,
        "mmseqs_threshold": "30% identity, >=80% bilateral full-protein coverage",
        "training_negative_ratio_cap": negative_ratio,
        "sampled_training_rows": sum(row["partition"] == "train" for row in sampled),
        "source_sha256": {
            "candidate_csv": sha256(candidate_csv),
            "candidate_fasta": sha256(candidate_fasta),
            "preparation_audit": sha256(preparation_audit),
            "homology_hits_tsv": sha256(hits_path),
        },
        "output_sha256": {name: sha256(output_dir / name)
                          for name in ("full_manifest.csv", "model_manifest.csv")},
    })
    (output_dir / "split_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--candidate-fasta", type=Path, required=True)
    parser.add_argument("--preparation-audit", type=Path, required=True)
    parser.add_argument("--homology-hits-tsv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-negative-ratio", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(lock(args.candidate_csv, args.candidate_fasta, args.preparation_audit,
                          args.homology_hits_tsv, args.output_dir,
                          args.seed, args.train_negative_ratio)["partitions_before_training_resampling"],
                     indent=2))


if __name__ == "__main__":
    main()
