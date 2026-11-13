from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from lock_kladb_human import connected_proteins
from prepare_kladb_human import CANDIDATE_COLUMNS, sha256


PARTITIONS = ("train", "validation", "internal_test")
FRACTIONS = {"train": 0.7, "validation": 0.1, "internal_test": 0.2}


def split_groups(
    rows: list[dict[str, str]], components: dict[str, str], seed: int
) -> tuple[list[dict[str, str]], dict[str, object]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[components[row["accession"]]].append(row)
    group_stats = {
        group: (len(members), sum(row["label"] == "1" for row in members))
        for group, members in groups.items()
    }
    total_rows = len(rows)
    total_positive = sum(row["label"] == "1" for row in rows)
    randomizer = random.Random(seed)
    best: tuple[float, dict[str, list[str]]] | None = None
    for _ in range(500):
        order = sorted(groups)
        randomizer.shuffle(order)
        selected: dict[str, list[str]] = {name: [] for name in PARTITIONS}
        cursor = 0
        for name in ("internal_test", "validation"):
            count = 0
            while cursor < len(order) and count < total_rows * FRACTIONS[name]:
                group = order[cursor]
                selected[name].append(group)
                count += group_stats[group][0]
                cursor += 1
        selected["train"] = order[cursor:]
        if any(not selected[name] for name in PARTITIONS):
            continue
        stats = {
            name: (
                sum(group_stats[group][0] for group in members),
                sum(group_stats[group][1] for group in members),
            ) for name, members in selected.items()
        }
        if any(positive == 0 or positive == count for count, positive in stats.values()):
            continue
        score = sum(
            abs(count / total_rows - FRACTIONS[name])
            + abs(positive / total_positive - FRACTIONS[name])
            for name, (count, positive) in stats.items()
        )
        if best is None or score < best[0]:
            best = score, selected
    if best is None:
        raise ValueError("Could not assign homology groups to three two-class partitions")

    assignment = {
        group: name for name, members in best[1].items() for group in members
    }
    locked = [
        {**row, "homology_group": components[row["accession"]],
         "partition": assignment[components[row["accession"]]]}
        for row in rows
    ]
    locked.sort(key=lambda row: (
        PARTITIONS.index(row["partition"]), row["accession"], int(row["position_1based"])
    ))
    summary = {
        name: {
            "rows": sum(row["partition"] == name for row in locked),
            "positives": sum(row["partition"] == name and row["label"] == "1" for row in locked),
            "putative_negatives": sum(row["partition"] == name and row["label"] == "0" for row in locked),
            "proteins": len({row["accession"] for row in locked if row["partition"] == name}),
            "homology_groups": len({row["homology_group"] for row in locked if row["partition"] == name}),
        } for name in PARTITIONS
    }
    return locked, summary


def lock(candidate_csv: Path, preparation_audit: Path, hits_tsv: Path,
         output_dir: Path, seed: int) -> dict[str, object]:
    preparation = json.loads(preparation_audit.read_text())
    if sha256(candidate_csv) != preparation["output_sha256"]["candidate_sites.csv"]:
        raise ValueError("Candidate inventory differs from its preparation audit")
    with candidate_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    components, homology = connected_proteins(rows, hits_tsv)
    locked, summary = split_groups(rows, components, seed)
    by_window = defaultdict(set)
    by_accession = defaultdict(set)
    by_group = defaultdict(set)
    for row in locked:
        by_window[row["window_51aa"]].add(row["partition"])
        by_accession[row["accession"]].add(row["partition"])
        by_group[row["homology_group"]].add(row["partition"])
    if any(len(partitions) != 1 for inventory in (by_window, by_accession, by_group)
           for partitions in inventory.values()):
        raise AssertionError("Cross-partition window, protein, or homology overlap")
    if len(by_window) != len(rows):
        raise ValueError("Candidate inventory has repeated windows")

    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = (*CANDIDATE_COLUMNS, "homology_group", "partition")
    for name in ("full_manifest.csv", "model_manifest.csv"):
        with (output_dir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(locked)
    report: dict[str, object] = {
        "protocol": "CLEAR-Lactyl protein- and full-protein-homology-grouped 70/10/20 split",
        "seed": seed,
        "requested_fractions": FRACTIONS,
        "partitions": summary,
        "homology": homology,
        "negative_definition": preparation["negative_definition"],
        "study_independent_external_test": False,
        "limitations": [
            "The authors' CSV lacks per-site study provenance; the internal test is protein-group-held-out, not study-held-out.",
            "Unreported lysines are putative negatives and may include undiscovered Kla sites.",
            "The homology screen excludes qualifying full-length hits, not all local domain similarity.",
            "No partition should be used for model selection except validation.",
        ],
        "source_sha256": {
            "candidate_csv": sha256(candidate_csv),
            "preparation_audit": sha256(preparation_audit),
            "homology_hits_tsv": sha256(hits_tsv),
        },
        "output_sha256": {
            name: sha256(output_dir / name)
            for name in ("full_manifest.csv", "model_manifest.csv")
        },
    }
    (output_dir / "split_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Lock protein-grouped CLEAR-Lactyl partitions")
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--preparation-audit", type=Path, required=True)
    parser.add_argument("--homology-hits-tsv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = lock(args.candidate_csv, args.preparation_audit, args.homology_hits_tsv,
                  args.output_dir, args.seed)
    print(json.dumps(report["partitions"], indent=2))


if __name__ == "__main__":
    main()
