from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from prepare_kladb_human import CANDIDATE_COLUMNS, sha256


def connected_proteins(
    rows: list[dict[str, str]], hits_path: Path
) -> tuple[dict[str, str], dict[str, int]]:
    accessions = {row["accession"] for row in rows}
    parent = {accession: accession for accession in accessions}

    def root(accession: str) -> str:
        while parent[accession] != accession:
            parent[accession] = parent[parent[accession]]
            accession = parent[accession]
        return accession

    def join(left: str, right: str) -> None:
        left_root, right_root = root(left), root(right)
        parent[max(left_root, right_root)] = min(left_root, right_root)

    self_hits: set[str] = set()
    hits = 0
    with hits_path.open(newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) != 5:
                raise ValueError("Homology TSV requires query,target,pident,qcov,tcov")
            left, right = row[:2]
            if left not in accessions or right not in accessions:
                raise ValueError("Homology hit references a protein outside the candidate inventory")
            identity, query_coverage, target_coverage = map(float, row[2:])
            if identity < 30 or query_coverage < 0.8 or target_coverage < 0.8:
                raise ValueError("Homology hit is below 30% identity or 80% bilateral coverage")
            if left == right:
                self_hits.add(left)
            join(left, right)
            hits += 1
    if self_hits != accessions:
        raise ValueError(f"Homology TSV lacks self-hits for {len(accessions - self_hits)} proteins")

    first_by_window: dict[str, str] = {}
    repeated_windows = 0
    labels_by_window: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        window = row["window_51aa"]
        labels_by_window[window].add(int(row["label"]))
        if window in first_by_window:
            repeated_windows += 1
            join(row["accession"], first_by_window[window])
        else:
            first_by_window[window] = row["accession"]
    if any(len(labels) > 1 for labels in labels_by_window.values()):
        raise ValueError("Positive and putative negative candidates share an exact window")
    return {accession: root(accession) for accession in accessions}, {
        "qualifying_homology_hits": hits,
        "self_hits": len(self_hits),
        "repeated_window_rows": repeated_windows,
        "connected_protein_groups": len({root(accession) for accession in accessions}),
    }


def assign_groups(
    rows: list[dict[str, str]], components: dict[str, str], holdout_pmid: str,
    *, seed: int = 42, internal_test_fraction: float = 0.2,
    validation_fraction: float = 0.1,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    if not 0 < internal_test_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Development fractions must be in (0, 1)")
    if internal_test_fraction + validation_fraction >= 1:
        raise ValueError("Development fractions leave no training partition")
    protein_studies: dict[str, set[str]] = {}
    for row in rows:
        studies = set(row["protein_source_pmids"].split(";"))
        if "" in studies or (row["accession"] in protein_studies and protein_studies[row["accession"]] != studies):
            raise ValueError("Protein has missing or inconsistent study provenance")
        protein_studies[row["accession"]] = studies
    if holdout_pmid not in set().union(*protein_studies.values()):
        raise ValueError(f"Holdout PMID {holdout_pmid} has no candidates")

    by_component: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_component[components[row["accession"]]].append(row)
    category: dict[str, str] = {}
    for component, members in by_component.items():
        statuses = {
            "external" if protein_studies[row["accession"]] == {holdout_pmid}
            else "mixed" if holdout_pmid in protein_studies[row["accession"]]
            else "development"
            for row in members
        }
        category[component] = next(iter(statuses)) if len(statuses) == 1 else "excluded"

    dev_groups = sorted(component for component, kind in category.items() if kind == "development")
    if not dev_groups or not any(kind == "external" for kind in category.values()):
        raise ValueError("Study-exclusive external or development groups are empty")
    group_totals = {group: len(by_component[group]) for group in dev_groups}
    group_positives = {
        group: sum(row["label"] == "1" for row in by_component[group]) for group in dev_groups
    }
    total_rows = sum(group_totals.values())
    total_positives = sum(group_positives.values())
    randomizer = random.Random(seed)
    best = None
    for _ in range(200):
        order = dev_groups.copy()
        randomizer.shuffle(order)
        selection = {"internal_test": [], "validation": [], "train": []}
        cursor = 0
        for name, fraction in (
            ("internal_test", internal_test_fraction),
            ("validation", validation_fraction),
        ):
            current = 0
            while cursor < len(order) and current < total_rows * fraction:
                group = order[cursor]
                selection[name].append(group)
                current += group_totals[group]
                cursor += 1
        selection["train"] = order[cursor:]
        if any(not selection[name] for name in selection):
            continue
        counts = {
            name: (
                sum(group_totals[g] for g in groups),
                sum(group_positives[g] for g in groups),
            ) for name, groups in selection.items()
        }
        if any(positive == 0 or positive == size for size, positive in counts.values()):
            continue
        score = sum(
            abs(counts[name][0] / total_rows - fraction)
            + abs(counts[name][1] / total_positives - fraction)
            for name, fraction in (
                ("internal_test", internal_test_fraction),
                ("validation", validation_fraction),
            )
        )
        if best is None or score < best[0]:
            best = (score, selection)
    if best is None:
        raise ValueError("Could not create three development partitions with both classes")

    partition_by_component = {
        component: name for name, groups in best[1].items() for component in groups
    }
    for component, kind in category.items():
        if kind == "external":
            partition_by_component[component] = "external_test"
    retained = []
    excluded_counts = Counter()
    for row in rows:
        group = components[row["accession"]]
        partition = partition_by_component.get(group)
        if partition is None:
            excluded_counts[category[group]] += 1
        else:
            retained.append({**row, "homology_group": group, "partition": partition})

    # One sequence contributes at most one model example, including within a partition.
    unique: dict[str, dict[str, str]] = {}
    for row in sorted(retained, key=lambda r: (
        r["window_51aa"], r["accession"], int(r["position_1based"])
    )):
        if row["window_51aa"] in unique:
            if unique[row["window_51aa"]]["partition"] != row["partition"]:
                raise AssertionError("Exact window crosses partitions")
        else:
            unique[row["window_51aa"]] = row
    locked = sorted(unique.values(), key=lambda r: (
        r["partition"], r["accession"], int(r["position_1based"])
    ))
    report = {
        "holdout_pmid": holdout_pmid,
        "protocol": "study-exclusive, full-protein homology-connected and exact-window-disjoint holdout",
        "development_split_seed": seed,
        "requested_internal_test_fraction": internal_test_fraction,
        "requested_validation_fraction": validation_fraction,
        "candidate_rows_removed_for_study_or_homology_overlap": dict(excluded_counts),
        "same_partition_duplicate_windows_removed": len(retained) - len(locked),
        "partitions_before_training_resampling": {
            name: {
                "rows": sum(row["partition"] == name for row in locked),
                "positives": sum(row["partition"] == name and row["label"] == "1" for row in locked),
                "putative_negatives": sum(row["partition"] == name and row["label"] == "0" for row in locked),
                "proteins": len({row["accession"] for row in locked if row["partition"] == name}),
                "homology_groups": len({row["homology_group"] for row in locked if row["partition"] == name}),
            } for name in ("train", "validation", "internal_test", "external_test")
        },
        "negative_definition": "Unreported lysines in positive-bearing proteins; putative, not confirmed non-Kla",
        "model_selection": "Training and validation only; internal and external tests must not guide tuning",
    }
    for name, counts in report["partitions_before_training_resampling"].items():
        if counts["positives"] == 0 or counts["putative_negatives"] == 0:
            raise ValueError(f"Partition {name} lacks one class")
    return locked, report


def training_sample(
    rows: list[dict[str, str]], ratio: int, seed: int, positive_cap: int = 0
) -> list[dict[str, str]]:
    if ratio < 1:
        raise ValueError("Training negative-to-positive ratio must be at least 1")
    if positive_cap < 0:
        raise ValueError("Training positive cap cannot be negative")
    positives = [row for row in rows if row["partition"] == "train" and row["label"] == "1"]
    negatives = [row for row in rows if row["partition"] == "train" and row["label"] == "0"]
    randomizer = random.Random(seed)
    if positive_cap and len(positives) > positive_cap:
        positives = randomizer.sample(positives, positive_cap)
    selected = randomizer.sample(negatives, min(len(negatives), ratio * len(positives)))
    return sorted(positives + selected + [row for row in rows if row["partition"] != "train"],
                  key=lambda r: (r["partition"], r["accession"], int(r["position_1based"])))


def main() -> None:
    parser = argparse.ArgumentParser(description="Lock a KlaDB human study-held-out benchmark")
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--preparation-audit", type=Path, required=True)
    parser.add_argument("--homology-hits-tsv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--holdout-pmid", default="41192556")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-negative-ratio", type=int, default=3)
    parser.add_argument("--train-positive-cap", type=int, default=0,
                        help="Pilot only: at most this many training positives; zero keeps all")
    args = parser.parse_args()
    preparation = json.loads(args.preparation_audit.read_text())
    if sha256(args.candidate_csv) != preparation["output_sha256"]["candidate_sites.csv"]:
        raise ValueError("Candidate CSV does not match preparation audit")
    with args.candidate_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    components, homology = connected_proteins(rows, args.homology_hits_tsv)
    locked, report = assign_groups(rows, components, args.holdout_pmid, seed=args.seed)
    sampled = training_sample(locked, args.train_negative_ratio, args.seed, args.train_positive_cap)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, records in (("full_manifest.csv", locked), ("model_manifest.csv", sampled)):
        path = args.output_dir / name
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=(*CANDIDATE_COLUMNS, "homology_group", "partition"),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(records)
    report.update({
        "homology": homology,
        "train_negative_ratio_cap": args.train_negative_ratio,
        "train_positive_cap": args.train_positive_cap,
        "training_resampling_seed": args.seed,
        "sampled_training_rows": sum(row["partition"] == "train" for row in sampled),
        "source_sha256": {
            "candidate_csv": sha256(args.candidate_csv),
            "preparation_audit": sha256(args.preparation_audit),
            "homology_hits_tsv": sha256(args.homology_hits_tsv),
        },
        "output_sha256": {
            name: sha256(args.output_dir / name) for name in ("full_manifest.csv", "model_manifest.csv")
        },
    })
    (args.output_dir / "split_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["partitions_before_training_resampling"], indent=2))


if __name__ == "__main__":
    main()
