from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from pcbert_kla_clean.data import KlaSplit, describe_split
from pcbert_kla_clean.review_dataset import ReviewPartitions, prepare_review_partitions


def load_window_candidates(path: Path) -> dict[str, set[str]]:
    candidates: dict[str, set[str]] = defaultdict(set)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"window_sequence", "accession"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain {sorted(required)}")
        for row in reader:
            accession = row["accession"].strip()
            if accession:
                candidates[row["window_sequence"].strip().upper()].add(accession)
    return dict(candidates)


def load_homology_clusters(path: Path) -> dict[str, str]:
    clusters: dict[str, str] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"accession", "cluster_id"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain accession and cluster_id")
        for row in reader:
            accession, cluster = row["accession"].strip(), row["cluster_id"].strip()
            if not accession or not cluster or (accession in clusters and clusters[accession] != cluster):
                raise ValueError(f"Invalid or conflicting homology assignment for {accession!r}")
            clusters[accession] = cluster
    return clusters


def _candidate_components(
    windows: list[str],
    candidates: dict[str, set[str]],
    homology_clusters: dict[str, str] | None,
) -> tuple[dict[str, str], dict[str, set[str]]]:
    parent: dict[str, str] = {}

    def root(accession: str) -> str:
        parent.setdefault(accession, accession)
        while parent[accession] != accession:
            parent[accession] = parent[parent[accession]]
            accession = parent[accession]
        return accession

    def union(left: str, right: str) -> None:
        left_root, right_root = root(left), root(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for window in windows:
        accessions = sorted(candidates.get(window, set()))
        if not accessions:
            raise ValueError(f"No candidate accession for window {window!r}")
        for accession in accessions:
            root(accession)
        for accession in accessions[1:]:
            union(accessions[0], accession)

    if homology_clusters is not None:
        missing = set(parent) - set(homology_clusters)
        if missing:
            raise ValueError(f"Homology file omits {len(missing)} candidate accessions")
        first_by_cluster: dict[str, str] = {}
        for accession in sorted(parent):
            cluster = homology_clusters[accession]
            if cluster in first_by_cluster:
                union(accession, first_by_cluster[cluster])
            else:
                first_by_cluster[cluster] = accession

    window_groups = {window: root(min(candidates[window])) for window in windows}
    group_accessions: dict[str, set[str]] = defaultdict(set)
    for accession in parent:
        group_accessions[root(accession)].add(accession)
    return window_groups, dict(group_accessions)


def _choose_group_partitions(
    labels: np.ndarray,
    group_ids: list[str],
    seed: int,
    test_fraction: float,
    validation_fraction: float,
) -> dict[str, np.ndarray]:
    by_group: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(group_ids):
        by_group[group].append(index)
    groups = sorted(by_group)
    rng = np.random.default_rng(seed)
    target_total = np.array([test_fraction, validation_fraction]) * len(labels)
    target_positive = np.array([test_fraction, validation_fraction]) * int(labels.sum())
    best: tuple[float, dict[str, np.ndarray]] | None = None

    for _ in range(5000):
        shuffled = rng.permutation(groups)
        chosen: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
        for name, target in (("test", target_total[0]), ("validation", target_total[1])):
            remaining = [group for group in shuffled if group not in chosen["test"] and group not in chosen["validation"]]
            count = 0
            for group in remaining:
                if count >= target:
                    break
                chosen[name].append(group)
                count += len(by_group[group])
        used = set(chosen["test"]) | set(chosen["validation"])
        chosen["train"] = [group for group in groups if group not in used]
        selected = {
            name: np.array(sorted(index for group in selected_groups for index in by_group[group]), dtype=int)
            for name, selected_groups in chosen.items()
        }
        if any(len(set(labels[indices])) != 2 for indices in selected.values()):
            continue
        totals = np.array([len(selected["test"]), len(selected["validation"])])
        positives = np.array([labels[selected["test"]].sum(), labels[selected["validation"]].sum()])
        score = float(
            np.abs(totals - target_total).sum() / len(labels)
            + 2 * np.abs(positives - target_positive).sum() / max(1, labels.sum())
        )
        if best is None or score < best[0]:
            best = score, selected
    if best is None:
        raise ValueError("Could not assign all three partitions with both labels")
    return best[1]


def prepare_candidate_group_partitions(
    train: KlaSplit,
    test: KlaSplit,
    *,
    mapping_csv: Path,
    homology_csv: Path | None = None,
    seed: int = 42,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.1,
) -> ReviewPartitions:
    base = prepare_review_partitions(
        train, test, seed=seed, test_fraction=test_fraction, validation_fraction=validation_fraction
    )
    candidates = load_window_candidates(mapping_csv)
    source_windows = set(train.sequences) | set(test.sequences)
    if set(candidates) != source_windows:
        raise ValueError(
            f"Mapping coverage mismatch: {len(source_windows - set(candidates))} missing and "
            f"{len(set(candidates) - source_windows)} extra windows"
        )
    clusters = load_homology_clusters(homology_csv) if homology_csv is not None else None
    eligible = [*base.train.records, *base.validation.records, *base.test.records]
    features = np.concatenate((base.train.features, base.validation.features, base.test.features))
    windows = [record.sequence for record in eligible]
    window_groups, group_accessions = _candidate_components(sorted(source_windows), candidates, clusters)
    selected = _choose_group_partitions(
        np.asarray([record.label for record in eligible]),
        [window_groups[window] for window in windows],
        seed,
        test_fraction,
        validation_fraction,
    )

    def make_split(indices: np.ndarray) -> KlaSplit:
        return KlaSplit(
            records=[eligible[int(index)] for index in indices],
            feature_names=train.feature_names,
            features=features[indices].copy(),
        )

    parts = {name: make_split(indices) for name, indices in selected.items()}
    accession_sets = {
        name: set().union(*(candidates[record.sequence] for record in part.records))
        for name, part in parts.items()
    }
    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    if any(accession_sets[left] & accession_sets[right] for left, right in pairs):
        raise AssertionError("A candidate protein appears in multiple partitions")
    if clusters is not None:
        cluster_sets = {name: {clusters[accession] for accession in accessions} for name, accessions in accession_sets.items()}
        if any(cluster_sets[left] & cluster_sets[right] for left, right in pairs):
            raise AssertionError("A supplied homology cluster appears in multiple partitions")

    assignment_by_window = {
        eligible[int(index)].sequence: name for name, indices in selected.items() for index in indices
    }
    inventory = []
    for row in base.inventory:
        row = dict(row)
        window = str(row["sequence"])
        row["candidate_accessions"] = ";".join(sorted(candidates[window]))
        row["candidate_group"] = window_groups.get(window, "excluded")
        if row["assignment"] not in {"excluded_conflicting_labels", "excluded_inconsistent_features"}:
            row["assignment"] = assignment_by_window[window]
        inventory.append(row)

    report = dict(base.report)
    report.update({
        "partitions": {name: describe_split(part) for name, part in parts.items()},
        "candidate_protein_groups": len(group_accessions),
        "candidate_proteins": len(set().union(*group_accessions.values())),
        "candidate_protein_overlap": {
            f"{left}_{right}": len(accession_sets[left] & accession_sets[right])
            for left, right in pairs
        },
        "mapping_csv_sha256": hashlib.sha256(mapping_csv.read_bytes()).hexdigest(),
        "homology_csv_sha256": hashlib.sha256(homology_csv.read_bytes()).hexdigest() if homology_csv else None,
        "protocol": "candidate-protein-group-disjoint diagnostic" + (
            " with supplied homology clusters" if homology_csv else "; homology not screened"
        ),
        "provenance_limit": (
            "Candidate accessions are exact proteome matches, not verified original parent "
            "identities. Negative labels are putative. This is not a verified protein-disjoint "
            "or independent-study test."
        ),
    })
    return ReviewPartitions(**parts, report=report, inventory=inventory)
