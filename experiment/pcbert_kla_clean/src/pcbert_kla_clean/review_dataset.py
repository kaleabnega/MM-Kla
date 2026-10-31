from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pcbert_kla_clean.data import (
    KlaRecord,
    KlaSplit,
    describe_split,
    train_test_overlap_report,
)


@dataclass(frozen=True)
class ReviewPartitions:
    train: KlaSplit
    validation: KlaSplit
    test: KlaSplit
    report: dict[str, object]
    inventory: list[dict[str, object]]


def _window_groups(train: KlaSplit, test: KlaSplit):
    if train.feature_names != test.feature_names:
        raise ValueError("Train and test feature columns differ")
    groups = defaultdict(list)
    for source, split in (("train", train), ("test", test)):
        for index, record in enumerate(split.records):
            groups[record.sequence].append((source, index, record, split.features[index]))
    return groups


def _feature_consistent(rows: list[tuple]) -> bool:
    reference = rows[0][3]
    return all(
        np.allclose(reference, row[3], rtol=1e-5, atol=1e-5, equal_nan=False)
        for row in rows[1:]
    )


def _stratified_indices(
    labels: np.ndarray,
    seed: int,
    test_fraction: float,
    validation_fraction: float,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    selected: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    for label in (0, 1):
        class_indices = rng.permutation(np.flatnonzero(labels == label))
        test_count = max(1, round(len(class_indices) * test_fraction))
        validation_count = max(1, round(len(class_indices) * validation_fraction))
        if test_count + validation_count >= len(class_indices):
            raise ValueError("Each class needs enough windows for all three partitions")
        selected["test"].extend(class_indices[:test_count].tolist())
        selected["validation"].extend(
            class_indices[test_count : test_count + validation_count].tolist()
        )
        selected["train"].extend(class_indices[test_count + validation_count :].tolist())
    return {name: np.array(sorted(indices), dtype=int) for name, indices in selected.items()}


def prepare_review_partitions(
    train: KlaSplit,
    test: KlaSplit,
    *,
    seed: int = 42,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.1,
) -> ReviewPartitions:
    if not (0 < test_fraction < 1 and 0 < validation_fraction < 1 - test_fraction):
        raise ValueError("Test and validation fractions must be positive and sum to less than 1")

    groups = _window_groups(train, test)
    inventory: list[dict[str, object]] = []
    eligible: list[tuple[str, tuple]] = []
    conflicting_rows = 0
    inconsistent_feature_rows = 0
    conflicting_within_train = 0
    conflicting_within_test = 0
    conflicting_across_splits = 0

    for sequence, rows in sorted(groups.items()):
        labels = sorted({row[2].label for row in rows})
        train_labels = {row[2].label for row in rows if row[0] == "train"}
        test_labels = {row[2].label for row in rows if row[0] == "test"}
        conflicting_within_train += len(train_labels) > 1
        conflicting_within_test += len(test_labels) > 1
        conflicting_across_splits += bool(train_labels and test_labels and len(labels) > 1)
        consistent = _feature_consistent(rows)
        if len(labels) > 1:
            assignment = "excluded_conflicting_labels"
            conflicting_rows += len(rows)
        elif not consistent:
            assignment = "excluded_inconsistent_features"
            inconsistent_feature_rows += len(rows)
        else:
            assignment = "pending"
            eligible.append((sequence, rows[0]))

        inventory.append(
            {
                "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
                "sequence": sequence,
                "label_values": ",".join(str(value) for value in labels),
                "original_train_rows": sum(row[0] == "train" for row in rows),
                "original_test_rows": sum(row[0] == "test" for row in rows),
                "train_label_0_rows": sum(
                    row[0] == "train" and row[2].label == 0 for row in rows
                ),
                "train_label_1_rows": sum(
                    row[0] == "train" and row[2].label == 1 for row in rows
                ),
                "test_label_0_rows": sum(
                    row[0] == "test" and row[2].label == 0 for row in rows
                ),
                "test_label_1_rows": sum(
                    row[0] == "test" and row[2].label == 1 for row in rows
                ),
                "feature_consistent": consistent,
                "assignment": assignment,
            }
        )

    labels = np.array([row[1][2].label for row in eligible], dtype=int)
    if len(set(labels)) != 2 or min(Counter(labels).values()) < 3:
        raise ValueError("Both classes need at least three eligible unique windows")

    selected = _stratified_indices(labels, seed, test_fraction, validation_fraction)
    assignments = {
        int(index): name
        for name, selected in (
            ("train", selected["train"]),
            ("validation", selected["validation"]),
            ("test", selected["test"]),
        )
        for index in selected
    }
    assignment_by_sequence = {
        sequence: assignments[index] for index, (sequence, _) in enumerate(eligible)
    }
    for row in inventory:
        if row["assignment"] == "pending":
            row["assignment"] = assignment_by_sequence[row["sequence"]]

    def make_split(selected: np.ndarray) -> KlaSplit:
        selected_rows = [eligible[int(index)] for index in selected]
        records = [
            KlaRecord(
                name="window_" + hashlib.sha256(sequence.encode("ascii")).hexdigest()[:16],
                position=source_row[2].position,
                label=source_row[2].label,
                sequence=sequence,
            )
            for sequence, source_row in selected_rows
        ]
        features = np.stack([source_row[3] for _, source_row in selected_rows]).astype(
            np.float32
        )
        return KlaSplit(records=records, feature_names=train.feature_names, features=features)

    parts = {
        name: make_split(indices) for name, indices in selected.items()
    }
    sets = {name: set(part.sequences) for name, part in parts.items()}
    if any(sets[a] & sets[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise AssertionError("A sequence appears in more than one review partition")

    train_sequences = set(train.sequences)
    test_overlap_by_label = {
        str(label): sum(
            record.label == label and record.sequence in train_sequences
            for record in test.records
        )
        for label in (0, 1)
    }
    positive_train_unique = len(
        {record.sequence for record in train.records if record.label == 1}
    )
    report: dict[str, object] = {
        "original_train": describe_split(train),
        "original_test": describe_split(test),
        "original_overlap": train_test_overlap_report(train, test),
        "original_test_overlap_rows_by_label": test_overlap_by_label,
        "original_positive_train_unique_windows": positive_train_unique,
        "combined_unique_windows": len(groups),
        "excluded_conflicting_label_windows": sum(
            row["assignment"] == "excluded_conflicting_labels" for row in inventory
        ),
        "excluded_conflicting_label_rows": conflicting_rows,
        "conflicting_label_windows_within_original_train": conflicting_within_train,
        "conflicting_label_windows_within_original_test": conflicting_within_test,
        "conflicting_label_windows_across_original_splits": conflicting_across_splits,
        "excluded_inconsistent_feature_windows": sum(
            row["assignment"] == "excluded_inconsistent_features" for row in inventory
        ),
        "excluded_inconsistent_feature_rows": inconsistent_feature_rows,
        "partition_seed": seed,
        "test_fraction": test_fraction,
        "validation_fraction": validation_fraction,
        "partitions": {name: describe_split(part) for name, part in parts.items()},
        "protocol": "exact-window-disjoint diagnostic, not protein- or homology-disjoint",
        "provenance_limit": (
            "The source files use generic row names and do not provide verified "
            "parent-protein identities or homology clusters. This diagnostic does "
            "not establish performance on unseen proteins."
        ),
    }
    return ReviewPartitions(**parts, report=report, inventory=inventory)


def save_review_partitions(partitions: ReviewPartitions, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_audit.json").write_text(
        json.dumps(partitions.report, indent=2, sort_keys=True) + "\n"
    )
    with (output_dir / "window_inventory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(partitions.inventory[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(partitions.inventory)


def combine_development_partitions(partitions: ReviewPartitions) -> tuple[KlaSplit, np.ndarray]:
    train, validation = partitions.train, partitions.validation
    combined = KlaSplit(
        records=train.records + validation.records,
        feature_names=train.feature_names,
        features=np.concatenate((train.features, validation.features), axis=0),
    )
    validation_indices = np.arange(len(train.records), len(combined.records), dtype=int)
    return combined, validation_indices
