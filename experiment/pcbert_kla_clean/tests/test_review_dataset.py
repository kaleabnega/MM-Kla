from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from pcbert_kla_clean.data import KlaRecord, KlaSplit, load_split
from pcbert_kla_clean.review_dataset import (
    combine_development_partitions,
    prepare_review_partitions,
)


class ReviewDatasetTests(unittest.TestCase):
    def make_split(self, records: list[tuple[str, int]], feature_override=None) -> KlaSplit:
        parsed = [
            KlaRecord(name=f"Protein {i}", position=1, label=label, sequence=sequence)
            for i, (sequence, label) in enumerate(records)
        ]
        features = np.array(
            [[float(len(sequence)), float(sequence.count("A"))] for sequence, _ in records],
            dtype=np.float32,
        )
        if feature_override is not None:
            features[feature_override[0]] = feature_override[1]
        return KlaSplit(parsed, ["length", "A_count"], features)

    def test_disjoint_deterministic_and_conflicts_quarantined(self) -> None:
        negatives = [(f"K{'A' * i}C", 0) for i in range(3, 23)]
        positives = [(f"K{'G' * i}T", 1) for i in range(3, 23)]
        train = self.make_split(negatives + positives + [("KCCC", 1), ("KDDD", 1)])
        test = self.make_split(
            [("KCCC", 0), (negatives[0][0], 0), ("KDDD", 1)],
            feature_override=(1, [999.0, 999.0]),
        )
        first = prepare_review_partitions(train, test, seed=7)
        again = prepare_review_partitions(train, test, seed=7)

        self.assertEqual(first.report["excluded_conflicting_label_windows"], 1)
        self.assertEqual(first.report["excluded_inconsistent_feature_windows"], 1)
        self.assertEqual(first.report["original_overlap"]["overlap_test_rows"], 3)
        self.assertEqual(first.test.sequences, again.test.sequences)
        self.assertEqual(first.validation.sequences, again.validation.sequences)
        sets = [set(part.sequences) for part in (first.train, first.validation, first.test)]
        self.assertFalse(sets[0] & sets[1])
        self.assertFalse(sets[0] & sets[2])
        self.assertFalse(sets[1] & sets[2])
        self.assertNotIn("KCCC", set.union(*sets))
        self.assertNotIn(negatives[0][0], set.union(*sets))
        for part in (first.train, first.validation, first.test):
            self.assertEqual(set(part.labels), {0, 1})
        combined, validation_indices = combine_development_partitions(first)
        self.assertEqual(len(combined.records), len(first.train.records) + len(first.validation.records))
        self.assertEqual(
            [combined.records[index].sequence for index in validation_indices],
            first.validation.sequences,
        )

    def test_bundled_benchmark_audit(self) -> None:
        root = PROJECT_DIR.parents[1] / "baselines" / "PCBert-Kla-original" / "data"
        train = load_split(root / "train.csv", root / "feature_train.csv")
        test = load_split(root / "test.csv", root / "feature_test.csv")
        parts = prepare_review_partitions(train, test)
        report = parts.report

        self.assertEqual(report["original_overlap"]["overlap_test_rows"], 255)
        self.assertEqual(report["original_test_overlap_rows_by_label"], {"0": 85, "1": 170})
        self.assertEqual(report["original_positive_train_unique_windows"], 172)
        self.assertEqual(report["excluded_conflicting_label_windows"], 5)
        self.assertEqual(report["excluded_conflicting_label_rows"], 50)
        self.assertEqual(report["conflicting_label_windows_across_original_splits"], 4)
        self.assertEqual(report["conflicting_label_windows_within_original_test"], 5)
        self.assertEqual(report["excluded_inconsistent_feature_windows"], 0)
        self.assertEqual(sum(len(getattr(parts, name).records) for name in ("train", "validation", "test")), 1932)


if __name__ == "__main__":
    unittest.main()
