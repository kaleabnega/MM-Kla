import unittest
import csv
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from pcbert_kla_clean.stacked_development import (
    PositionalSequenceBranch, grouped_splits, protbert_args,
    read_positional_reference, sequence_tokens,
)
from pcbert_kla_clean.source_study import sha256


class StackedDevelopmentTests(unittest.TestCase):
    def test_grouped_folds_keep_proteins_together(self):
        groups = np.repeat(np.arange(20), 4).astype(str)
        labels = np.tile(np.array([0, 0, 1, 1]), 20)
        splits = grouped_splits(labels, groups, seed=42)
        self.assertEqual(len(splits), 5)
        held = np.concatenate([test for _, test in splits])
        np.testing.assert_array_equal(np.sort(held), np.arange(len(labels)))
        for train, test in splits:
            self.assertFalse(set(groups[train]) & set(groups[test]))

    def test_positional_branch_scores_one_central_lysine_per_window(self):
        windows = ["A" * 25 + "K" + "G" * 25, "G" * 25 + "K" + "A" * 25]
        tokens = sequence_tokens(windows)
        self.assertEqual(tuple(tokens.shape), (2, 51))
        model = PositionalSequenceBranch()
        logits = model(tokens)
        self.assertEqual(tuple(logits.shape), (2,))
        self.assertTrue(torch.isfinite(logits).all())
        with self.assertRaises(ValueError):
            sequence_tokens(["A" * 51])

    def test_protbert_control_keeps_original_learning_rates_and_four_layers(self):
        args = protbert_args(30)
        self.assertEqual(args.encoder_layers, 4)
        self.assertEqual(args.learning_rate, 2e-5)
        self.assertEqual(args.head_learning_rate, 1e-4)
        self.assertEqual(args.scheduler, "linear")

    def test_saved_positional_scores_require_matching_rows_and_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock = root / "split_audit.json"
            lock.write_text("locked")
            reference = root / "reference"
            reference.mkdir()
            split = SimpleNamespace(
                features=np.zeros((3, 1627)), labels=np.array([0, 1, 1]),
                records=[SimpleNamespace(name="P1", position=1, label=0),
                         SimpleNamespace(name="P2", position=2, label=1),
                         SimpleNamespace(name="P3", position=3, label=1)],
            )
            groups = np.array(["g1", "g2"])
            report = {
                "seed": 42, "lock_sha256": sha256(lock), "training_rows": 2,
                "validation_rows": 1, "feature_dim": 1627,
                "internal_test_scored": False,
                "folds": [{"positional_selected_epoch": 2,
                           "positional_inner_auprc": 0.5} for _ in range(5)],
            }
            (reference / "report.json").write_text(json.dumps(report))
            np.savez_compressed(reference / "oof_branch_scores.npz",
                                scores=np.array([[0.2, 0.3], [0.7, 0.8]]),
                                labels=np.array([0, 1]), groups=groups)
            with (reference / "validation_predictions.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["accession", "position_1based", "label", "positional_sequence"])
                writer.writerow(["P3", 3, 1, 0.75])
            oof, validation, _ = read_positional_reference(
                reference, lock, split, 2, np.array([2]), groups, 42,
            )
            np.testing.assert_allclose(oof, [0.3, 0.8])
            np.testing.assert_allclose(validation, [0.75])
            split.records[2].name = "WRONG"
            with self.assertRaises(ValueError):
                read_positional_reference(reference, lock, split, 2,
                                          np.array([2]), groups, 42)


if __name__ == "__main__":
    unittest.main()
