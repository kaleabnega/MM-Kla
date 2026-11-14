import unittest

import numpy as np
import torch

from pcbert_kla_clean.stacked_development import (
    PositionalSequenceBranch, grouped_splits, protbert_args, sequence_tokens,
)


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


if __name__ == "__main__":
    unittest.main()
