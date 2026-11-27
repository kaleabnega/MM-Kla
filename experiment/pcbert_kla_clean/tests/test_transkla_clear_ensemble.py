from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_transkla_clear_ensemble import fold_rows  # noqa: E402


class TransKlaEnsembleTests(unittest.TestCase):
    def test_inner_selection_excludes_outer_holdout_and_validation(self):
        train = [{"id": index} for index in range(6)]
        validation = [{"id": "V"}]
        inner, stop, outer, score = fold_rows(
            train, validation, [0, 1, 2, 3], [4, 5], [0, 2], [1, 3],
        )
        self.assertEqual([row["id"] for row in inner], [0, 2])
        self.assertEqual([row["id"] for row in stop], [1, 3])
        self.assertEqual([row["id"] for row in outer], [0, 1, 2, 3])
        self.assertEqual([row["id"] for row in score], [4, 5, "V"])

    def test_bad_fold_coverage_is_rejected(self):
        rows = [{"id": index} for index in range(6)]
        with self.assertRaises(ValueError):
            fold_rows(rows, [], np.array([0, 1, 2, 3]), np.array([4, 5]),
                      np.array([0, 1]), np.array([2]))


if __name__ == "__main__":
    unittest.main()
