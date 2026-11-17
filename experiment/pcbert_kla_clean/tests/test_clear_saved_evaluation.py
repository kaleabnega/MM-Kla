from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_clear_saved import checked_metrics  # noqa: E402
from pcbert_kla_clean.stacked_development import run  # noqa: E402


class ClearSavedEvaluationTests(unittest.TestCase):
    def test_saved_validation_comparison_rejects_mismatch(self) -> None:
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        expected = {"AUPRC": 1.0, "AUC": 1.0, "MCC": 1.0}
        self.assertAlmostEqual(checked_metrics(labels, scores, expected, "test")["AUPRC"], 1.0)
        with self.assertRaisesRegex(ValueError, "did not reproduce"):
            checked_metrics(labels, scores, {**expected, "AUPRC": 0.5}, "test")
        with self.assertRaisesRegex(ValueError, "Invalid"):
            checked_metrics(labels, np.array([0.1, np.nan, 0.8, 0.9]), expected, "test")

    def test_stacked_test_requires_prior_report_before_training(self) -> None:
        with self.assertRaisesRegex(ValueError, "prior validation report"):
            run(Path("missing"), Path("missing"), torch.device("cpu"), seed=42,
                score_internal_test=True)


if __name__ == "__main__":
    unittest.main()
