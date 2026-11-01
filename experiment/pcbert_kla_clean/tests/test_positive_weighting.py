import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from pcbert_kla_clean.data import KlaRecord, KlaSplit
from pcbert_kla_clean.experiments import train_and_predict_independent
from pcbert_kla_clean.training import make_training_criterion


class PositiveWeightingTests(unittest.TestCase):
    def test_train_ratio_uses_only_training_indices(self):
        labels = np.array([1, 0, 1, 0, 0, 0, 1])
        criterion, counts = make_training_criterion(
            labels, np.array([0, 1, 2, 3, 4]), "train_ratio"
        )
        self.assertEqual(counts["training_positive_rows"], 2)
        self.assertEqual(counts["training_negative_rows"], 3)
        self.assertEqual(counts["training_positive_weight"], 1.5)
        probabilities = torch.tensor([0.8, 0.2], requires_grad=True)
        targets = torch.tensor([1.0, 0.0])
        actual = criterion(probabilities, targets)
        expected = F.binary_cross_entropy(
            probabilities, targets, weight=torch.tensor([1.5, 1.0])
        )
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(torch.isfinite(probabilities.grad).all())

    def test_default_loss_remains_unweighted(self):
        criterion, counts = make_training_criterion(
            np.array([0, 1]), np.array([0, 1]), "none"
        )
        self.assertIsInstance(criterion, torch.nn.BCELoss)
        self.assertEqual(counts["training_positive_weight"], 1.0)

    def test_development_run_never_predicts_test_partitions(self):
        labels = [0, 0, 1, 0, 1]
        split = KlaSplit(
            records=[KlaRecord(str(i), 26, label, "A" * 25 + "K" + "C" * 25)
                     for i, label in enumerate(labels)],
            feature_names=["feature"],
            features=np.arange(5, dtype=np.float32).reshape(-1, 1),
        )
        args = Namespace(
            device="cpu", sequence_format="raw", batch_size=2, max_length=64,
            epochs=1, patience=0, decision_threshold=0.5, monitor_metric="AUPRC",
            calibrate_threshold="none", save_predictions=True, save_models=False,
            architecture="baseline", positive_weighting="train_ratio",
        )
        with tempfile.TemporaryDirectory() as temporary, \
             patch("pcbert_kla_clean.experiments.make_loader", side_effect=lambda records, indices, *rest: (records, indices)) as make_loader, \
             patch("pcbert_kla_clean.experiments.build_model", return_value=torch.nn.Linear(1, 1)), \
             patch("pcbert_kla_clean.experiments.build_optimizer", return_value=object()), \
             patch("pcbert_kla_clean.experiments.build_scheduler", return_value=None), \
             patch("pcbert_kla_clean.experiments.train_one_epoch", return_value=0.1) as train_epoch, \
             patch("pcbert_kla_clean.experiments.evaluate", return_value=(0.2, {"ACC": 1.0, "AUPRC": 1.0, "MCC": 1.0, "AUC": 1.0})) as evaluate, \
             patch("pcbert_kla_clean.experiments.predict", return_value=(0.2, np.array([0, 1]), np.array([0.1, 0.9]))) as predict:
            output = Path(temporary)
            result = train_and_predict_independent(
                args, split, None, seed=42, tokenizer=object(),
                device=torch.device("cpu"), output_dir=output,
                fixed_validation_indices=np.array([3, 4]),
            )
            self.assertEqual(make_loader.call_count, 2)
            self.assertEqual(predict.call_count, 1)
            self.assertIsInstance(train_epoch.call_args.args[3], torch.nn.Module)
            self.assertIsInstance(evaluate.call_args.args[2], torch.nn.BCELoss)
            self.assertEqual(result["validation_metrics"]["AUPRC"], 1.0)
            self.assertTrue((output / "validation_metrics.json").exists())
            self.assertTrue((output / "validation_predictions.csv").exists())
            self.assertFalse((output / "test_metrics.json").exists())
            self.assertFalse((output / "rice_external_metrics.json").exists())
            config = json.loads((output / "run_config.json").read_text())
            self.assertEqual(config["training_positive_weight"], 2.0)
            self.assertEqual(config["training_positive_rows"], 1)
            self.assertEqual(config["training_negative_rows"], 2)
            self.assertEqual(config["test_rows"], 0)


if __name__ == "__main__":
    unittest.main()
