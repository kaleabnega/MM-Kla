import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pcbert_kla_clean.hybridkla_models import (
    EsmEmbeddingHead, FeatureDNN, MetaDNN, SequenceLSTM,
    average_precision, encode_lstm_sequences, fit_predictor, predict_scores,
)


class HybridKlaModelTests(unittest.TestCase):
    def test_branch_shapes(self):
        with torch.no_grad():
            self.assertEqual(FeatureDNN("PSEAAC")(torch.zeros(3, 20)).shape, (3,))
            self.assertEqual(EsmEmbeddingHead(480)(torch.zeros(3, 480)).shape, (3,))
            self.assertEqual(MetaDNN()(torch.zeros(3, 8)).shape, (3,))
            sequence = "A" * 25 + "K" + "A" * 25
            tokens = torch.as_tensor(encode_lstm_sequences([sequence, sequence]))
            self.assertEqual(SequenceLSTM(embedding_dim=8, hidden_dim=8, layers=1)(tokens).shape,
                             (2,))

    def test_average_precision_ties(self):
        self.assertAlmostEqual(average_precision(np.array([1, 0]),
                                                 np.array([0.5, 0.5])), 0.5)
        self.assertAlmostEqual(average_precision(np.array([1, 0, 1]),
                                                 np.array([0.9, 0.4, 0.8])), 1.0)

    def test_fit_and_predict_feature_head(self):
        rng = np.random.default_rng(42)
        train_x = rng.random((20, 20), dtype=np.float32)
        train_y = np.array([0, 1] * 10)
        val_x = rng.random((6, 20), dtype=np.float32)
        val_y = np.array([0, 1] * 3)
        torch.manual_seed(42)
        result = fit_predictor(FeatureDNN("PSEAAC"), train_x, train_y, val_x, val_y,
                               device=torch.device("cpu"), epochs=2, batch_size=4,
                               learning_rate=1e-3, patience=0, seed=42)
        scores = predict_scores(result.model, val_x, torch.device("cpu"))
        self.assertEqual(scores.shape, (6,))
        self.assertTrue(np.logical_and(scores >= 0, scores <= 1).all())
        self.assertGreaterEqual(result.selected_epoch, 1)


if __name__ == "__main__":
    unittest.main()
