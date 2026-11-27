import unittest

import numpy as np
import torch

from pcbert_kla_clean.attentionkla_development import (
    average_fold_logits, positive_probabilities, validate_tokenized_windows,
)


class _Tokenizer:
    def convert_tokens_to_ids(self, token):
        assert token == "K"
        return 42


class AttentionKlaDevelopmentTests(unittest.TestCase):
    def test_esm_window_validation(self):
        ids = torch.ones((2, 53), dtype=torch.long)
        ids[:, 26] = 42
        mask = torch.ones_like(ids)
        windows = ["A" * 25 + "K" + "A" * 25] * 2
        validate_tokenized_windows(_Tokenizer(), windows, ids, mask)
        ids[1, 26] = 1
        with self.assertRaisesRegex(ValueError, "central lysine"):
            validate_tokenized_windows(_Tokenizer(), windows, ids, mask)


    def test_ensemble_uses_mean_logits_then_softmax(self):
        first = np.array([[0.0, 2.0], [2.0, 0.0]])
        second = np.array([[0.0, 0.0], [0.0, 2.0]])
        combined = average_fold_logits([first, second])
        np.testing.assert_allclose(combined, [[0.0, 1.0], [1.0, 1.0]])
        np.testing.assert_allclose(positive_probabilities(combined),
                                   [1 / (1 + np.exp(-1)), 0.5])


    def test_nonfinite_fold_logits_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            average_fold_logits([np.array([[np.nan, 0.0]])])


if __name__ == "__main__":
    unittest.main()
