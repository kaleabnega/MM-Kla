from __future__ import annotations

import unittest

import torch

from pcbert_kla_clean.deepkla import DeepKlaAligned, encode_windows


class DeepKlaTests(unittest.TestCase):
    def test_encoding_matches_released_script(self) -> None:
        window = "A" * 25 + "K" + "Y" * 25
        encoded = encode_windows([window])
        self.assertEqual(tuple(encoded.shape), (1, 200))
        self.assertTrue(torch.all(encoded[0, :148] == 0))
        self.assertEqual(encoded[0, 148].item(), 1)
        self.assertEqual(encoded[0, 149].item(), 4)
        self.assertEqual(encoded[0, 174].item(), 12)
        self.assertEqual(encoded[0, 175].item(), 2)

    def test_rejects_invalid_center_or_residues(self) -> None:
        with self.assertRaisesRegex(ValueError, "K-centered"):
            encode_windows(["A" * 51])
        with self.assertRaisesRegex(ValueError, "canonical"):
            encode_windows(["X" + "A" * 24 + "K" + "A" * 25])

    def test_forward_and_backward(self) -> None:
        model = DeepKlaAligned()
        scores = model(encode_windows(["A" * 25 + "K" + "A" * 25,
                                       "C" * 25 + "K" + "C" * 25]))
        self.assertEqual(tuple(scores.shape), (2,))
        self.assertTrue(torch.all(torch.isfinite(scores)))
        self.assertTrue(torch.all((scores >= 0) & (scores <= 1)))
        scores.mean().backward()
        self.assertIsNotNone(model.embedding.weight.grad)


if __name__ == "__main__":
    unittest.main()
