import unittest

import torch

from pcbert_kla_clean.positional_multiscale_local import (
    MultiscaleLocalOrderContextPositionalBranch,
)
from pcbert_kla_clean.positional_order_context_development import (
    OrderContextPositionalBranch, protein_tokens,
)


class MultiscaleLocalTests(unittest.TestCase):
    def test_local_encoder_preserves_window_shape(self):
        model = MultiscaleLocalOrderContextPositionalBranch()
        tokens = torch.randint(0, 20, (3, 51))
        self.assertEqual(model.encode_local(tokens).shape, (3, 51, 64))
        self.assertTrue(torch.isfinite(model.encode_local(tokens)).all())

    def test_reference_local_encoder_is_unchanged(self):
        model = OrderContextPositionalBranch()
        tokens = torch.randint(0, 20, (2, 51))
        positions = torch.arange(51)
        expected = model.residue(tokens) + model.position(positions)[None, :, :]
        torch.testing.assert_close(model.encode_local(tokens), expected)

    def test_gradients_reach_all_local_scales_and_context(self):
        model = MultiscaleLocalOrderContextPositionalBranch()
        local = torch.ones((2, 51), dtype=torch.long)
        protein = protein_tokens("A" * 25 + "K" + "R" * 38)
        logits = model(local, protein.repeat(2, 1), torch.tensor([64, 64]),
                       torch.tensor([25, 25]))
        self.assertEqual(logits.shape, (2,))
        logits.sum().backward()
        self.assertTrue(torch.isfinite(logits).all())
        for layer in model.local_scales:
            self.assertTrue(torch.isfinite(layer.weight.grad).all())
            self.assertGreater(layer.weight.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(model.protein_depthwise.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
