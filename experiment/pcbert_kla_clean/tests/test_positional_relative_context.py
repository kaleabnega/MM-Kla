import unittest

import torch

from pcbert_kla_clean.positional_order_context_development import protein_tokens
from pcbert_kla_clean.positional_relative_context import (
    RelativeContextPositionalBranch, relative_bin_ids,
)


class RelativeContextTests(unittest.TestCase):
    def test_signed_distance_boundaries(self):
        protein = torch.ones((1, 501), dtype=torch.long)
        bins, valid = relative_bin_ids(protein, torch.tensor([501]), torch.tensor([250]))
        expected = {0: 0, 50: 1, 200: 2, 225: 3, 249: 3, 250: 4,
                    251: 5, 275: 5, 276: 6, 300: 6, 450: 7, 451: 8}
        for position, bin_id in expected.items():
            self.assertEqual(int(bins[0, position]), bin_id)
        self.assertTrue(valid.all())

    def test_empty_distance_bands_are_masked(self):
        model = RelativeContextPositionalBranch().eval()
        protein = protein_tokens("A" * 25 + "K" + "A" * 25)[None, :]
        mask = model.global_padding_mask(protein, torch.tensor([51]), torch.tensor([25]))
        self.assertEqual(mask.shape, (1, 9))
        self.assertEqual(mask[0].tolist(), [True, True, True, False, False,
                                             False, True, True, True])

    def test_padding_does_not_change_context(self):
        torch.manual_seed(7)
        model = RelativeContextPositionalBranch().eval()
        protein = protein_tokens("A" * 25 + "K" + "R" * 38)
        short = model.encode_protein(protein[None, :], torch.tensor([64]), torch.tensor([25]))
        padded = torch.nn.functional.pad(protein, (0, 20))[None, :]
        long = model.encode_protein(padded, torch.tensor([64]), torch.tensor([25]))
        torch.testing.assert_close(short, long, atol=1e-6, rtol=1e-6)

    def test_forward_backward_with_different_lengths(self):
        model = RelativeContextPositionalBranch()
        local = torch.ones((2, 51), dtype=torch.long)
        a = protein_tokens("A" * 25 + "K" + "R" * 38)
        b = protein_tokens("A" * 10 + "K" + "R" * 40)
        proteins = torch.nn.utils.rnn.pad_sequence((a, b), batch_first=True)
        scores = model(local, proteins, torch.tensor([64, 51]), torch.tensor([25, 10]))
        self.assertEqual(scores.shape, (2,))
        self.assertTrue(torch.isfinite(scores).all())
        scores.sum().backward()
        self.assertTrue(torch.isfinite(model.protein_depthwise.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
