import unittest
from types import SimpleNamespace

import numpy as np
import torch

from pcbert_kla_clean.positional_order_context_development import (
    OrderContextPositionalBranch, batch_targets, make_protein_batch, protein_tokens,
)


class PositionalOrderContextTests(unittest.TestCase):
    def test_singleton_minibatch_preserves_label_dimension(self):
        labels = np.array([0, 1, 0])
        targets = batch_targets(labels, torch.tensor([1]), torch.device("cpu"))
        self.assertEqual(targets.shape, (1,))
        self.assertEqual(targets.tolist(), [1.0])

    def test_protein_batch_preserves_all_residues_and_site_positions(self):
        records = [SimpleNamespace(name="P1", position=3),
                   SimpleNamespace(name="P2", position=2)]
        bank = {"P1": protein_tokens("AAKRA"), "P2": protein_tokens("AKR")}
        batch, lengths, sites = make_protein_batch(
            torch.tensor([0, 1]), records, bank, torch.device("cpu"),
        )
        self.assertEqual(batch.shape, (2, 5))
        self.assertEqual(lengths.tolist(), [5, 3])
        self.assertEqual(sites.tolist(), [2, 1])
        self.assertEqual(batch[1, 3:].tolist(), [0, 0])

    def test_order_changes_context_even_with_identical_composition(self):
        torch.manual_seed(5)
        model = OrderContextPositionalBranch().eval()
        # Each eight-residue segment contains the same counts but different motifs.
        a = protein_tokens("AAAARRRR" * 8)
        b = protein_tokens("ARARARAR" * 8)
        with torch.no_grad():
            context = model.encode_protein(torch.stack((a, b)),
                                           torch.tensor([64, 64]),
                                           torch.tensor([31, 31]))
        self.assertGreater((context[0] - context[1]).abs().max().item(), 1e-5)

    def test_padding_cannot_change_eight_context_tokens(self):
        torch.manual_seed(7)
        model = OrderContextPositionalBranch().eval()
        sequence = protein_tokens("AAKRA" * 11)
        short = model.encode_protein(sequence[None, :], torch.tensor([55]),
                                     torch.tensor([27]))
        padded = torch.nn.functional.pad(sequence, (0, 20))[None, :]
        long = model.encode_protein(padded, torch.tensor([55]),
                                    torch.tensor([27]))
        torch.testing.assert_close(short, long, atol=1e-6, rtol=1e-6)

    def test_forward_and_backward_are_finite(self):
        model = OrderContextPositionalBranch()
        local = torch.zeros((2, 51), dtype=torch.long)
        local[:, 25] = 8
        sequence = protein_tokens("A" * 25 + "K" + "A" * 38)
        scores = model(local, sequence.repeat(2, 1), torch.tensor([64, 64]),
                       torch.tensor([25, 25]))
        self.assertEqual(scores.shape, (2,))
        scores.sum().backward()
        self.assertTrue(torch.isfinite(scores).all())
        self.assertIsNotNone(model.protein_depthwise.weight.grad)
        self.assertTrue(torch.isfinite(model.protein_depthwise.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
