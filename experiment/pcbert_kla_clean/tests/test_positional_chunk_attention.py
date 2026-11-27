import unittest

import torch

from pcbert_kla_clean.positional_chunk_attention import (
    ChunkAttentiveOrderContextPositionalBranch, attentive_chunk_pool,
)
from pcbert_kla_clean.positional_order_context_development import protein_tokens


class ChunkAttentionTests(unittest.TestCase):
    def test_target_query_changes_residues_selected_within_same_chunk(self):
        states = torch.zeros((1, 16, 64))
        states[0, 0, 0] = 2
        states[0, 1, 0] = -2
        positive = torch.zeros((1, 64))
        positive[0, 0] = 8
        negative = -positive
        first = attentive_chunk_pool(states, positive, torch.tensor([16]))
        second = attentive_chunk_pool(states, negative, torch.tensor([16]))
        self.assertGreater(first[0, 0, 0].item(), 1)
        self.assertLess(second[0, 0, 0].item(), -1)
        self.assertEqual(first.shape, (1, 8, 64))

    def test_padding_and_short_proteins_do_not_create_nan(self):
        torch.manual_seed(7)
        states = torch.randn((1, 5, 64))
        query = torch.randn((1, 64))
        short = attentive_chunk_pool(states, query, torch.tensor([5]))
        padded = torch.cat((states, torch.randn((1, 12, 64))), dim=1)
        long = attentive_chunk_pool(padded, query, torch.tensor([5]))
        torch.testing.assert_close(short, long, atol=1e-6, rtol=1e-6)
        self.assertTrue(torch.isfinite(long).all())

    def test_end_to_end_gradients_reach_query_and_protein_encoder(self):
        model = ChunkAttentiveOrderContextPositionalBranch()
        local = torch.ones((2, 51), dtype=torch.long)
        sequence = protein_tokens("A" * 25 + "K" + "R" * 38)
        logits = model(local, sequence.repeat(2, 1), torch.tensor([64, 64]),
                       torch.tensor([25, 25]))
        self.assertEqual(logits.shape, (2,))
        logits.sum().backward()
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(model.global_query.weight.grad).all())
        self.assertTrue(torch.isfinite(model.protein_depthwise.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
