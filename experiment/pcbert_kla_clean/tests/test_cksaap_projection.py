import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from pcbert_kla_clean.model import TokenGatedCKSAAPProjectionPCBertKla


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=12)
        self.embeddings = nn.Embedding(30, 12)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([nn.Linear(12, 12) for _ in range(4)])

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        hidden = self.embeddings(input_ids)
        for layer in self.encoder.layer:
            hidden = torch.tanh(layer(hidden))
        return SimpleNamespace(last_hidden_state=hidden)


class CKSAAPProjectionTests(unittest.TestCase):
    def test_feature_projection_preserves_descriptor_branch_and_gradients(self):
        with patch("pcbert_kla_clean.model.load_encoder", return_value=TinyEncoder()):
            model = TokenGatedCKSAAPProjectionPCBertKla(
                feature_dim=35, encoder_layers=4, fusion_dim=16,
                attention_dim=8, site_token_index=26,
            )
        self.assertEqual(model.cksaap_projection[1].in_features, 8)
        self.assertEqual(model.cksaap_projection[1].out_features, 128)
        self.assertEqual(model.cksaap_projection[3].p, 0.3)
        self.assertEqual(model.feature_projection[1].in_features, 155)
        input_ids = torch.randint(0, 30, (2, 53))
        scores = model(input_ids, torch.ones_like(input_ids), torch.randn(2, 35))
        self.assertEqual(scores.shape, (2,))
        scores.sum().backward()
        self.assertIsNotNone(model.cksaap_projection[1].weight.grad)
        self.assertTrue(all(p.grad is not None for p in model.encoder.parameters()))

    def test_requires_expanded_feature_vector(self):
        with patch("pcbert_kla_clean.model.load_encoder", return_value=TinyEncoder()):
            with self.assertRaisesRegex(ValueError, "CKSAAP projection requires"):
                TokenGatedCKSAAPProjectionPCBertKla(feature_dim=27)


if __name__ == "__main__":
    unittest.main()
