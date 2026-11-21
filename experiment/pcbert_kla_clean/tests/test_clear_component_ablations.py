import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from pcbert_kla_clean.model import (
    ConcatFusion, GatedFusion, ResidualBlock, ScalarGatedFusion,
    TokenGatedPCBertKla,
)


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embeddings = nn.Embedding(32, 8)
        self.encoder = SimpleNamespace(layer=nn.ModuleList([nn.Linear(8, 8) for _ in range(4)]))

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        hidden = self.embeddings(input_ids)
        for layer in self.encoder.layer:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


class ClearComponentAblationTests(unittest.TestCase):
    def make_model(self, ablation):
        with patch("pcbert_kla_clean.model.load_encoder", return_value=TinyEncoder()):
            return TokenGatedPCBertKla(
                feature_dim=27, fusion_dim=16, attention_dim=8,
                dropout=0, site_token_index=26, ablation=ablation,
            )

    def test_all_modes_produce_probabilities_and_feature_gradients(self):
        for mode in ("none", "cls_pool", "mean_pool", "max_pool",
                     "concat_fusion", "scalar_gate", "no_residual"):
            with self.subTest(mode=mode):
                model = self.make_model(mode)
                features = torch.randn(2, 27, requires_grad=True)
                scores = model(torch.randint(0, 32, (2, 53)),
                               torch.ones(2, 53, dtype=torch.long), features)
                self.assertEqual(tuple(scores.shape), (2,))
                self.assertTrue(((scores >= 0) & (scores <= 1)).all())
                scores.sum().backward()
                self.assertGreater(features.grad.abs().sum().item(), 0)

    def test_pooling_excludes_special_tokens(self):
        embeddings = torch.tensor([[[100., 100.], [1., 3.], [3., 5.],
                                    [200., 200.], [300., 300.]]])
        mask = torch.tensor([[1, 1, 1, 1, 0]])
        expected = {"cls_pool": [100., 100.], "mean_pool": [2., 4.],
                    "max_pool": [3., 5.]}
        for mode, values in expected.items():
            with self.subTest(mode=mode):
                model = self.make_model(mode)
                actual = model.sequence_context(embeddings, mask)
                torch.testing.assert_close(actual, torch.tensor([values]))

    def test_only_named_component_changes(self):
        expected_fusion = {
            "none": GatedFusion, "concat_fusion": ConcatFusion,
            "scalar_gate": ScalarGatedFusion, "no_residual": GatedFusion,
        }
        for mode, cls in expected_fusion.items():
            model = self.make_model(mode)
            self.assertIsInstance(model.fusion, cls)
            blocks = sum(isinstance(layer, ResidualBlock) for layer in model.classifier)
            self.assertEqual(blocks, 0 if mode == "no_residual" else 2)


if __name__ == "__main__":
    unittest.main()
