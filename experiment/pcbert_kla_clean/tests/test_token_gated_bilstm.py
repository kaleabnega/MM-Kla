import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from pcbert_kla_clean.model import TokenGatedBiLSTMPCBertKla, TokenGatedPCBertKla


class SmallEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=12)
        self.embeddings = nn.Embedding(30, 12)
        self.encoder = SimpleNamespace(layer=nn.ModuleList([nn.Linear(12, 12) for _ in range(4)]))

    def get_input_embeddings(self):
        return self.embeddings

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        hidden = self.embeddings(input_ids)
        for layer in self.encoder.layer:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


def make_inputs():
    ids = torch.zeros(2, 53, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for row, length in enumerate((51, 27)):
        ids[row, 0] = 1
        ids[row, 1:length + 1] = torch.randint(3, 29, (length,))
        ids[row, 26] = 12
        ids[row, length + 1] = 2
        mask[row, :length + 2] = 1
    return ids, mask, torch.randn(2, 27)


class TokenGatedBiLSTMTests(unittest.TestCase):
    def make_model(self):
        with patch("pcbert_kla_clean.model.load_encoder", return_value=SmallEncoder()):
            return TokenGatedBiLSTMPCBertKla(
                feature_dim=27, fusion_dim=16, attention_dim=8,
                dropout=0.0, site_token_index=26,
            )

    def test_forward_ignores_padded_tokens_and_receives_gradients(self):
        torch.manual_seed(7)
        model = self.make_model()
        input_ids, attention_mask, features = make_inputs()
        model.eval()
        original = model(input_ids, attention_mask, features)
        changed = input_ids.clone()
        changed[1, 29:] = 17
        masked_padding = model(changed, attention_mask, features)
        self.assertTrue(torch.allclose(original, masked_padding, atol=1e-6))
        self.assertEqual(tuple(original.shape), (2,))
        self.assertTrue(torch.isfinite(original).all())
        original.sum().backward()
        for parameter in (
            model.residue_embedding.weight,
            model.residue_lstm.weight_ih_l0,
            model.sequence_projection[1].weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_rejects_windows_truncated_before_target(self):
        model = self.make_model()
        input_ids, attention_mask, features = make_inputs()
        attention_mask[1, 26:] = 0
        with self.assertRaisesRegex(ValueError, "target lysine"):
            model(input_ids, attention_mask, features)

    def test_control_and_variant_use_same_optimizer_group_rates(self):
        try:
            from pcbert_kla_clean.training import build_optimizer
        except ModuleNotFoundError as error:
            self.skipTest(f"Training dependencies unavailable: {error}")
        with patch("pcbert_kla_clean.model.load_encoder", side_effect=lambda *_: SmallEncoder()):
            control = TokenGatedPCBertKla(feature_dim=27, fusion_dim=16, attention_dim=8)
            variant = TokenGatedBiLSTMPCBertKla(feature_dim=27, fusion_dim=16, attention_dim=8)
        args = SimpleNamespace(
            optimizer="adamw", learning_rate=2e-5, head_learning_rate=1e-4,
            weight_decay=0.01, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8,
        )
        for model in (control, variant):
            optimizer = build_optimizer(args, model)
            self.assertEqual([group["lr"] for group in optimizer.param_groups], [2e-5, 1e-4])
            self.assertEqual(
                sum(len(group["params"]) for group in optimizer.param_groups),
                sum(parameter.requires_grad for parameter in model.parameters()),
            )


if __name__ == "__main__":
    unittest.main()
