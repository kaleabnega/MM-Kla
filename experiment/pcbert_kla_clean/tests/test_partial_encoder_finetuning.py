import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from pcbert_kla_clean.model import TokenGatedPCBertKla
from pcbert_kla_clean.backbones import infer_sequence_format, infer_site_token_index


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=12)
        self.embeddings = nn.Embedding(30, 12)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([nn.Linear(12, 12) for _ in range(6)])

    def get_input_embeddings(self):
        return self.embeddings

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        hidden = self.embeddings(input_ids)
        for layer in self.encoder.layer:
            hidden = torch.tanh(layer(hidden))
        return SimpleNamespace(last_hidden_state=hidden)


class PartialEncoderFinetuningTests(unittest.TestCase):
    def make_model(self, **kwargs):
        with patch("pcbert_kla_clean.model.load_encoder", return_value=TinyEncoder()):
            return TokenGatedPCBertKla(
                encoder_layers=6, train_last_encoder_layers=2,
                fusion_dim=16, attention_dim=8, dropout=0.0, **kwargs,
            )

    def test_only_last_layers_and_head_receive_gradients(self):
        torch.manual_seed(11)
        model = self.make_model()
        self.assertFalse(model.encoder.embeddings.weight.requires_grad)
        for index, layer in enumerate(model.encoder.encoder.layer):
            self.assertEqual(layer.weight.requires_grad, index >= 4)

        input_ids = torch.randint(0, 30, (2, 53))
        attention_mask = torch.ones_like(input_ids)
        features = torch.randn(2, 27)
        model(input_ids, attention_mask, features).sum().backward()
        for index, layer in enumerate(model.encoder.encoder.layer):
            if index < 4:
                self.assertIsNone(layer.weight.grad)
            else:
                self.assertIsNotNone(layer.weight.grad)
                self.assertGreater(layer.weight.grad.abs().sum().item(), 0)
        self.assertIsNotNone(model.classifier[-1].weight.grad)

    def test_rejects_invalid_layer_counts_and_conflicting_modes(self):
        for count in (0, 7):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "layer count"):
                with patch("pcbert_kla_clean.model.load_encoder", return_value=TinyEncoder()):
                    TokenGatedPCBertKla(encoder_layers=6, train_last_encoder_layers=count)
        with self.assertRaisesRegex(ValueError, "Last-layer training"):
            self.make_model(freeze_encoder=True)
        with self.assertRaisesRegex(ValueError, "Last-layer training"):
            self.make_model(encoder_adaptation="lora")

    def test_default_still_trains_all_retained_layers(self):
        with patch("pcbert_kla_clean.model.load_encoder", return_value=TinyEncoder()):
            model = TokenGatedPCBertKla(encoder_layers=4)
        self.assertEqual(len(model.encoder.encoder.layer), 4)
        self.assertTrue(all(parameter.requires_grad for parameter in model.encoder.parameters()))

    def test_esm2_full_depth_uses_raw_sequence_and_trains_every_layer(self):
        name = "facebook/esm2_t30_150M_UR50D"
        self.assertEqual(infer_sequence_format(name), "raw")
        self.assertEqual(infer_site_token_index(name), 26)
        with patch("pcbert_kla_clean.model.load_encoder", return_value=TinyEncoder()):
            model = TokenGatedPCBertKla(
                model_name=name, encoder_layers=6,
                fusion_dim=16, attention_dim=8,
            )
        self.assertEqual(len(model.encoder.encoder.layer), 6)
        self.assertTrue(all(parameter.requires_grad for parameter in model.encoder.parameters()))


if __name__ == "__main__":
    unittest.main()
