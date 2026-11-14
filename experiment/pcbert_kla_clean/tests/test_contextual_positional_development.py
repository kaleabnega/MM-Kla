import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from pcbert_kla_clean.contextual_positional_development import (
    ContextualPositionalBranch, frozen_context,
)
from pcbert_kla_clean.stacked_development import sequence_tokens


class FakeTokenizer:
    def __call__(self, sequences, return_tensors, padding):
        assert all(len(sequence.split()) == 51 for sequence in sequences)
        assert return_tensors == "pt" and padding
        return {"input_ids": torch.arange(53).expand(len(sequences), -1),
                "attention_mask": torch.ones(len(sequences), 53, dtype=torch.long)}


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, input_ids, attention_mask):
        assert not self.training
        assert not self.weight.requires_grad
        return SimpleNamespace(last_hidden_state=input_ids[..., None].expand(-1, -1, 8).float())


class ContextualPositionalTests(unittest.TestCase):
    def setUp(self):
        self.sequences = ["A" * 25 + "K" + "G" * 25,
                          "G" * 25 + "K" + "A" * 25]

    def test_frozen_context_excludes_special_tokens_and_preserves_site(self):
        with (patch("pcbert_kla_clean.contextual_positional_development.load_tokenizer",
                    return_value=FakeTokenizer()),
              patch("pcbert_kla_clean.contextual_positional_development.load_encoder",
                    return_value=FakeEncoder())):
            context = frozen_context(self.sequences, torch.device("cpu"), batch_size=1)
        self.assertEqual(tuple(context.shape), (2, 51, 8))
        self.assertEqual(context.dtype, torch.float16)
        self.assertEqual(float(context[0, 0, 0]), 1.0)
        self.assertEqual(float(context[0, 25, 0]), 26.0)
        self.assertEqual(float(context[0, 50, 0]), 51.0)

    def test_context_projection_and_positional_encoder_receive_gradients(self):
        tokens = sequence_tokens(self.sequences)
        context = torch.randn(2, 51, 8, dtype=torch.float16)
        model = ContextualPositionalBranch(8)
        logits = model(tokens, context)
        self.assertEqual(tuple(logits.shape), (2,))
        logits.sum().backward()
        self.assertIsNotNone(model.context_projection[1].weight.grad)
        self.assertIsNotNone(model.encoder.weight_ih_l0.grad)
        with self.assertRaises(ValueError):
            model(tokens, context[:, :-1, :])


if __name__ == "__main__":
    unittest.main()
