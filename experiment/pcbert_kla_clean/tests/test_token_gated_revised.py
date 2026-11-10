import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from pcbert_kla_clean.model import TokenGatedRevisedPCBertKla


class SmallEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=12)
        self.embeddings = nn.Embedding(30, 12)
        self.encoder = SimpleNamespace(layer=nn.ModuleList([nn.Linear(12, 12) for _ in range(4)]))

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        hidden = self.embeddings(input_ids)
        for layer in self.encoder.layer:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


class TokenGatedRevisedTests(unittest.TestCase):
    def test_convolution_branches_receive_gradients(self):
        with patch("pcbert_kla_clean.model.load_encoder", return_value=SmallEncoder()):
            model = TokenGatedRevisedPCBertKla(
                feature_dim=27, fusion_dim=16, attention_dim=8,
                dropout=0.0, site_token_index=26,
            )
        self.assertEqual([branch[0].kernel_size[0] for branch in model.local_filters], [3, 5, 9])
        self.assertEqual(model.sequence_projection[1].in_features, 12 + 3 * 64)
        scores = model(
            input_ids=torch.randint(0, 30, (2, 53)),
            attention_mask=torch.ones(2, 53, dtype=torch.long),
            features=torch.randn(2, 27),
        )
        self.assertEqual(tuple(scores.shape), (2,))
        self.assertTrue(torch.isfinite(scores).all())
        scores.sum().backward()
        for branch in model.local_filters:
            self.assertIsNotNone(branch[0].weight.grad)
            self.assertGreater(branch[0].weight.grad.abs().sum().item(), 0)

    def test_kaggle_notebook_is_validation_only_and_compiles(self):
        path = Path(__file__).resolve().parents[1] / "kaggle_human_local_conv_three_models.ipynb"
        notebook = json.loads(path.read_text())
        sources = []
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            compile(source, f"{path}:cell{index}", "exec")
            self.assertEqual(notebook["cells"][index - 1]["cell_type"], "markdown")
            self.assertEqual(cell["outputs"], [])
            self.assertIsNone(cell["execution_count"])
            sources.append(source)
        joined = "\n".join(sources)
        self.assertIn("RUN_MODE = 'check'", sources[0])
        self.assertIn("'token_gated_revised', 'cuda:1'", joined)
        self.assertIn("human-study-development", joined)
        self.assertIn("--development-only", joined)
        self.assertNotIn("--run', 'human-study'", joined)


if __name__ == "__main__":
    unittest.main()
