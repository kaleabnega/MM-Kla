import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from pcbert_kla_clean.model import (
    ConcatFusion,
    TokenGatedConvConcatPCBertKla,
    TokenGatedRevisedPCBertKla,
)


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
    def test_kaggle_notebooks_checkout_pinned_commit_before_training(self):
        root = Path(__file__).resolve().parents[1]
        for filename in (
            "kaggle_human_local_conv_three_models.ipynb",
            "kaggle_human_conv_concat_three_models.ipynb",
        ):
            with self.subTest(notebook=filename):
                notebook = json.loads((root / filename).read_text())
                setup = "".join(notebook["cells"][4]["source"])
                self.assertIn("if EXPECTED_COMMIT is None:\n                clone_command.extend(['--depth', '1'])", setup)
                self.assertIn("clone_command.extend([REPO_URL, str(REPO)])", setup)
                self.assertIn("['git', 'checkout', '--detach', EXPECTED_COMMIT]", setup)
                self.assertLess(setup.index("['git', 'checkout', '--detach', EXPECTED_COMMIT]"), setup.index("'-m', 'pip'"))

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

    def test_convolution_concatenation_model_receives_both_branch_gradients(self):
        with patch("pcbert_kla_clean.model.load_encoder", return_value=SmallEncoder()):
            model = TokenGatedConvConcatPCBertKla(
                feature_dim=27, fusion_dim=16, attention_dim=8,
                dropout=0.0, site_token_index=26,
            )
        self.assertIsInstance(model.fusion, ConcatFusion)
        self.assertEqual(model.fusion.projection[1].in_features, 32)
        self.assertEqual([branch[0].kernel_size[0] for branch in model.local_filters], [3, 5, 9])
        features = torch.randn(2, 27, requires_grad=True)
        scores = model(
            input_ids=torch.randint(0, 30, (2, 53)),
            attention_mask=torch.ones(2, 53, dtype=torch.long),
            features=features,
        )
        self.assertEqual(tuple(scores.shape), (2,))
        self.assertTrue(torch.isfinite(scores).all())
        scores.sum().backward()
        self.assertGreater(model.fusion.projection[1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(features.grad.abs().sum().item(), 0)
        for branch in model.local_filters:
            self.assertGreater(branch[0].weight.grad.abs().sum().item(), 0)

    def test_conv_concat_notebook_is_validation_only_and_compiles(self):
        path = Path(__file__).resolve().parents[1] / "kaggle_human_conv_concat_three_models.ipynb"
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
        self.assertIn("'token_gated_conv_concat', 'cuda:1'", joined)
        self.assertIn("human-study-development", joined)
        self.assertIn("--development-only", joined)
        self.assertNotIn("--run', 'human-study'", joined)

    def test_conv_concat_notebook_writes_three_model_validation_table(self):
        import pandas as pd

        path = Path(__file__).resolve().parents[1] / "kaggle_human_conv_concat_three_models.ipynb"
        summary_source = "".join(json.loads(path.read_text())["cells"][12]["source"])
        metrics = {name: 0.2 for name in ("AUPRC", "AUC", "MCC", "F1", "Pre", "Rec", "SP", "ACC")}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seed_dir = root / "seed_42"
            for architecture in ("baseline", "token_gated_conv_concat"):
                run_dir = seed_dir / architecture / "human_study" / "runs" / "run123"
                run_dir.mkdir(parents=True)
                config = {
                    "run": "human-study-development", "architecture": architecture,
                    "test_rows": 0, "extra_test_rows": {}, "model_seed": 42,
                    "human_lock_sha256": "audit", "selected_epoch": 3,
                    "model_name": "Rostlab/prot_bert", "encoder_layers": 4,
                    "optimizer": "adamw", "learning_rate": 2e-5,
                    "weight_decay": 0.01, "scheduler": "linear",
                    "warmup_ratio": 0.1, "epochs": 30, "patience": 5,
                    "batch_size": 4, "positive_weighting": "none",
                    "monitor_metric": "AUPRC", "decision_threshold": 0.5,
                    "training_rows": 39880, "validation_rows": 17713,
                }
                (run_dir / "run_config.json").write_text(json.dumps(config))
                (run_dir / "validation_metrics.json").write_text(json.dumps(metrics))
            hybrid_dir = seed_dir / "hybridkla_architecture_aligned"
            hybrid_dir.mkdir()
            hybrid = {
                "mode": "development-only", "test_metrics": {},
                "benchmark_lock_sha256": "audit", "benchmark_manifest_sha256": "manifest",
                "seed": 42, "folds": 5, "model_name": "facebook/esm2_t30_150M_UR50D",
                "selection_metric": "validation AUPRC", "threshold": 0.5,
                "hyperparameters": {
                    "esm_epochs": 40, "lstm_epochs": 30, "feature_epochs": 100,
                    "meta_epochs": 50, "esm_batch_size": 32, "lstm_batch_size": 32,
                    "feature_batch_size": 2048, "patience": 6,
                },
                "validation_metrics": metrics, "fold_reports": {"META": {"selected_epoch": 4}},
            }
            (hybrid_dir / "run_report.json").write_text(json.dumps(hybrid))
            namespace = {
                "RUN_MODE": "development", "RUN_SEED": 42, "OUTPUT": root,
                "COMMIT": "same-code", "AUDIT_SHA": "audit", "MANIFEST_SHA": "manifest",
                "json": json,
            }
            with contextlib.redirect_stdout(io.StringIO()):
                exec(summary_source, namespace)
            table = pd.read_csv(root / "validation_results.csv")
            self.assertEqual(len(table), 3)
            self.assertEqual(set(table["architecture"]), {
                "baseline", "token_gated_conv_concat", "hybridkla_architecture_aligned",
            })
            self.assertTrue(set(metrics).issubset(table.columns))
            self.assertEqual(len(pd.read_csv(root / "validation_paired_auprc.csv")), 1)
            saved = json.loads((root / "seed_42_summary.json").read_text())
            self.assertEqual(saved["settings"]["fusion_mode"], "concatenate_projected_branches")
            run_dir = seed_dir / "baseline" / "human_study" / "runs" / "run123"
            config = json.loads((run_dir / "run_config.json").read_text())
            config["test_rows"] = 1
            (run_dir / "run_config.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "development boundary"):
                exec(summary_source, namespace)


if __name__ == "__main__":
    unittest.main()
