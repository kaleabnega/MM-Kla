import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from pcbert_kla_clean.model import TokenGatedPCBertKla


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=12)
        self.embeddings = nn.Embedding(30, 12)
        self.encoder = SimpleNamespace(layer=nn.ModuleList([nn.Linear(12, 12)]))

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        hidden = self.encoder.layer[0](self.embeddings(input_ids))
        if hasattr(self, "adapter"):
            hidden = hidden + self.adapter(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


class Esm2LoraNotebookTests(unittest.TestCase):
    def test_lora_path_keeps_classifier_trainable(self):
        def fake_adapter(encoder, rank, alpha, dropout):
            self.assertEqual((rank, alpha, dropout), (8, 32, 0.3))
            for parameter in encoder.parameters():
                parameter.requires_grad = False
            encoder.adapter = nn.Linear(12, 12, bias=False)
            return encoder

        with patch("pcbert_kla_clean.model.load_encoder", return_value=TinyEncoder()), patch(
            "pcbert_kla_clean.model.add_lora_to_encoder", side_effect=fake_adapter
        ):
            model = TokenGatedPCBertKla(
                feature_dim=27, fusion_dim=16, attention_dim=8,
                encoder_layers=1, encoder_adaptation="lora",
            )
        self.assertFalse(model.encoder.embeddings.weight.requires_grad)
        self.assertTrue(model.encoder.adapter.weight.requires_grad)
        self.assertTrue(model.classifier[-1].weight.requires_grad)
        scores = model(
            input_ids=torch.randint(0, 30, (2, 53)),
            attention_mask=torch.ones(2, 53, dtype=torch.long),
            features=torch.randn(2, 27),
        )
        self.assertEqual(tuple(scores.shape), (2,))
        scores.sum().backward()
        self.assertIsNone(model.encoder.embeddings.weight.grad)
        self.assertGreater(model.encoder.adapter.weight.grad.abs().sum().item(), 0)

    def test_notebook_is_compilable_and_validation_only(self):
        path = Path(__file__).resolve().parents[1] / "kaggle_human_esm2_lora_three_models.ipynb"
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
        combined = "\n".join(sources)
        self.assertIn("RUN_MODE = 'check'", sources[0])
        self.assertIn("--encoder-adaptation', 'lora'", combined)
        self.assertIn("'facebook/esm2_t33_650M_UR50D'", combined)
        self.assertIn("human-study-development", combined)
        self.assertIn("--development-only", combined)
        self.assertNotIn("--run', 'human-study'", combined)
        self.assertIn("AUDIT_SHA", combined)
        self.assertIn("MANIFEST_SHA", combined)

    def test_validation_table_reads_three_saved_runs_and_rejects_test_rows(self):
        import pandas as pd

        path = Path(__file__).resolve().parents[1] / "kaggle_human_esm2_lora_three_models.ipynb"
        source = "".join(json.loads(path.read_text())["cells"][14]["source"])
        metrics = {name: 0.2 for name in ("AUPRC", "AUC", "MCC", "F1", "Pre", "Rec", "SP", "ACC")}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            seed_dir = output / "seed_42"
            for architecture, model_name, layers, adaptation in (
                ("baseline", "Rostlab/prot_bert", 4, "full"),
                ("token_gated", "facebook/esm2_t33_650M_UR50D", 33, "lora"),
            ):
                run_dir = seed_dir / architecture / "human_study" / "runs" / "test"
                run_dir.mkdir(parents=True)
                config = {
                    "run": "human-study-development", "architecture": architecture,
                    "test_rows": 0, "extra_test_rows": {}, "model_seed": 42,
                    "human_lock_sha256": "audit", "model_name": model_name,
                    "encoder_layers": layers, "encoder_adaptation": adaptation,
                    "optimizer": "adamw", "learning_rate": 2e-5,
                    "weight_decay": 0.01, "scheduler": "linear",
                    "warmup_ratio": 0.1, "epochs": 30, "patience": 5,
                    "batch_size": 4, "positive_weighting": "none",
                    "monitor_metric": "AUPRC", "decision_threshold": 0.5,
                    "training_rows": 39880, "validation_rows": 17713,
                    "selected_epoch": 3, "lora_rank": 8, "lora_alpha": 32,
                    "lora_dropout": 0.3,
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
                "hyperparameters": {"esm_epochs": 40, "lstm_epochs": 30,
                                    "feature_epochs": 100, "meta_epochs": 50,
                                    "esm_batch_size": 32, "lstm_batch_size": 32,
                                    "feature_batch_size": 2048, "patience": 6},
                "validation_metrics": metrics,
                "fold_reports": {"META": {"selected_epoch": 4}},
            }
            (hybrid_dir / "run_report.json").write_text(json.dumps(hybrid))
            namespace = {"RUN_MODE": "development", "RUN_SEED": 42,
                         "OUTPUT": output, "COMMIT": "same-code",
                         "AUDIT_SHA": "audit", "MANIFEST_SHA": "manifest",
                         "json": json}
            exec(source, namespace)
            table = pd.read_csv(output / "validation_results.csv")
            self.assertEqual(set(table.architecture), {
                "baseline", "token_gated", "hybridkla_architecture_aligned",
            })
            config["test_rows"] = 1
            (run_dir / "run_config.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "development boundary"):
                exec(source, namespace)


if __name__ == "__main__":
    unittest.main()
