import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path


class KaggleHumanAdamwNotebookTests(unittest.TestCase):
    @staticmethod
    def notebook():
        path = Path(__file__).resolve().parents[1] / "kaggle_human_adamw_three_models.ipynb"
        return path, json.loads(path.read_text())

    def test_cells_compile_and_development_is_disabled_by_default(self):
        path, notebook = self.notebook()
        self.assertEqual(notebook["nbformat"], 4)
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
        self.assertIn("RUN_MODE = 'check'", sources[0])
        self.assertIn("DECLARED_SEEDS = (42, 123, 777)", sources[0])
        self.assertIn("human-study-development", "\n".join(sources))
        self.assertIn("--development-only", "\n".join(sources))
        self.assertNotIn("--run', 'human-study'", "\n".join(sources))

    def test_summary_uses_three_validation_results_and_combines_seeds(self):
        import pandas as pd

        _, notebook = self.notebook()
        summary = "".join(notebook["cells"][12]["source"])
        metrics = {key: 0.2 for key in
                   ("AUPRC", "AUC", "MCC", "F1", "Pre", "Rec", "SP", "ACC")}

        def make_seed(root, inputs, seed):
            seed_dir = root / f"seed_{seed}"
            for architecture in ("baseline", "token_gated"):
                run_dir = seed_dir / architecture / "human_study" / "runs" / "run123"
                run_dir.mkdir(parents=True)
                config = {
                    "run": "human-study-development", "architecture": architecture,
                    "test_rows": 0, "extra_test_rows": {}, "model_seed": seed,
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
            hybrid = {"mode": "development-only", "test_metrics": {},
                      "benchmark_lock_sha256": "audit",
                      "benchmark_manifest_sha256": "manifest", "seed": seed,
                      "folds": 5, "model_name": "facebook/esm2_t30_150M_UR50D",
                      "selection_metric": "validation AUPRC", "threshold": 0.5,
                      "hyperparameters": {"esm_epochs": 40, "lstm_epochs": 30,
                                          "feature_epochs": 100, "meta_epochs": 50,
                                          "esm_batch_size": 32,
                                          "lstm_batch_size": 32,
                                          "feature_batch_size": 2048,
                                          "patience": 6},
                      "validation_metrics": metrics,
                      "fold_reports": {"META": {"selected_epoch": 4}}}
            (hybrid_dir / "run_report.json").write_text(json.dumps(hybrid))
            namespace = {"RUN_MODE": "development", "SEED": seed,
                         "SEED_DIR": seed_dir, "OUTPUT": root,
                         "INPUT": inputs, "COMMIT": "unchanged-code",
                         "AUDIT_SHA": "audit", "MANIFEST_SHA": "manifest",
                         "DECLARED_SEEDS": (42, 123, 777), "json": json}
            with contextlib.redirect_stdout(io.StringIO()):
                exec(summary, namespace)
            return namespace

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second, third = (root / name for name in ("first", "second", "third"))
            for directory in (first, second, third):
                directory.mkdir()
            first_input = root / "first_input"
            second_input = root / "second_input"
            third_input = root / "third_input"
            for directory in (first_input, second_input, third_input):
                directory.mkdir()
            make_seed(first, first_input, 42)
            self.assertEqual(len(pd.read_csv(first / "validation_results.csv")), 3)
            shutil.copy(first / "seed_42_summary.json", second_input)
            make_seed(second, second_input, 123)
            self.assertEqual(len(pd.read_csv(second / "validation_results.csv")), 6)
            shutil.copy(first / "seed_42_summary.json", third_input)
            shutil.copy(second / "seed_123_summary.json", third_input)
            make_seed(third, third_input, 777)
            table = pd.read_csv(third / "validation_results.csv")
            self.assertEqual(len(table), 9)
            self.assertEqual(set(table["architecture"]),
                             {"baseline", "token_gated", "hybridkla_architecture_aligned"})


if __name__ == "__main__":
    unittest.main()
