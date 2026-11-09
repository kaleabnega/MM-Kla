import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace


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
        self.assertIn("RUN_SEED = 42", sources[0])
        self.assertIn("RUN_SEED not in DECLARED_SEEDS", sources[0])
        self.assertIn("run_seed(RUN_SEED)", sources[4])
        self.assertIn("human-study-development", "\n".join(sources))
        self.assertIn("--development-only", "\n".join(sources))
        self.assertNotIn("--run', 'human-study'", "\n".join(sources))

    def test_summary_uses_three_validation_results_for_selected_seed(self):
        import pandas as pd

        _, notebook = self.notebook()
        summary = "".join(notebook["cells"][12]["source"])
        metrics = {key: 0.2 for key in
                   ("AUPRC", "AUC", "MCC", "F1", "Pre", "Rec", "SP", "ACC")}

        def make_seed(root, seed):
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
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_seed(root, 42)
            namespace = {"RUN_MODE": "development", "RUN_SEED": 42,
                         "OUTPUT": root, "COMMIT": "unchanged-code",
                         "AUDIT_SHA": "audit", "MANIFEST_SHA": "manifest",
                         "DECLARED_SEEDS": (42, 123, 777), "json": json}
            with contextlib.redirect_stdout(io.StringIO()):
                exec(summary, namespace)
            table = pd.read_csv(root / "validation_results.csv")
            self.assertEqual(len(table), 3)
            self.assertEqual(set(table["seed"]), {42})
            self.assertEqual(set(table["architecture"]),
                             {"baseline", "token_gated", "hybridkla_architecture_aligned"})
            self.assertEqual(len(pd.read_csv(root / "validation_paired_auprc.csv")), 1)
            self.assertFalse((root / "validation_mean_std.csv").exists())
            self.assertTrue((root / "seed_42_summary.json").is_file())

            (root / "seed_42" / "hybridkla_architecture_aligned" / "run_report.json").unlink()
            with self.assertRaises(FileNotFoundError):
                exec(summary, namespace)

    def test_development_launches_three_validation_only_runs(self):
        _, notebook = self.notebook()
        source = "".join(notebook["cells"][10]["source"])
        commands = []

        class CompletedProcess:
            returncode = 0

            def poll(self):
                return 0

        def launch(command, **kwargs):
            commands.append(command)
            return CompletedProcess()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            namespace = {
                "RUN_MODE": "development", "RUN_SEED": 42,
                "OUTPUT": root, "PROJECT": root, "LOCK": root / "split_audit.json",
                "RESOURCE_DIR": root / "resources", "sys": sys, "time": time,
                "subprocess": SimpleNamespace(Popen=launch, STDOUT=subprocess.STDOUT),
            }
            with contextlib.redirect_stdout(io.StringIO()):
                exec(source, namespace)
        self.assertEqual(len(commands), 3)
        self.assertTrue(all(command[command.index("--seed") + 1] == "42" for command in commands))
        self.assertEqual(sum("human-study-development" in command for command in commands), 2)
        self.assertEqual(sum("--development-only" in command for command in commands), 1)
        self.assertTrue(all("human-study" not in command for command in commands))

    def test_aggregation_rejects_mismatched_bundles(self):
        script = Path(__file__).resolve().parents[1] / "scripts/aggregate_human_adamw_seeds.py"
        spec = importlib.util.spec_from_file_location("aggregate_human_adamw_seeds", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        metrics = {key: 0.2 for key in module.METRICS}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for seed in (42, 123, 777):
                records = [{"seed": seed, "architecture": architecture,
                            "selected_epoch": 2, **metrics}
                           for architecture in sorted(module.ARCHITECTURES)]
                bundle = {"schema_version": 1, "seed": seed,
                          "git_commit": "same-code", "audit_sha256": "audit",
                          "manifest_sha256": "manifest", "settings": {"lr": 2e-5},
                          "records": records}
                path = root / f"seed_{seed}_summary.json"
                path.write_text(json.dumps(bundle))
                paths.append(path)
            rows, paired, summary = module.aggregate(paths)
            self.assertEqual(len(rows), 9)
            self.assertEqual(len(paired), 3)
            self.assertEqual(len(summary), 3)
            with self.assertRaisesRegex(ValueError, "exactly 3"):
                module.aggregate(paths[:2])
            changed = json.loads(paths[2].read_text())
            changed["git_commit"] = "different-code"
            paths[2].write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "git_commit"):
                module.aggregate(paths)


if __name__ == "__main__":
    unittest.main()
