import contextlib
import json
import hashlib
import io
import shutil
import tempfile
import unittest
from pathlib import Path


class KaggleHumanNotebookTests(unittest.TestCase):
    @staticmethod
    def notebook():
        path = Path(__file__).resolve().parents[1] / "kaggle_human_study.ipynb"
        return json.loads(path.read_text())

    def test_notebook_cells_compile_and_tests_are_disabled_by_default(self):
        path = Path(__file__).resolve().parents[1] / "kaggle_human_study.ipynb"
        notebook = self.notebook()
        self.assertEqual(notebook["nbformat"], 4)
        cells = notebook["cells"]
        self.assertTrue(cells)
        self.assertEqual(cells[0]["cell_type"], "markdown")
        code = []
        for index, cell in enumerate(cells):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            compile(source, f"{path}:cell{index}", "exec")
            self.assertEqual(cell["outputs"], [])
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cells[index - 1]["cell_type"], "markdown")
            code.append(source)
        self.assertIn("RUN_FINAL_EVALUATION = False", code[0])
        self.assertIn("RUN_HYBRIDKLA_PREPARATION = False", code[0])
        self.assertIn("RUN_HYBRIDKLA_PILOT = False", code[0])
        self.assertIn("RUN_HYBRIDKLA_ADAPTED = False", code[0])
        self.assertIn("FINAL_SEEDS = (42, 123, 777)", code[0])
        self.assertIn("FINAL_SEED = 42", code[0])
        self.assertIn("human-study-development", "\n".join(code))
        self.assertIn("human-study'", "\n".join(code))
        self.assertIn("EXPECTED_MANIFEST_SHA256", "\n".join(code))
        results = next(cell for cell in cells if cell["id"] == "final-results")
        self.assertIn("human_paired_differences.csv", "".join(results["source"]))
        preparation = next(cell for cell in cells if cell["id"] == "hybridkla-preparation")
        self.assertIn("export_hybridkla_splits.py", "".join(preparation["source"]))
        self.assertIn("audit_hybridkla_overlap.py", "".join(preparation["source"]))
        self.assertNotIn("run_replication.py", "".join(preparation["source"]))
        adapted = next(cell for cell in cells if cell["id"] == "hybridkla-adapted-run")
        self.assertIn("run_hybridkla_adapted.py", "".join(adapted["source"]))
        self.assertIn("--human-lock", "".join(adapted["source"]))

    def test_separate_seed_outputs_aggregate_only_when_complete(self):
        import pandas as pd

        cell = next(cell for cell in self.notebook()["cells"] if cell["id"] == "final-results")
        source = "".join(cell["source"])

        def sha256(path):
            return hashlib.sha256(path.read_bytes()).hexdigest()

        audit = {
            "sampled_training_rows": 39880,
            "partitions_before_training_resampling": {
                "validation": {"rows": 17713},
                "internal_test": {"rows": 35455},
                "external_test": {"rows": 5095},
            },
        }
        metrics = {name: 0.7 for name in ("ACC", "AUC", "AUPRC", "F1", "MCC", "Pre", "Rec", "SP")}

        def run_seed(root, inputs, seed):
            for architecture in ("baseline", "token_gated"):
                run_dir = root / "final" / f"seed_{seed}" / "human_study" / "runs" / architecture
                run_dir.mkdir(parents=True)
                config = {
                    "architecture": architecture, "model_seed": seed, "run": "human-study",
                    "human_lock_sha256": "audit", "model_name": "Rostlab/prot_bert",
                    "encoder_layers": 4, "freeze_encoder": False, "epochs": 30,
                    "patience": 5, "batch_size": 4, "optimizer": "sgd",
                    "learning_rate": 0.003, "weight_decay": 0.0, "scheduler": "none",
                    "positive_weighting": "none", "monitor_metric": "AUPRC",
                    "decision_threshold": 0.5, "calibrate_threshold": "none",
                    "ablation": "none", "training_rows": 39880, "validation_rows": 17713,
                    "test_rows": 35455, "extra_test_rows": {"external_study": 5095},
                    "completed_epochs": 12, "selected_epoch": 7, "save_predictions": True,
                }
                (run_dir / "run_config.json").write_text(json.dumps(config))
                for name in ("test_metrics.json", "external_study_metrics.json"):
                    (run_dir / name).write_text(json.dumps(metrics))
                for name in ("diagnostic_test_predictions.csv", "external_study_predictions.csv"):
                    (run_dir / name).write_text("label,score\n1,0.8\n")
            namespace = {
                "OUTPUT_ROOT": root, "INPUT_DIR": inputs, "final_times": {"completed": True},
                "FINAL_SEED": seed, "FINAL_SEEDS": (42, 123, 777),
                "GIT_COMMIT": "unchanged-code", "EXPECTED_AUDIT_SHA256": "audit",
                "EXPECTED_MANIFEST_SHA256": "manifest", "FINAL_EPOCHS": 30,
                "FINAL_PATIENCE": 5, "BATCH_SIZE": 4,
                "ARCHITECTURES": ("baseline", "token_gated"),
                "full_audit": audit, "file_sha256": sha256, "json": json,
            }
            with contextlib.redirect_stdout(io.StringIO()):
                exec(source, namespace)
            return namespace

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            first = base / "first"
            first_inputs = base / "first_inputs"
            first.mkdir()
            first_inputs.mkdir()
            run_seed(first, first_inputs, 42)
            self.assertTrue((first / "final_seed_42.json").is_file())
            self.assertFalse((first / "human_final_summary.csv").exists())

            second = base / "second"
            second_inputs = base / "second_inputs"
            second.mkdir()
            second_inputs.mkdir()
            shutil.copytree(first, second_inputs / "version_42")
            run_seed(second, second_inputs, 123)
            self.assertFalse((second / "human_final_summary.csv").exists())

            third = base / "third"
            third_inputs = base / "third_inputs"
            third.mkdir()
            third_inputs.mkdir()
            shutil.copytree(first, third_inputs / "version_42")
            shutil.copytree(second, third_inputs / "version_123")
            run_seed(third, third_inputs, 777)
            result = pd.read_csv(third / "human_final_metrics.csv")
            self.assertEqual(len(result), 12)
            self.assertEqual(set(result["seed"]), {42, 123, 777})
            self.assertTrue((third / "human_final_summary.csv").is_file())
            self.assertTrue((third / "human_paired_differences.csv").is_file())

            shutil.copytree(first, third_inputs / "duplicate_42")
            with self.assertRaisesRegex(ValueError, "Duplicate final result bundle"):
                run_seed(base / "duplicate_run", third_inputs, 777)


if __name__ == "__main__":
    unittest.main()
