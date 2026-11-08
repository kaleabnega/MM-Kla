import json
import csv
import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT = Path(__file__).resolve().parents[1]


def load_runner():
    import importlib.util

    path = PROJECT / "scripts/run_original_kladb_external.py"
    spec = importlib.util.spec_from_file_location("run_original_kladb_external", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OriginalKladbExternalNotebookTests(unittest.TestCase):
    def test_notebook_cells_compile_and_run_three_full_data_models(self):
        path = PROJECT / "kaggle_original_train_kladb_external.ipynb"
        notebook = json.loads(path.read_text())
        self.assertEqual(notebook["nbformat"], 4)
        code = []
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                compile(source, f"{path}:cell{index}", "exec")
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(notebook["cells"][index - 1]["cell_type"], "markdown")
                code.append(source)
        self.assertIn("RUN_THREE_MODEL = True", code[0])
        self.assertIn("RUN_HYBRIDKLA_PILOT = False", code[0])
        self.assertIn("RUN_SEEDS = (42, 123, 777)", code[0])
        self.assertIn("HYBRID_ESM_EPOCHS = 40", code[0])
        self.assertIn("HYBRID_META_EPOCHS = 50", code[0])
        self.assertIn("DECLARED_SEEDS = (42, 123, 777)", code[0])
        self.assertIn("--check-only", "\n".join(code))
        self.assertIn("cuda:1", "\n".join(code))
        self.assertIn("external_predictions.csv", "\n".join(code))
        self.assertIn("--original-external-csv", "\n".join(code))
        self.assertIn("--training-scope", "\n".join(code))
        self.assertIn("--pilot", "\n".join(code))
        self.assertIn("--full-refit", "\n".join(code))
        self.assertIn("--esm-epochs", "\n".join(code))
        self.assertIn("three_model_summary.csv", "\n".join(code))
        self.assertNotIn("matched_subset", "\n".join(code))

    def test_external_export_is_deterministic_when_source_is_available(self):
        source = PROJECT / "outputs/kladb_human_locked/full_manifest.csv"
        if not source.exists():
            self.skipTest("Generated KlaDB source manifest is not present")
        import importlib.util

        path = PROJECT / "scripts/export_kladb_external.py"
        spec = importlib.util.spec_from_file_location("export_kladb_external", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            manifest = module.export(source, Path(temporary))
            self.assertEqual(manifest["sha256"], "0165ebe6d79e52bd74a848aa76e788e27804bf78ba2e8c98126d119cb449eae9")
            self.assertEqual(manifest["counts"], {"rows": 5095, "positives": 254, "putative_negatives": 4841, "proteins": 81})
            self.assertEqual(json.loads((Path(temporary) / "external_test_manifest.json").read_text()), manifest)

    def test_predictions_use_full_protein_position(self):
        runner = load_runner()
        split = SimpleNamespace(records=[SimpleNamespace(name="P12345:778", sequence="A" * 25 + "K" + "A" * 25, label=1)])
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "predictions.csv"
            runner.save_external_predictions(output, split, [0.8])
            with output.open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["position_1based"], "778")
            self.assertEqual(row["predicted_label"], "1")

    def test_hybridkla_uses_same_grouped_original_rows(self):
        external_dir = PROJECT / "outputs/kladb_external_upload"
        if not (external_dir / "kladb_human_external_test.csv").exists():
            self.skipTest("Generated KlaDB external export is not present")
        if importlib.util.find_spec("sklearn") is None or importlib.util.find_spec("Bio") is None:
            self.skipTest("Local ML data dependencies are not installed")
        from types import SimpleNamespace

        sys.path.insert(0, str(PROJECT / "scripts"))
        sys.path.insert(0, str(PROJECT / "src"))
        from run_hybridkla_architecture import grouped_folds, read_original_kladb
        from pcbert_kla_clean.data import load_split
        from run_original_kladb_external import grouped_validation_indices

        source = PROJECT.parents[1] / "baselines/PCBert-Kla-original/data"
        train = load_split(source / "train.csv", source / "feature_train.csv")
        mapping = PROJECT.parents[1] / "experiment-1/structure_aware_kla/results/botrytis_b0510_window_mapping/all_window_matches.csv"
        clusters = PROJECT / "review_data/homology_clusters_mmseqs2.csv"
        for seed in (42, 123, 777):
            args = SimpleNamespace(original_external_csv=external_dir / "kladb_human_external_test.csv",
                                   original_external_manifest=external_dir / "external_test_manifest.json", seed=seed)
            partitions, audit = read_original_kladb(args)
            train_idx, val_idx, validation = grouped_validation_indices(train, mapping, clusters, seed)
            self.assertEqual(len(partitions["train"]), len(train_idx))
            self.assertEqual(len(partitions["validation"]), len(val_idx))
            self.assertEqual(len(partitions["full_train"]), 3487)
            self.assertEqual(sum(row["label"] == "1" for row in partitions["full_train"]), 1720)
            self.assertEqual(audit["training_indices_sha256"], validation["training_indices_sha256"])
            self.assertEqual(audit["validation_indices_sha256"], validation["validation_indices_sha256"])
            self.assertEqual(len(partitions["external_test"]), 5095)
            self.assertEqual(sum(len(fold) for fold in grouped_folds(partitions["train"], 5, seed)), len(train_idx))
            self.assertEqual(sum(len(fold) for fold in grouped_folds(partitions["full_train"], 5, seed)), 3487)

    def test_result_table_requires_full_data_refits(self):
        external_dir = PROJECT / "outputs/kladb_external_upload"
        external_csv = external_dir / "kladb_human_external_test.csv"
        if not external_csv.is_file():
            self.skipTest("Generated KlaDB external export is not present")
        import numpy as np
        import pandas as pd

        manifest_file = external_dir / "external_test_manifest.json"
        manifest = json.loads(manifest_file.read_text())
        locked = pd.read_csv(external_csv)
        names = locked["accession"].astype(str) + ":" + locked["position_1based"].astype(str)
        metrics = {name: 0.5 for name in ("ACC", "AUC", "AUPRC", "F1", "MCC", "Pre", "Rec", "SP")}
        split_hash = ("training-hash", "validation-hash")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            for architecture in ("baseline", "token_gated"):
                destination = output / "seed_42" / architecture
                destination.mkdir(parents=True)
                report = {
                    "seed": 42, "architecture": architecture, "training_scope": "full_refit",
                    "git_commit": "test-commit", "selected_epoch": 3, "refit_epochs": 3,
                    "preflight": {"external_sha256": manifest["sha256"], "validation": {
                        "training_indices_sha256": split_hash[0], "validation_indices_sha256": split_hash[1]}},
                    "refit_training": {"training_positive_rows": 1720,
                                       "training_negative_rows": 1767},
                    "external_metrics": metrics,
                }
                (destination / "run_report.json").write_text(json.dumps(report))
                pd.DataFrame({"name": names, "true_label": locked["label"],
                              "window_51aa": locked["window_51aa"], "score": 0.5}).to_csv(
                                  destination / "external_predictions.csv", index=False)
            hybrid = output / "seed_42/hybridkla_architecture_aligned"
            hybrid.mkdir(parents=True)
            (hybrid / "run_report.json").write_text(json.dumps({
                "seed": 42, "git_commit": "test-commit", "mode": "locked-test-evaluation",
                "benchmark_lock_sha256": hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
                "benchmark_input_audit": {"external_csv_sha256": manifest["sha256"],
                    "training_rows": 2815, "refit_training_rows": 3487,
                    "training_scope": "full original training rows; grouped OOF stacking",
                    "training_indices_sha256": split_hash[0], "validation_indices_sha256": split_hash[1]},
                "fold_reports": {"META": {"selected_epoch": 2}},
                "full_refit_fold_reports": {"META": {"trained_epochs": 2}},
                "test_metrics": {"external_test": metrics},
            }))
            pd.DataFrame({"Site_ID": names, "Label": locked["label"], "Score": 0.5}).to_csv(
                hybrid / "external_test_predictions.csv", index=False)
            notebook = json.loads((PROJECT / "kaggle_original_train_kladb_external.ipynb").read_text())
            source = "".join(next(cell["source"] for cell in notebook["cells"]
                                  if cell["cell_type"] == "code" and
                                  "three_model_summary.csv" in "".join(cell["source"])))
            namespace = {"OUTPUT": output, "EXTERNAL_CSV": external_csv,
                         "EXTERNAL_MANIFEST": manifest_file, "COMMIT": "test-commit",
                         "DECLARED_SEEDS": (42, 123, 777), "manifest": manifest,
                         "json": json,
                         "sha256": lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                         "display": lambda _: None}
            exec(source, namespace)
            results = namespace["results"]
            self.assertEqual(len(results), 3)
            self.assertEqual(set(results["training_rows"]), {3487})
            self.assertFalse((output / "three_model_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
