import ast
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PROJECT / "src"))

from lock_pcbert_revised import ROOT, build_lock, sha256
from run_hybridkla_architecture import grouped_folds, read_lock
from pcbert_kla_clean.pcbert_revised import load_pcbert_revised


class RevisedPCBertBenchmarkTests(unittest.TestCase):
    def test_committed_lock_regenerates_and_loads(self):
        committed = PROJECT / "review_data/pcbert_revised_v1"
        with tempfile.TemporaryDirectory() as temporary:
            rebuilt = Path(temporary)
            report = build_lock(
                ROOT / "baselines/PCBert-Kla-original",
                ROOT / "experiment-1/structure_aware_kla/results/botrytis_b0510_window_mapping/all_window_matches.csv",
                PROJECT / "review_data/homology_clusters_mmseqs2.csv",
                PROJECT / "review_data/all_vs_all_mmseqs2.tsv",
                rebuilt,
            )
            for name in ("full_manifest.csv", "model_manifest.csv", "split_audit.json"):
                self.assertEqual(sha256(rebuilt / name), sha256(committed / name), name)
        self.assertEqual(report["original_test_if_train_overlaps_removed"],
                         {"rows": 99, "positives": 7, "negatives": 92})
        development, validation_indices, test = load_pcbert_revised(
            committed / "split_audit.json"
        )
        self.assertEqual(len(development.records) - len(validation_indices), 488)
        self.assertEqual(len(validation_indices), 194)
        self.assertEqual(len(test.records), 387)
        self.assertEqual(int(test.labels.sum()), 35)
        self.assertEqual(development.features.shape[1], 27)

    def test_manifest_rejects_tampering(self):
        committed = PROJECT / "review_data/pcbert_revised_v1"
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in ("full_manifest.csv", "model_manifest.csv", "split_audit.json"):
                (directory / name).write_bytes((committed / name).read_bytes())
            with (directory / "model_manifest.csv").open("a") as handle:
                handle.write("invalid\n")
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_pcbert_revised(directory / "split_audit.json")
            with self.assertRaisesRegex(ValueError, "does not match"):
                read_lock(directory / "split_audit.json")

    def test_hybridkla_uses_the_same_locked_rows(self):
        lock = PROJECT / "review_data/pcbert_revised_v1/split_audit.json"
        partitions, audit = read_lock(lock)
        self.assertEqual({name: len(rows) for name, rows in partitions.items()},
                         {"train": 488, "validation": 194, "internal_test": 387,
                          "external_test": 0})
        self.assertEqual(sum(row["label"] == "1" for row in partitions["internal_test"]), 35)
        self.assertEqual(len(grouped_folds(partitions["train"], 5, 42)), 5)
        for name in ("train", "validation", "internal_test"):
            self.assertTrue(all(row["source_site_id"] == row["window_id"] and
                                row["group_id"] == row["candidate_group"] and
                                row["window_51aa"] == row["sequence"]
                                for row in partitions[name]))
        self.assertEqual(audit["sampled_training_rows"], len(partitions["train"]))

    def test_notebook_cells_parse(self):
        notebook = json.loads((PROJECT / "kaggle_pcbert_revised.ipynb").read_text())
        code = []
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                ast.parse(source)
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(notebook["cells"][index - 1]["cell_type"], "markdown")
                code.append(source)
        self.assertIn("RUN_PAIRED_MODELS = False", code[0])
        self.assertIn("RUN_HYBRIDKLA_PILOT = False", code[0])
        self.assertIn("RUN_HYBRIDKLA_FULL = False", code[0])
        self.assertTrue(any("run_hybridkla_architecture.py" in source for source in code))
        self.assertTrue(any("three_model_internal_test.csv" in source for source in code))
        self.assertTrue(any("internal_test_predictions.csv" in source for source in code))

    def test_notebook_comparison_reads_three_locked_models(self):
        notebook = json.loads((PROJECT / "kaggle_pcbert_revised.ipynb").read_text())
        source = next("".join(cell["source"]) for cell in notebook["cells"]
                      if "three_model_internal_test.csv" in "".join(cell["source"]))
        lock = PROJECT / "review_data/pcbert_revised_v1/split_audit.json"
        metrics = {name: 0.5 for name in ("AUPRC", "AUC", "MCC", "F1",
                                          "Pre", "Rec", "SP", "ACC")}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            for name in ("baseline", "token_gated"):
                directory = output / "seed_42" / name / "pcbert_revised" / "runs" / "fixture"
                directory.mkdir(parents=True)
                (directory / "test_metrics.json").write_text(json.dumps(metrics))
                (directory / "run_config.json").write_text(json.dumps({
                    "revised_lock_sha256": sha256(lock), "model_seed": 42,
                }))
            hybrid = output / "seed_42" / "hybridkla"
            hybrid.mkdir(parents=True)
            (hybrid / "run_report.json").write_text(json.dumps({
                "benchmark_lock_sha256": sha256(lock), "seed": 42,
                "test_metrics": {"internal_test": metrics},
            }))
            namespace = {"OUTPUT": output, "SEEDS": (42,), "lock": lock,
                         "sha256": sha256, "json": json, "display": lambda value: None}
            exec(compile(source, "comparison cell", "exec"), namespace)
            self.assertEqual(set(namespace["results"]["architecture"]),
                             {"baseline", "token_gated", "hybridkla_architecture_aligned"})
            self.assertTrue((output / "three_model_summary.csv").is_file())

    def test_notebook_intervals_accept_all_three_prediction_formats(self):
        import pandas as pd

        notebook = json.loads((PROJECT / "kaggle_pcbert_revised.ipynb").read_text())
        source = next("".join(cell["source"]) for cell in notebook["cells"]
                      if "three_model_group_bootstrap_intervals.csv" in "".join(cell["source"]))
        committed = PROJECT / "review_data/pcbert_revised_v1"
        audit = json.loads((committed / "split_audit.json").read_text())
        with (committed / "model_manifest.csv").open(newline="") as handle:
            test_rows = [row for row in csv.DictReader(handle)
                         if row["partition"] == "internal_test"]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            for name in ("baseline", "token_gated"):
                directory = output / "seed_42" / name / "pcbert_revised" / "runs" / "fixture"
                directory.mkdir(parents=True)
                with (directory / "diagnostic_test_predictions.csv").open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(("name", "true_label", "score"))
                    writer.writerows((row["window_id"], row["label"],
                                      0.8 if row["label"] == "1" else 0.2)
                                     for row in test_rows)
            hybrid = output / "seed_42" / "hybridkla"
            hybrid.mkdir(parents=True)
            with (hybrid / "internal_test_predictions.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("Site_ID", "Label", "Score"))
                writer.writerows((row["window_id"], row["label"],
                                  0.8 if row["label"] == "1" else 0.2)
                                 for row in test_rows)
            namespace = {"OUTPUT": output, "SEEDS": (42,), "committed": committed,
                         "audit": audit, "pd": pd, "display": lambda value: None}
            sklearn = ModuleType("sklearn")
            metrics = ModuleType("sklearn.metrics")
            metrics.average_precision_score = lambda labels, scores: 1.0
            metrics.roc_auc_score = lambda labels, scores: 1.0
            sklearn.metrics = metrics
            with patch.dict(sys.modules, {"sklearn": sklearn, "sklearn.metrics": metrics}):
                exec(compile(source, "interval cell", "exec"), namespace)
            self.assertEqual(len(namespace["intervals"]), 3)
            self.assertTrue((output / "three_model_group_bootstrap_intervals.csv").is_file())


if __name__ == "__main__":
    unittest.main()
