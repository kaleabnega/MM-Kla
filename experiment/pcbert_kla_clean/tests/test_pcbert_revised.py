import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PROJECT / "src"))

from lock_pcbert_revised import ROOT, build_lock, sha256
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

    def test_notebook_cells_parse(self):
        notebook = json.loads((PROJECT / "kaggle_pcbert_revised.ipynb").read_text())
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))


if __name__ == "__main__":
    unittest.main()
