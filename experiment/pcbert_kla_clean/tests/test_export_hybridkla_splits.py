import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from export_hybridkla_splits import export_splits, sha256


class HybridKlaExportTests(unittest.TestCase):
    def make_lock(self, root, duplicate_window=False):
        manifest = root / "model_manifest.csv"
        partitions = ("train", "validation", "internal_test", "external_test")
        rows = []
        for index, partition in enumerate(partitions):
            window = "A" * 25 + "K" + chr(ord("A") + index) * 25
            if duplicate_window and index == 1:
                window = rows[0]["window_51aa"]
            rows.append({
                "accession": f"P{index}", "position_1based": "26",
                "window_51aa": window, "label": str(int(index % 2 == 0)),
                "homology_group": f"G{index}", "partition": partition,
            })
        with manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0], lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        audit = {
            "output_sha256": {"model_manifest.csv": sha256(manifest)},
            "sampled_training_rows": 1,
            "partitions_before_training_resampling": {
                partition: {"rows": 1, "positives": int(index % 2 == 0)}
                for index, partition in enumerate(partitions)
            },
        }
        lock = root / "split_audit.json"
        lock.write_text(json.dumps(audit))
        return lock

    def test_export_preserves_partitions_and_unique_site_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = self.make_lock(root)
            output = root / "export"
            report = export_splits(lock, output)
            self.assertEqual(report["counts"]["train"], {"rows": 1, "positives": 1})
            self.assertEqual(set(report["output_sha256"]), {
                "train.csv", "validation.csv", "internal_test.csv", "external_test.csv"
            })
            with (output / "train.csv").open(newline="") as handle:
                train_row = next(csv.DictReader(handle))
            self.assertEqual(train_row["Site_ID"], "P0_26")
            self.assertEqual(train_row["UniProt_ID"], "P0")
            self.assertEqual(train_row["ACC_ID"], "P0")
            self.assertEqual(train_row["Homology_group"], "G0")
            self.assertEqual(train_row["Sequence"][25], "K")

    def test_cross_partition_window_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = self.make_lock(root, duplicate_window=True)
            with self.assertRaisesRegex(ValueError, "Window crosses partitions"):
                export_splits(lock, root / "export")

    def test_modified_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = self.make_lock(root)
            with (root / "model_manifest.csv").open("a") as handle:
                handle.write("extra")
            with self.assertRaisesRegex(ValueError, "does not match"):
                export_splits(lock, root / "export")


if __name__ == "__main__":
    unittest.main()
