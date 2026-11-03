import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from lock_hybridkla_benchmark import assign_protein_groups, clean_records, lock_benchmark, sha256


class HybridKlaSourceLockTests(unittest.TestCase):
    @staticmethod
    def records(proteins=20):
        rows = []
        for protein in range(proteins):
            for site in range(4):
                prefix = "A" * 22 + "ACDE"[site] + "A" * 2
                suffix = ("ACDEFGHIKLMNPQRSTVWY"[protein % 20] * 24) + "A"
                rows.append({"UniProt_ID": f"P{protein}_{26 + site}",
                             "Sequence": prefix + "K" + suffix,
                             "Label": str(site % 2)})
        return rows

    def test_noncanonical_exclusion_and_conflicts(self):
        rows = self.records()
        rows[0]["Sequence"] = rows[0]["Sequence"][:-1] + "*"
        eligible, audit = clean_records(rows)
        self.assertEqual(audit["excluded"], {"noncanonical_label_0": 1})
        self.assertEqual(len(eligible), 79)
        conflicting = self.records()
        conflicting.append({**conflicting[0], "Label": "1"})
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            clean_records(conflicting)

    def test_deterministic_protein_disjoint_split(self):
        eligible, _ = clean_records(self.records())
        locked, audit = assign_protein_groups(eligible, 42)
        self.assertEqual((locked, audit), assign_protein_groups(eligible, 42))
        self.assertEqual(len(locked), 80)
        partitions = {}
        for row in locked:
            self.assertEqual(partitions.setdefault(row["accession"], row["partition"]),
                             row["partition"])
        self.assertEqual(set(audit["partitions"]), {"train", "validation", "internal_test"})

    def test_lock_hashes_and_refuses_changed_workbook(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workbook = root / "train_data.xlsx"
            workbook.write_bytes(b"fixture")
            with patch("lock_hybridkla_benchmark.read_workbook_rows", return_value=self.records()):
                report = lock_benchmark(workbook, root / "out", expected_sha256=sha256(workbook))
            self.assertEqual(report["benchmark_status"],
                             "provisional internal benchmark; source accessions and study provenance unverified")
            with (root / "out" / "model_manifest.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 80)
            self.assertEqual(json.loads((root / "out" / "split_audit.json").read_text()), report)
            with self.assertRaisesRegex(ValueError, "differs"):
                lock_benchmark(workbook, root / "other")


if __name__ == "__main__":
    unittest.main()
