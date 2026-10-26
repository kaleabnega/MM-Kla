import csv
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_mapped_sites import audit_mapping


class MappedSiteAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.train = self.root / "train.csv"
        self.test = self.root / "test.csv"
        self.mapping = self.root / "mapping.csv"
        self.train.write_text("data\nProtein 0|3|1\nAAKAA\nProtein 1|3|0\nCCKCC\n")
        self.test.write_text("data\nProtein 0|3|0\nGGKGG\n")
        self.rows = [
            ("train", "Protein 0", 1, "AAKAA", "A", 20),
            ("train", "Protein 1", 0, "CCKCC", "B", 30),
            ("train", "Protein 1", 0, "CCKCC", "C", 40),
            ("test", "Protein 0", 0, "GGKGG", "A", 40),
        ]

    def write_mapping(self):
        with self.mapping.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "split",
                    "record_name",
                    "label",
                    "window_position",
                    "window_sequence",
                    "match_status",
                    "accession",
                    "central_lysine_full_position_1based",
                ],
            )
            writer.writeheader()
            for split, name, label, sequence, accession, position in self.rows:
                writer.writerow(
                    {
                        "split": split,
                        "record_name": name,
                        "label": label,
                        "window_position": 3,
                        "window_sequence": sequence,
                        "match_status": "multiple_matches" if sequence == "CCKCC" else "unique_match",
                        "accession": accession,
                        "central_lysine_full_position_1based": position,
                    }
                )

    def test_reports_shared_protein_for_distinct_test_window(self):
        self.write_mapping()
        report, manifest = audit_mapping(self.mapping, self.train, self.test)
        self.assertEqual(report["benchmark_rows"], 3)
        self.assertEqual(report["match_status_rows"], {"unique_site": 2, "ambiguous_site": 1})
        self.assertEqual(report["accessions_in_both_original_splits_from_unique_mappings"], 1)
        self.assertEqual(report["test_rows_without_exact_train_window_on_shared_unique_accessions"], 1)
        self.assertEqual(manifest[1]["candidate_sites"], "B:K30;C:K40")

    def test_rejects_incomplete_mapping(self):
        self.rows.pop()
        self.write_mapping()
        with self.assertRaisesRegex(ValueError, "does not cover every"):
            audit_mapping(self.mapping, self.train, self.test)


if __name__ == "__main__":
    unittest.main()
