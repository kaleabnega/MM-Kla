import gzip
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_source_site_evidence import (
    audit_records,
    compare_previous_mapping,
    source_site_window_summary,
)
from map_windows_to_proteome import WindowRecord, parse_fasta


class SourceSiteAuditTests(unittest.TestCase):
    def test_reads_gzipped_uniprot_fasta(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "proteome.fasta.gz"
            with gzip.open(path, "wt") as handle:
                handle.write(">sp|P12345|TEST Protein\nAAKAA\n")
            proteins = parse_fasta(path)
        self.assertEqual(len(proteins), 1)
        self.assertEqual(proteins[0].accession, "P12345")
        self.assertEqual(proteins[0].sequence, "AAKAA")

    def test_site_reconstruction_uses_one_based_central_lysine(self):
        sequence = "A" * 25 + "K" + "C" * 34
        window = sequence[:51]
        summary = source_site_window_summary(
            {("P12345", 26), ("P12345", 5)},
            {"P12345": sequence},
            {window},
            set(),
        )
        self.assertEqual(summary["reconstructed_sites"], 1)
        self.assertEqual(summary["site_not_a_lysine_at_reported_position"], 1)
        self.assertEqual(summary["windows_in_original_train"], 1)

    def test_exact_site_and_peptide_only_evidence_are_separate(self):
        sequence = "A" * 25 + "K" + "C" * 25
        windows = [
            WindowRecord("train", "Protein 0", 26, 1, sequence),
            WindowRecord("test", "Protein 0", 26, 0, sequence),
        ]
        matches = [
            {
                "split": record.split,
                "record_name": record.record_name,
                "accession": "NEW123",
                "central_lysine_full_position_1based": 26,
            }
            for record in windows
        ]
        source_rows = [{
            "Protein accession": "OLD123",
            "Position": "26",
            "Modified sequence": "AK(1)C",
        }]
        details, counts, _ = audit_records(windows, matches, {("OLD123", 26)}, source_rows)
        self.assertEqual(details[0]["evidence_status"], "peptide_match_only")
        self.assertEqual(details[0]["source_peptide_site_matches"], "OLD123:K26")
        self.assertEqual(counts["test_label_0_peptide_match_only"], 1)

        details, counts, _ = audit_records(windows, matches, {("NEW123", 26)}, source_rows)
        self.assertEqual(details[0]["evidence_status"], "one_source_site_match")
        self.assertEqual(counts["test_label_0_one_source_site_match"], 1)

    def test_previous_mapping_must_cover_same_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "previous.csv"
            path.write_text("original_split,record_name,candidate_sites\ntrain,Protein 0,P12345:K26\n")
            details = [{"split": "train", "record_name": "Protein 0", "candidate_sites": "P12345:K26"}]
            self.assertEqual(compare_previous_mapping(path, details)["changed_candidate_site_assignments"], 0)
            with self.assertRaisesRegex(ValueError, "does not cover"):
                compare_previous_mapping(path, [])


if __name__ == "__main__":
    unittest.main()
