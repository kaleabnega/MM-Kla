from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_candidate_studies import audit_rice, study_summary


class CandidateStudyAuditTests(unittest.TestCase):
    def test_rice_requires_primary_site_and_exact_full_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proteome = root / "rice.fasta"
            primary = root / "primary.csv"
            sequence = list("A" * 60)
            sequence[1] = "K"
            sequence[25] = "K"
            sequence = "".join(sequence)
            proteome.write_text(f">sp|P12345|TEST Rice protein\n{sequence}\n")
            with primary.open("w", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=["Protein accession", "Position", "Amino acid"])
                writer.writeheader()
                writer.writerows([
                    {"Protein accession": "P12345", "Position": "26", "Amino acid": "K"},
                    {"Protein accession": "P12345", "Position": "2", "Amino acid": "K"},
                    {"Protein accession": "P12345", "Position": "30", "Amino acid": "K"},
                    {"Protein accession": "MISSING", "Position": "26", "Amino acid": "K"},
                    {"Protein accession": "P12345", "Position": "40", "Amino acid": "A"},
                ])
            curated = [{"uniprot_id": "P12345", "position": "26", "window_51": sequence[:51]}]
            report, verified = audit_rice(primary, curated, proteome)
            self.assertEqual(len(verified), 1)
            self.assertEqual(verified[0]["window_51aa"], sequence[:51])
            self.assertEqual(report["site_status_counts"], {
                "accession_absent_from_proteome": 1,
                "primary_and_curated_window_verified": 1,
                "reported_position_not_lysine": 1,
                "site_too_close_to_terminus": 1,
                "source_row_not_lysine": 1,
            })

    def test_curated_window_mismatch_is_not_exported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proteome = root / "rice.fasta"
            primary = root / "primary.csv"
            sequence = "A" * 25 + "K" + "A" * 34
            proteome.write_text(f">sp|P12345|TEST\n{sequence}\n")
            primary.write_text("Protein accession,Position,Amino acid\nP12345,26,K\n")
            curated = [{"uniprot_id": "P12345", "position": "26", "window_51": "K" * 51}]
            report, verified = audit_rice(primary, curated, proteome)
            self.assertEqual(verified, [])
            self.assertEqual(report["site_status_counts"]["curated_window_mismatch"], 1)

    def test_repeated_windows_are_counted_separately_from_sites(self) -> None:
        window = "A" * 25 + "K" + "A" * 25
        rows = [
            {"uniprot_id": "P1", "position": "26", "window_51": window},
            {"uniprot_id": "P2", "position": "26", "window_51": window},
        ]
        result = study_summary(rows)
        self.assertEqual(result["distinct_accession_position_sites"], 2)
        self.assertEqual(result["repeated_unpadded_window_rows"], 1)


if __name__ == "__main__":
    unittest.main()
