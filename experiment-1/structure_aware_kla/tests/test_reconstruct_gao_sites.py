import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from reconstruct_gao_sites import reconstruct, site_window


class ReconstructGaoSitesTests(unittest.TestCase):
    def test_window_has_lysine_at_center(self):
        sequence = "A" * 25 + "K" + "C" * 26
        self.assertEqual(site_window(sequence, 26), sequence[:51])
        self.assertIsNone(site_window(sequence, 25))
        with self.assertRaisesRegex(ValueError, "not lysine"):
            site_window(sequence, 27)

    def test_reported_sites_and_putative_negatives_remain_distinct(self):
        sequence = "A" * 25 + "K" + "C" * 25 + "K" + "D" * 25
        source = [{"Protein accession": "P1", "Position": "26", "Amino acid": "K", "Modified sequence": "AAK(1)CC"}]
        evidence, candidates, _, counts = reconstruct(source, {"P1": sequence}, set())
        self.assertEqual(evidence[0]["reconstruction_status"], "window_available")
        self.assertEqual([row["label"] for row in candidates], [1, 0])
        self.assertEqual(candidates[1]["evidence"], "unannotated_lysine_putative_negative")
        self.assertEqual(counts["putative_negative_sites"], 1)

    def test_matching_benchmark_positive_cannot_be_negative(self):
        sequence = "A" * 25 + "K" + "C" * 25 + "K" + "D" * 25
        source = [{"Protein accession": "P1", "Position": "26", "Amino acid": "K", "Modified sequence": "AAK(1)CC"}]
        other_window = site_window(sequence, 52)
        _, candidates, excluded, counts = reconstruct(source, {"P1": sequence}, {other_window})
        self.assertEqual(len(candidates), 1)
        self.assertEqual(counts["unannotated_lysines_matching_known_positive_window"], 1)
        self.assertEqual(excluded[0]["reason"], "benchmark_or_source_positive_window_match")

    def test_matching_modified_peptide_cannot_be_negative(self):
        sequence = "A" * 25 + "K" + "C" * 25 + "K" + "D" * 25
        source = [{"Protein accession": "P1", "Position": "26", "Amino acid": "K", "Modified sequence": "CCK(1)DD"}]
        _, candidates, excluded, counts = reconstruct(source, {"P1": sequence}, set())
        self.assertEqual(len(candidates), 1)
        self.assertEqual(counts["unannotated_lysines_matching_modified_peptide"], 1)
        self.assertEqual(excluded[0]["reason"], "reported_modified_peptide_match")

    def test_missing_and_terminal_sites_are_retained_in_evidence_audit(self):
        source = [
            {"Protein accession": "P1", "Position": "1", "Amino acid": "K", "Modified sequence": "K(1)A"},
            {"Protein accession": "P2", "Position": "26", "Amino acid": "K", "Modified sequence": "AAK(1)AA"},
        ]
        evidence, candidates, _, _ = reconstruct(source, {"P1": "K" + "A" * 60}, set())
        self.assertEqual(len(evidence), 2)
        self.assertEqual(len(candidates), 0)
        self.assertEqual(evidence[0]["reconstruction_status"], "site_near_protein_terminus")
        self.assertEqual(evidence[1]["reconstruction_status"], "accession_missing_from_proteome")


if __name__ == "__main__":
    unittest.main()
