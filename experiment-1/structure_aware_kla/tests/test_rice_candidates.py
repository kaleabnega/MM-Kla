from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_rice_candidates import candidate_rows


class RiceCandidateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sequence = (
            "A" * 25 + "K" + "C" * 26 + "K" + "D" * 26 + "K"
            + "E" * 26 + "K" + "F" * 26 + "K" + "G" * 26 + "K" + "H" * 30
        )
        self.positions = [index + 1 for index, residue in enumerate(self.sequence) if residue == "K"]
        self.windows = {
            position: self.sequence[position - 26 : position + 25]
            for position in self.positions
        }
        self.positive = [{
            "accession": "P12345", "position_1based": str(self.positions[0]),
            "window_51aa": self.windows[self.positions[0]], "label": "1",
            "source_pmid": "34264677",
        }]

    def test_exclusions_preserve_only_unreported_unscreened_sites(self) -> None:
        positions = self.positions
        result, counts = candidate_rows(
            {"P12345": self.sequence},
            self.positive,
            {("P12345", positions[0]), ("P12345", positions[1])},
            {("P12345", positions[2])},
            {self.windows[positions[3]]},
            {self.windows[positions[4]]},
        )
        self.assertEqual([row["position_1based"] for row in result], [str(positions[0]), str(positions[5])])
        self.assertEqual(counts["excluded_modified_peptide_match"], 1)
        self.assertEqual(counts["excluded_known_positive_window"], 1)
        self.assertEqual(counts["excluded_development_window"], 1)
        self.assertEqual(counts["putative_negative_sites"], 1)

    def test_positive_overlapping_development_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive window occurs in development"):
            candidate_rows(
                {"P12345": self.sequence}, self.positive, set(), set(), set(),
                {self.windows[self.positions[0]]},
            )


if __name__ == "__main__":
    unittest.main()
