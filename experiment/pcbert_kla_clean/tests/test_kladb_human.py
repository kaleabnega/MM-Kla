from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_kladb_human.py"
SPEC = importlib.util.spec_from_file_location("prepare_kladb_human", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def evidence(accession: str, position: int, window: str, pmid: str) -> dict[str, str]:
    return {
        "species": "Homo sapiens", "uniprot_id": accession,
        "position": str(position), "window_51": window, "pmid": pmid,
    }


class HumanPreparationTests(unittest.TestCase):
    def test_window_at_pads_terminals(self) -> None:
        sequence = "AAK" + "A" * 50
        self.assertEqual(MODULE.window_at(sequence, 3), "*" * 23 + sequence[:28])

    def test_evidence_dedup_and_negative_exclusion(self) -> None:
        sequence = "A" * 25 + "K" + "A" * 39 + "C" + "K" + "A" * 25
        positive = MODULE.window_at(sequence, 26)
        rows, report, accessions = MODULE.prepare_candidates(
            [evidence("P1", 26, positive, "100"), evidence("P1", 26, positive, "200")],
            {"P1": sequence},
        )
        self.assertEqual(accessions, {"P1"})
        self.assertEqual(report["evidence_rows"], 2)
        self.assertEqual(report["unique_reported_sites"], 1)
        self.assertEqual(report["usable_positive_sites"], 1)
        self.assertEqual(report["putative_negative_sites"], 1)
        self.assertEqual([row["label"] for row in rows], [1, 0])
        self.assertEqual(rows[0]["source_pmids"], "100;200")

    def test_unusable_positive_is_not_relabelled_negative(self) -> None:
        sequence = "A" * 24 + "K" + "A" * 50
        rows, report, accessions = MODULE.prepare_candidates(
            [evidence("P1", 25, MODULE.window_at(sequence, 25), "100")],
            {"P1": sequence},
        )
        self.assertEqual(rows, [])
        self.assertEqual(accessions, set())
        self.assertEqual(report["exclusions"]["terminal_or_noncanonical_window"], 1)

    def test_sequence_disagreement_fails_closed(self) -> None:
        sequence = "A" * 25 + "K" + "A" * 25
        with self.assertRaisesRegex(ValueError, "disagrees"):
            MODULE.prepare_candidates(
                [evidence("P1", 26, "C" + sequence[1:], "100")], {"P1": sequence}
            )

    def test_positive_window_with_missing_accession_excludes_matching_negative(self) -> None:
        sequence = "A" * 25 + "K" + "A" * 39 + "C" + "K" + "A" * 25
        missing_window = MODULE.window_at(sequence, 26)
        usable_window = MODULE.window_at(sequence, 67)
        rows, report, _ = MODULE.prepare_candidates(
            [evidence("MISSING", 26, missing_window, "100"),
             evidence("P1", 67, usable_window, "200")],
            {"P1": sequence},
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(report["exclusions"]["accession_not_in_proteome"], 1)
        self.assertEqual(report["exclusions"]["negative_matches_positive_window"], 1)


if __name__ == "__main__":
    unittest.main()
