from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_cross_study_homology import read_hits


class CrossStudyHomologyTests(unittest.TestCase):
    def test_merges_both_directions_without_losing_reverse_only_hits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            forward = Path(directory) / "forward.tsv"
            reverse = Path(directory) / "reverse.tsv"
            forward.write_text("G1\tR1\t50.0\t0.8\t0.9\t80\t100\t90\n")
            reverse.write_text(
                "R1\tG1\t50.0\t0.9\t0.8\t80\t90\t100\n"
                "R2\tG1\t34.0\t0.5\t0.6\t50\t100\t90\n"
            )
            result = read_hits(forward, reverse, {"G1"}, {"R1", "R2"}, 30.0, 0.4)
            self.assertEqual(set(result), {("G1", "R1"), ("G1", "R2")})
            self.assertTrue(result[("G1", "R1")]["forward_seen"])
            self.assertTrue(result[("G1", "R1")]["reverse_seen"])
            self.assertFalse(result[("G1", "R2")]["forward_seen"])

    def test_rejects_alignment_below_claimed_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            forward = Path(directory) / "forward.tsv"
            reverse = Path(directory) / "reverse.tsv"
            forward.write_text("G1\tR1\t50.0\t0.3\t0.9\t30\t100\t90\n")
            reverse.write_text("")
            with self.assertRaisesRegex(ValueError, "fails stated thresholds"):
                read_hits(forward, reverse, {"G1"}, {"R1"}, 30.0, 0.4)


if __name__ == "__main__":
    unittest.main()
