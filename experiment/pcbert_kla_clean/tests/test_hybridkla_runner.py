import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_hybridkla_adapted import BRANCHES, split_training_groups


class HybridKlaRunnerTests(unittest.TestCase):
    def test_group_disjoint_meta_holdout_is_deterministic(self):
        rows = [
            {"ACC_ID": f"P{protein}", "Homology_group": f"G{protein // 2}",
             "Label": str(site % 2)}
            for protein in range(20) for site in range(10)
        ]
        base, meta = split_training_groups(rows, 0.2, 42)
        self.assertEqual((base, meta), split_training_groups(rows, 0.2, 42))
        self.assertEqual(set(base) | set(meta), set(range(len(rows))))
        self.assertFalse(set(base) & set(meta))
        self.assertFalse({rows[index]["ACC_ID"] for index in base}
                         & {rows[index]["ACC_ID"] for index in meta})
        self.assertFalse({rows[index]["Homology_group"] for index in base}
                         & {rows[index]["Homology_group"] for index in meta})

    def test_all_eight_branches_are_declared(self):
        self.assertEqual(BRANCHES, ("ESM2", "LSTM", "ACF", "AAINDEX",
                                    "CKSAAP", "GPS", "OBC", "PSEAAC"))


if __name__ == "__main__":
    unittest.main()
