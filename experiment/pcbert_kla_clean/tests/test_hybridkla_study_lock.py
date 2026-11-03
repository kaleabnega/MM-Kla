import ast
import json
import sys
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

from lock_hybridkla_study import assign_partitions


class HybridKlaStudyLockTests(unittest.TestCase):
    def test_mixed_homology_group_is_excluded_and_partitions_are_disjoint(self):
        rows = []
        components = {}
        for number in range(5):
            protein = f"9606:P{number}"
            components[protein] = protein
            for label in ("0", "1"):
                rows.append({
                    "protein_id": protein, "accession": f"P{number}",
                    "species": "Homo sapiens", "position_1based": str(26 + int(label)),
                    "window_51aa": ("A" * 24 + ("C" if label == "0" else "D")
                                    + "K" + ("F", "G", "H", "I", "L")[number] * 25),
                    "label": label, "evidence": "test", "source_table": "S1",
                    "partition_eligibility": "development",
                })
        for number in range(2):
            protein = f"10090:M{number}"
            components[protein] = "9606:P0" if number == 0 else protein
            for label in ("0", "1"):
                rows.append({
                    "protein_id": protein, "accession": f"M{number}",
                    "species": "Mus musculus", "position_1based": str(26 + int(label)),
                    "window_51aa": ("G" * 24 + ("L" if label == "0" else "M")
                                    + "K" + chr(80 + number) * 25),
                    "label": label, "evidence": "test", "source_table": "S7",
                    "partition_eligibility": "external_candidate",
                })
        locked, report = assign_partitions(rows, components, seed=42)
        self.assertEqual(report["homology_overlap_excluded"]["groups"], 1)
        self.assertFalse(any(row["protein_id"] in {"9606:P0", "10090:M0"} for row in locked))
        self.assertEqual({row["partition"] for row in locked},
                         {"train", "validation", "internal_test", "external_test"})
        self.assertEqual(len({row["window_51aa"] for row in locked}), len(locked))
        self.assertEqual(locked, assign_partitions(rows, components, seed=42)[0])
        for name, counts in report["partitions_before_training_resampling"].items():
            self.assertGreater(counts["positives"], 0, name)
            self.assertGreater(counts["putative_negatives"], 0, name)

    def test_notebook_code_is_valid_python(self):
        notebook = json.loads((PROJECT / "kaggle_hybridkla_study.ipynb").read_text())
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))


if __name__ == "__main__":
    unittest.main()
