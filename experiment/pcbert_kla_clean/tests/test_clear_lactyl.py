from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pcbert_kla_clean.features import calculate_cksaap_features, cksaap_feature_names


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from lock_clear_lactyl import split_groups  # noqa: E402
from prepare_clear_lactyl import prepare  # noqa: E402


class ClearLactylTests(unittest.TestCase):
    def test_cksaap_uses_ordered_pairs_at_four_fixed_spacings(self) -> None:
        sequence = "A" * 25 + "K" + "A" * 25
        features = calculate_cksaap_features([sequence, sequence])
        names = cksaap_feature_names()
        self.assertEqual(features.shape, (2, 1600))
        self.assertEqual(len(names), 1600)
        self.assertTrue(np.array_equal(features[0], features[1]))
        for gap in range(4):
            self.assertAlmostEqual(float(features[0, gap * 400:(gap + 1) * 400].sum()), 1)
        self.assertAlmostEqual(float(features[0, names.index("CKSAAP_k0_AA")]), 48 / 50)
        self.assertAlmostEqual(float(features[0, names.index("CKSAAP_k0_AK")]), 1 / 50)
        self.assertAlmostEqual(float(features[0, names.index("CKSAAP_k0_KA")]), 1 / 50)
        self.assertNotEqual(
            float(features[0, names.index("CKSAAP_k0_AA")]),
            float(features[0, names.index("CKSAAP_k1_AA")]),
        )
        with self.assertRaisesRegex(ValueError, "canonical"):
            calculate_cksaap_features(["X" + sequence[1:]])

    def test_preparation_filters_unverifiable_and_known_positive_negatives(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            source = base / "source.csv"
            fasta = base / "proteins.fasta"
            kladb = base / "kladb.csv"
            pos = "A" * 25 + "K" + "A" * 25
            neg = "C" * 25 + "K" + "C" * 25
            conflict = "D" * 25 + "K" + "D" * 25
            with source.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=(
                    "Protein accession", "Gene name", "Positions", "Sequence", "label"
                ))
                writer.writeheader()
                for accession, window, label in (
                    ("P1", pos, 1), ("P2", neg, 0), ("P3", conflict, 0),
                    ("P4", "<pad>" * 25 + "K" + "A" * 25, 1),
                ):
                    writer.writerow({"Protein accession": accession, "Gene name": "G",
                                     "Positions": 26, "Sequence": window, "label": label})
            fasta.write_text(f">P1\n{pos}\n>P2\n{neg}\n>P3\n{conflict}\n>P4\n{pos}\n")
            with kladb.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=(
                    "species", "uniprot_id", "position", "window_51", "pmid"
                ))
                writer.writeheader()
                writer.writerow({"species": "Homo sapiens", "uniprot_id": "P3",
                                 "position": 26, "window_51": conflict, "pmid": 123})
            report = prepare(source, fasta, kladb, base / "out")
            counts = report["counts"]
            self.assertEqual(counts["retained_positive"], 1)
            self.assertEqual(counts["retained_putative_negative"], 1)
            self.assertEqual(counts["negative_conflicts_with_KlaDB_positive"], 1)
            self.assertEqual(counts["padded_or_noncanonical_window"], 1)

    def test_grouped_split_keeps_both_classes_and_groups_disjoint(self) -> None:
        rows = [
            {"accession": f"P{group}", "position_1based": str(site + 1),
             "window_51aa": f"W{group}_{site}", "label": str(site)}
            for group in range(30) for site in (0, 1)
        ]
        components = {f"P{group}": f"G{group // 2}" for group in range(30)}
        locked, summary = split_groups(rows, components, seed=42)
        self.assertEqual(sum(part["rows"] for part in summary.values()), len(rows))
        self.assertTrue(all(part["positives"] > 0 and part["putative_negatives"] > 0
                            for part in summary.values()))
        partition_by_group = {}
        for row in locked:
            group = row["homology_group"]
            if group in partition_by_group:
                self.assertEqual(partition_by_group[group], row["partition"])
            partition_by_group[group] = row["partition"]


if __name__ == "__main__":
    unittest.main()
