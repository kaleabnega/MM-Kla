from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from pcbert_kla_clean.candidate_groups import prepare_candidate_group_partitions
from pcbert_kla_clean.data import KlaRecord, KlaSplit, load_split
from prepare_homology_clusters import convert_cluster_tsv, export_candidate_fasta


class CandidateGroupTests(unittest.TestCase):
    def make_split(self, rows: list[tuple[str, int]]) -> KlaSplit:
        records = [KlaRecord(f"Protein {index}", 1, label, window) for index, (window, label) in enumerate(rows)]
        features = np.asarray([[float(len(window))] for window, _ in rows], dtype=np.float32).reshape(-1, 1)
        return KlaSplit(records, ["length"], features)

    def write_csv(self, path: Path, columns: list[str], rows: list[tuple[str, ...]]) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(rows)

    def test_ambiguous_candidates_and_homology_stay_together(self) -> None:
        rows = [(f"K{'A' * index}C", 0) for index in range(3, 27)] + [
            (f"K{'G' * index}T", 1) for index in range(3, 27)
        ]
        train = self.make_split(rows)
        test = self.make_split([])
        with tempfile.TemporaryDirectory() as directory:
            mapping = Path(directory) / "mapping.csv"
            clusters = Path(directory) / "clusters.csv"
            accessions = [f"P{index:03}" for index in range(len(rows))]
            mapping_rows = [(window, accessions[index]) for index, (window, _) in enumerate(rows)]
            mapping_rows.extend([(rows[0][0], accessions[1]), (rows[0][0], accessions[1])])
            self.write_csv(mapping, ["window_sequence", "accession"], mapping_rows)
            cluster_rows = [(accession, accession) for accession in accessions]
            cluster_rows[2] = (accessions[2], "linked")
            cluster_rows[3] = (accessions[3], "linked")
            self.write_csv(clusters, ["accession", "cluster_id"], cluster_rows)

            parts = prepare_candidate_group_partitions(
                train, test, mapping_csv=mapping, homology_csv=clusters, seed=11
            )
            inventory = {row["sequence"]: row for row in parts.inventory}
            self.assertEqual(inventory[rows[0][0]]["assignment"], inventory[rows[1][0]]["assignment"])
            self.assertEqual(inventory[rows[2][0]]["assignment"], inventory[rows[3][0]]["assignment"])
            self.assertEqual(set(parts.report["candidate_protein_overlap"].values()), {0})
            self.assertEqual(sum(len(getattr(parts, name).records) for name in ("train", "validation", "test")), len(rows))
            again = prepare_candidate_group_partitions(train, test, mapping_csv=mapping, homology_csv=clusters, seed=11)
            self.assertEqual(parts.test.sequences, again.test.sequences)

    def test_mapping_must_cover_every_window(self) -> None:
        train = self.make_split([(f"K{'A' * index}C", 0) for index in range(3, 10)] + [
            (f"K{'G' * index}T", 1) for index in range(3, 10)
        ])
        with tempfile.TemporaryDirectory() as directory:
            mapping = Path(directory) / "mapping.csv"
            self.write_csv(mapping, ["window_sequence", "accession"], [(train.sequences[0], "P1")])
            with self.assertRaisesRegex(ValueError, "Mapping coverage mismatch"):
                prepare_candidate_group_partitions(train, self.make_split([]), mapping_csv=mapping)

    def test_excluded_conflict_still_bridges_proteins(self) -> None:
        rows = [(f"K{'A' * index}C", 0) for index in range(3, 27)] + [
            (f"K{'G' * index}T", 1) for index in range(3, 27)
        ]
        conflict = "KCCCCCCCCCC"
        train = self.make_split(rows + [(conflict, 1)])
        test = self.make_split([(conflict, 0)])
        with tempfile.TemporaryDirectory() as directory:
            mapping = Path(directory) / "mapping.csv"
            mapping_rows = [(window, f"P{index:03}") for index, (window, _) in enumerate(rows)]
            mapping_rows.extend([(conflict, "P000"), (conflict, "P001")])
            self.write_csv(mapping, ["window_sequence", "accession"], mapping_rows)
            parts = prepare_candidate_group_partitions(train, test, mapping_csv=mapping)
            inventory = {row["sequence"]: row for row in parts.inventory}
            self.assertEqual(inventory[conflict]["assignment"], "excluded_conflicting_labels")
            self.assertEqual(inventory[rows[0][0]]["assignment"], inventory[rows[1][0]]["assignment"])

    def test_bundled_mapping_has_no_candidate_protein_overlap(self) -> None:
        root = PROJECT_DIR.parents[1]
        data = root / "baselines/PCBert-Kla-original/data"
        mapping = root / "experiment-1/structure_aware_kla/results/botrytis_b0510_window_mapping/all_window_matches.csv"
        train = load_split(data / "train.csv", data / "feature_train.csv")
        test = load_split(data / "test.csv", data / "feature_test.csv")
        parts = prepare_candidate_group_partitions(train, test, mapping_csv=mapping)
        self.assertEqual(set(parts.report["candidate_protein_overlap"].values()), {0})
        self.assertEqual(sum(len(getattr(parts, name).records) for name in ("train", "validation", "test")), 1932)
        self.assertIn("homology not screened", parts.report["protocol"])

    def test_cluster_prep_checks_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            mapping = folder / "mapping.csv"
            fasta = folder / "proteins.fasta"
            extracted = folder / "candidates.fasta"
            tsv = folder / "clusters.tsv"
            output = folder / "clusters.csv"
            self.write_csv(mapping, ["accession"], [("P1",), ("P2",)])
            fasta.write_text(">sp|P1|one\nAKK\n>sp|P2|two\nMKT\n>sp|P3|other\nAAA\n")
            self.assertEqual(export_candidate_fasta(mapping, fasta, extracted), 2)
            self.assertNotIn("P3", extracted.read_text())
            tsv.write_text("P1\tP1\nP1\tP2\n")
            self.assertEqual(convert_cluster_tsv(mapping, tsv, output), 2)
            self.assertIn("P2,P1", output.read_text())
            tsv.write_text("P1\tP1\n")
            with self.assertRaisesRegex(ValueError, "missing"):
                convert_cluster_tsv(mapping, tsv, output)


if __name__ == "__main__":
    unittest.main()
