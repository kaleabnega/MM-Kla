from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_transkla_clear import load_lock, write_inputs  # noqa: E402


class TransKlaClearAdapterTests(unittest.TestCase):
    def test_locked_windows_supply_exact_centered_41_residues(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protein = "A" * 30 + "K" + "C" * 30
            window = protein[5:56]
            manifest = root / "model_manifest.csv"
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=(
                    "accession", "position_1based", "window_51aa", "label",
                    "homology_group", "partition",
                ))
                writer.writeheader()
                for part, accession, group in (
                    ("train", "P1", "G1"),
                    ("validation", "P2", "G2"),
                    ("internal_test", "P3", "G3"),
                ):
                    writer.writerow({
                        "accession": accession, "position_1based": 31,
                        "window_51aa": window if accession == "P1" else
                        window.replace("A", "G" if accession == "P2" else "T"),
                        "label": 1, "homology_group": group, "partition": part,
                    })
            proteins = {
                "P1": protein,
                "P2": protein.replace("A", "G"),
                "P3": protein.replace("A", "T"),
            }
            proteome = root / "proteome.fasta.gz"
            with gzip.open(proteome, "wt") as handle:
                for accession, sequence in proteins.items():
                    handle.write(f">sp|{accession}|protein\n{sequence}\n")
            audit = {
                "protocol": "CLEAR-Lactyl protein-grouped test",
                "output_sha256": {"model_manifest.csv": hashlib.sha256(manifest.read_bytes()).hexdigest()},
                "partitions": {name: {"rows": 1, "positives": 1} for name in
                               ("train", "validation", "internal_test")},
            }
            lock = root / "split_audit.json"
            lock.write_text(json.dumps(audit))
            partitions, loaded_proteins, _ = load_lock(lock, proteome)
            output = root / "develop"
            output.mkdir()
            data_dir = write_inputs(output, partitions, loaded_proteins,
                                    ("train", "validation"))
            train = (data_dir / "train.fasta").read_text().splitlines()
            self.assertEqual(train[0], ">P1|pos=31|1")
            self.assertEqual(train[1], window[5:46])
            self.assertEqual(len(train[1]), 41)
            self.assertEqual(train[1][20], "K")
            self.assertFalse((data_dir / "internal_test.fasta").exists())
            self.assertNotIn("P3", (data_dir / "proteins.csv").read_text())
            with (data_dir / "proteins.csv").open(newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)


if __name__ == "__main__":
    unittest.main()
