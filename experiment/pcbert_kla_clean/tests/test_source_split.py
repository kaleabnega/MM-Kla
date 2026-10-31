import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prepare_source_split import load_candidates, prepare_split, verify_reconstruction


class SourceSplitTests(unittest.TestCase):
    def test_bundled_source_manifest(self):
        directory = Path(__file__).resolve().parents[1] / "review_data" / "gao_2020_source_split"
        manifest = directory / "source_split_manifest.csv"
        audit = json.loads((directory / "source_split_audit.json").read_text())
        self.assertEqual(hashlib.sha256(manifest.read_bytes()).hexdigest(), audit["manifest_sha256"])
        with manifest.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 5153)
        by_protein = {}
        by_cluster = {}
        for row in rows:
            accession, cluster, partition = row["accession"], row["homology_cluster"], row["partition"]
            self.assertEqual(by_protein.setdefault(accession, partition), partition)
            self.assertEqual(by_cluster.setdefault(cluster, partition), partition)
        self.assertEqual(audit["homology_search"]["cross_partition_hits"], 0)
        self.assertEqual(sum(part["reported_positive_sites"] for part in audit["partitions"].values()), 208)

    def test_loader_rejects_mislabeled_control(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "candidates.csv"
            path.write_text(
                "accession,position_1based,window_51aa,label,evidence,source_doi\n"
                + "P0,26," + "A" * 25 + "K" + "C" * 25
                + ",0,reported_Kla_site,10.3389/fmicb.2020.594743\n"
            )
            with self.assertRaisesRegex(ValueError, "disagree"):
                load_candidates(path)

    def test_split_keeps_homologs_and_sites_together(self):
        with tempfile.TemporaryDirectory() as temporary:
            hits = Path(temporary) / "hits.tsv"
            clusters = {f"P{i}": f"P{i}" for i in range(6)}
            clusters["P1"] = "P0"
            lines = [f"{accession}\t{accession}\t100\t1\t1\n" for accession in clusters]
            lines.extend(["P0\tP1\t50\t1\t1\n", "P1\tP0\t50\t1\t1\n"])
            hits.write_text("".join(lines))
            rows = []
            for index, accession in enumerate(clusters):
                for label, evidence in ((1, "reported_Kla_site"), (0, "unannotated_lysine_putative_negative")):
                    rows.append({
                        "accession": accession,
                        "position_1based": str(index * 2 + label + 26),
                        "window_51aa": "A" * 24 + chr(65 + index) + "K" + "C" * 24 + ("D" if label else "E"),
                        "label": str(label),
                        "evidence": evidence,
                    })
            split_rows, report = prepare_split(rows, clusters, hits)
        by_accession = {row["accession"]: row["partition"] for row in split_rows}
        self.assertEqual(by_accession["P0"], by_accession["P1"])
        self.assertEqual(report["homology_search"]["cross_partition_hits"], 0)
        self.assertEqual(sum(part["sites"] for part in report["partitions"].values()), len(rows))

    def test_reconstruction_hash_must_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            candidate = base / "candidate.csv"
            audit = base / "audit.json"
            candidate.write_text("example")
            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            audit.write_text(json.dumps({"output_sha256": {"candidate_sites.csv": digest}, "input_sha256": {"gao": "sourcehash"}}))
            self.assertEqual(verify_reconstruction(candidate, audit), {"gao": "sourcehash"})
            candidate.write_text("changed")
            with self.assertRaisesRegex(ValueError, "does not match"):
                verify_reconstruction(candidate, audit)


if __name__ == "__main__":
    unittest.main()
