import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from lock_kladb_human import assign_groups, connected_proteins, training_sample


def row(accession, label, pmid="", window=None):
    return {
        "accession": accession,
        "position_1based": "26" if label == 1 else "52",
        "window_51aa": window or ("A" * 25 + "K" + accession[-1] * 25),
        "label": str(label),
        "evidence": "reported_Kla_site" if label else "unreported_lysine_putative_negative",
        "source_pmids": pmid,
        "protein_source_pmids": pmid or "other",
    }


class HumanLockTests(unittest.TestCase):
    def test_homology_and_exact_windows_join_components(self):
        rows = [row("P1", 1), row("P2", 1), row("P3", 1, window="A" * 25 + "K" + "1" * 25)]
        with tempfile.TemporaryDirectory() as temporary:
            hits = Path(temporary) / "hits.tsv"
            with hits.open("w", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t")
                for accession in ("P1", "P2", "P3"):
                    writer.writerow([accession, accession, 100, 1, 1])
                writer.writerow(["P1", "P2", 31, 0.8, 0.8])
            groups, audit = connected_proteins(rows, hits)
        self.assertEqual(groups["P1"], groups["P2"])
        self.assertEqual(groups["P1"], groups["P3"])
        self.assertEqual(audit["self_hits"], 3)

    def test_external_study_is_exclusive_and_training_sample_does_not_touch_tests(self):
        rows = []
        for index in range(10):
            accession = f"P{index}"
            pmid = "held" if index == 0 else "other"
            rows.extend([
                row(accession, 1, pmid, window=f"{'A' * 24}{chr(65 + index)}K{'C' * 25}"),
                row(accession, 0, window=f"{'D' * 24}{chr(65 + index)}K{'E' * 25}"),
            ])
            rows[-1]["protein_source_pmids"] = pmid
        components = {row["accession"]: row["accession"] for row in rows}
        locked, audit = assign_groups(rows, components, "held")
        external = [record for record in locked if record["partition"] == "external_test"]
        self.assertEqual({record["accession"] for record in external}, {"P0"})
        self.assertEqual(len(external), 2)
        sampled = training_sample(locked, ratio=3, seed=42)
        self.assertEqual(
            [record for record in sampled if record["partition"] != "train"],
            [record for record in locked if record["partition"] != "train"],
        )
        self.assertGreater(audit["partitions_before_training_resampling"]["train"]["positives"], 0)
        pilot = training_sample(locked, ratio=3, seed=42, positive_cap=2)
        self.assertEqual(sum(r["partition"] == "train" and r["label"] == "1" for r in pilot), 2)
        self.assertEqual(
            [record for record in pilot if record["partition"] != "train"],
            [record for record in locked if record["partition"] != "train"],
        )

    def test_mixed_study_protein_is_excluded(self):
        rows = []
        for index in range(10):
            accession = f"P{index}"
            pmid = "held" if index == 1 else "other"
            rows.extend([
                row(accession, 1, pmid, window=f"{'A' * 24}{chr(65 + index)}K{'C' * 25}"),
                row(accession, 0, window=f"{'D' * 24}{chr(65 + index)}K{'E' * 25}"),
            ])
            rows[-1]["protein_source_pmids"] = pmid
            if index == 0:
                for record in rows[-2:]:
                    record["protein_source_pmids"] = "held;other"
        components = {record["accession"]: record["accession"] for record in rows}
        locked, audit = assign_groups(rows, components, "held")
        self.assertNotIn("P0", {record["accession"] for record in locked})
        self.assertEqual(audit["candidate_rows_removed_for_study_or_homology_overlap"]["mixed"], 2)


if __name__ == "__main__":
    unittest.main()
