import csv
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import run_hybridkla_adapted as runner
from pcbert_kla_clean.hybridkla_features import FEATURE_DIMS, GPS_AA, STANDARD_AA


class TinyFeature(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.linear = nn.Linear(FEATURE_DIMS[name], 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


class TinyLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(22, 4)
        self.linear = nn.Linear(4, 1)

    def forward(self, x):
        return self.linear(self.embedding(x).mean(dim=1)).squeeze(-1)


class TinyHead(nn.Module):
    def __init__(self, features=8):
        super().__init__()
        self.linear = nn.Linear(features, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


class HybridKlaRunnerFlowTests(unittest.TestCase):
    def make_resources(self, root):
        top10 = root / "top10.txt"
        with top10.open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["name", *STANDARD_AA])
            for index in range(10):
                writer.writerow([str(index), *range(20)])
        blosum = root / "BLOSUM62R.txt"
        with blosum.open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["intercept", *[GPS_AA[i] + GPS_AA[j]
                                            for i in range(24) for j in range(i, 24)]])
            writer.writerow(["1", *(["1"] * 300)])
        return {name: runner.sha256(root / name)
                for name in ("top10.txt", "BLOSUM62R.txt")}

    def make_lock(self, root):
        partitions = {"train": 20, "validation": 4, "internal_test": 4, "external_test": 4}
        rows = []
        counter = 0
        for partition, protein_count in partitions.items():
            for protein in range(protein_count):
                for site in range(2):
                    digits = []
                    value = counter
                    for _ in range(3):
                        digits.append(STANDARD_AA[value % 20])
                        value //= 20
                    sequence = "A" * 22 + "".join(digits) + "K" + "A" * 25
                    rows.append({"accession": f"{partition}_P{protein}",
                                 "position_1based": str(26 + site),
                                 "window_51aa": sequence, "label": str(site),
                                 "homology_group": f"{partition}_G{protein}",
                                 "partition": partition})
                    counter += 1
        manifest = root / "model_manifest.csv"
        with manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        audit = {
            "output_sha256": {"model_manifest.csv": runner.sha256(manifest)},
            "sampled_training_rows": 40,
            "partitions_before_training_resampling": {
                name: {"rows": count * 2, "positives": count}
                for name, count in partitions.items()
            },
        }
        lock = root / "split_audit.json"
        lock.write_text(json.dumps(audit))
        return lock

    def test_complete_runner_flow_without_large_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = self.make_lock(root)
            resource_dir = root / "resources"
            resource_dir.mkdir()
            hashes = self.make_resources(resource_dir)
            args = Namespace(human_lock=lock, resource_dir=resource_dir,
                             output_dir=root / "output", seed=42, device="cpu",
                             esm_device="cpu", esm_model="stub", esm_batch_size=4,
                             feature_batch_size=4, lstm_batch_size=4,
                             meta_fraction=0.2, epochs=1, meta_epochs=1, patience=0,
                             pilot=False)

            def fake_embeddings(partitions, model_name, revision, device, batch_size):
                arrays = {name: np.asarray([[row["Sequence"].count("A") / 51,
                                             row["Sequence"].count("C") / 51,
                                             0.5, 1.0] for row in rows], dtype=np.float32)
                          for name, rows in partitions.items()}
                return arrays, 4

            def fake_evaluation(name, rows, scores, output):
                self.assertEqual(len(rows), len(scores))
                self.assertTrue(np.isfinite(scores).all())
                return {"AUPRC": 0.1}

            fake_hf = SimpleNamespace(HfApi=lambda: SimpleNamespace(
                model_info=lambda name: SimpleNamespace(sha="a" * 40)))
            with patch.dict(sys.modules, {"huggingface_hub": fake_hf}), \
                 patch.object(runner, "RESOURCE_SHA256", hashes), \
                 patch.object(runner, "extract_frozen_esm_embeddings", fake_embeddings), \
                 patch.object(runner, "FeatureDNN", TinyFeature), \
                 patch.object(runner, "SequenceLSTM", TinyLSTM), \
                 patch.object(runner, "EsmEmbeddingHead", TinyHead), \
                 patch.object(runner, "MetaDNN", TinyHead), \
                 patch.object(runner, "evaluate_and_save", fake_evaluation):
                report = runner.run(args)
                args.pilot = True
                args.pilot_train_rows = 100
                args.pilot_validation_rows = 20
                args.output_dir = root / "pilot_output"
                original_read = runner.read_partition

                def development_only_read(path):
                    if path.stem in {"internal_test", "external_test"}:
                        raise AssertionError("Pilot read a locked test partition")
                    return original_read(path)

                with patch.object(runner, "read_partition", development_only_read):
                    pilot_report = runner.run(args)
            self.assertEqual(report["training_rows"], 40)
            self.assertEqual(set(report["selected_epochs"]), set(runner.BRANCHES) | {"META"})
            self.assertEqual(set(report["test_metrics"]), {"internal_test", "external_test"})
            self.assertTrue((root / "output" / "gps_reference.npy").is_file())
            self.assertTrue((root / "output" / "checkpoints" / "META.pth").is_file())
            self.assertTrue((root / "output" / "run_report.json").is_file())
            self.assertEqual(pilot_report["mode"], "training-only-pilot")
            self.assertEqual(pilot_report["test_metrics"], {})


if __name__ == "__main__":
    unittest.main()
