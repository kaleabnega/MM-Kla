import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_hybridkla_architecture import grouped_folds, read_lock, refit_full_stack, train_esm_fold
import run_hybridkla_architecture as runner
from pcbert_kla_clean.hybridkla_models import fit_predictor


class DummyTokenizer:
    all_special_ids = [0]

    def __call__(self, sequences, **kwargs):
        ids = torch.tensor([[0, 1 if sequence[0] == "A" else 2, 1]
                            for sequence in sequences])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class DummyEsmBranch(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.embedding = nn.Embedding(3, 1)

    def forward(self, ids, mask, special):
        return self.embedding(ids[:, 1]).squeeze(-1)


class HybridKlaArchitectureTests(unittest.TestCase):
    def make_human_lock(self, root):
        rows = []
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        for partition, groups in (("train", 10), ("validation", 2),
                                  ("internal_test", 2), ("external_test", 2)):
            for group in range(groups):
                for label in (0, 1):
                    site = len(rows)
                    triplet = "".join(amino_acids[(site // (20 ** power)) % 20]
                                      for power in (0, 1, 2))
                    sequence = "A" * 22 + triplet + "K" + "A" * 25
                    rows.append({
                        "accession": f"{partition}_P{group}",
                        "position_1based": str(26 + label),
                        "window_51aa": sequence, "label": str(label),
                        "source_pmids": ("41192556" if partition == "external_test" and label
                                         else "12345" if label else ""),
                        "homology_group": f"{partition}_G{group}",
                        "partition": partition,
                    })
        manifest = root / "model_manifest.csv"
        with manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        audit = {
            "holdout_pmid": "41192556",
            "sampled_training_rows": 20,
            "output_sha256": {"model_manifest.csv": runner.sha256(manifest)},
            "partitions_before_training_resampling": {
                name: {"rows": count * 2, "positives": count}
                for name, count in (("train", 10), ("validation", 2),
                                    ("internal_test", 2), ("external_test", 2))
            },
        }
        lock = root / "split_audit.json"
        lock.write_text(json.dumps(audit))
        return lock

    def test_human_lock_keeps_grouped_partitions(self):
        with tempfile.TemporaryDirectory() as temporary:
            lock = self.make_human_lock(Path(temporary))
            partitions, _ = read_lock(lock)
            self.assertEqual({name: len(rows) for name, rows in partitions.items()},
                             {"train": 20, "validation": 4,
                              "internal_test": 4, "external_test": 4})
            self.assertEqual(partitions["train"][0]["group_id"], "train_G0")
            self.assertEqual(partitions["external_test"][1]["source_site_id"],
                             "external_test_P0:27")

    def test_human_development_run_never_scores_tests(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = self.make_human_lock(root)
            resources = root / "resources"
            resources.mkdir()
            for name in ("top10.txt", "BLOSUM62R.txt"):
                (resources / name).write_text("stub")
            hashes = {name: runner.sha256(resources / name)
                      for name in ("top10.txt", "BLOSUM62R.txt")}
            head = lambda: nn.Sequential(nn.Linear(1, 1), nn.Flatten(0))
            args = SimpleNamespace(
                device="cpu", resource_dir=resources, original_external_csv=None,
                benchmark_lock=lock, pilot=False, development_only=True,
                full_refit=False, esm_model="stub", seed=42, output_dir=root / "output",
                esm_epochs=1, lstm_epochs=1, feature_epochs=1, meta_epochs=1,
                esm_batch_size=4, lstm_batch_size=4, feature_batch_size=4,
                patience=0,
            )
            fake_hf = SimpleNamespace(HfApi=lambda: SimpleNamespace(
                model_info=lambda name: SimpleNamespace(sha="a" * 40)))
            fake_transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: object()))
            features = lambda _name, items, _resources, _gps: np.asarray(
                [[item["label"]] for item in items], dtype=np.float32)
            with patch.dict(sys.modules, {"huggingface_hub": fake_hf,
                                          "transformers": fake_transformers}), \
                 patch.object(runner, "RESOURCE_SHA256", hashes), \
                 patch.object(runner, "BRANCHES", ("LSTM",)), \
                 patch.object(runner, "load_resources", return_value=None), \
                 patch.object(runner, "fit_gps_reference", return_value=np.zeros(1)), \
                 patch.object(runner, "branch_arrays", features), \
                 patch.object(runner, "SequenceLSTM", head), \
                 patch.object(runner, "MetaDNN", head):
                report = runner.run(args)
            self.assertEqual(report["mode"], "development-only")
            self.assertEqual(report["test_metrics"], {})
            self.assertIn("AUPRC", report["validation_metrics"])
            self.assertTrue((args.output_dir / "validation_predictions.csv").is_file())
            self.assertFalse((args.output_dir / "internal_test_predictions.csv").exists())
            self.assertFalse((args.output_dir / "external_test_predictions.csv").exists())

    def test_grouped_folds_partition_training_rows(self):
        rows = [{"group_id": f"P{group}", "label": str(site % 2)}
                for group in range(20) for site in range(4)]
        first = grouped_folds(rows, 5, 42)
        self.assertEqual(first, grouped_folds(rows, 5, 42))
        self.assertEqual(sorted(index for fold in first for index in fold),
                         list(range(len(rows))))
        self.assertEqual(len({row["group_id"] for row in rows}),
                         sum(len({rows[index]["group_id"] for index in fold})
                             for fold in first))
        self.assertTrue(all({rows[index]["label"] for index in fold} == {"0", "1"}
                            for fold in first))

    def test_esm_training_selects_checkpoint_without_test_rows(self):
        tokenizer = DummyTokenizer()
        make_row = lambda letter, label: {"window_51aa": letter * 25 + "K" + letter * 25,
                                           "label": str(label)}
        train = [make_row("A", 1), make_row("C", 0)] * 4
        validation = [make_row("A", 1), make_row("C", 0)]
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "esm.pth"
            with patch("run_hybridkla_architecture.TrainableEsmBranch", DummyEsmBranch):
                result = train_esm_fold(
                    "unused", "unused", tokenizer, train, validation,
                    device=torch.device("cpu"), epochs=2, batch_size=2, patience=0,
                    seed=42, checkpoint=checkpoint,
                )
            self.assertTrue(checkpoint.is_file())
            self.assertIn(result["report"]["selected_epoch"], (1, 2))
            self.assertEqual(result["report"]["completed_epochs"], 2)

    def test_fixed_epoch_refits_do_not_require_validation(self):
        tokenizer = DummyTokenizer()
        make_row = lambda letter, label: {"window_51aa": letter * 25 + "K" + letter * 25,
                                           "label": str(label)}
        train = [make_row("A", 1), make_row("C", 0)] * 4
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "esm.pth"
            with patch("run_hybridkla_architecture.TrainableEsmBranch", DummyEsmBranch):
                result = train_esm_fold(
                    "unused", "unused", tokenizer, train, None,
                    device=torch.device("cpu"), epochs=2, batch_size=2, patience=0,
                    seed=42, checkpoint=checkpoint, fixed_epochs=True,
                )
            self.assertTrue(checkpoint.is_file())
            self.assertEqual(result["report"]["selected_epoch"], 2)
            self.assertIsNone(result["report"]["validation_auprc"])
        fitted = fit_predictor(
            nn.Sequential(nn.Linear(1, 1), nn.Flatten(0)),
            np.arange(8, dtype=np.float32).reshape(-1, 1),
            np.array([0, 1] * 4), None, None, device=torch.device("cpu"),
            epochs=2, batch_size=2, learning_rate=1e-3, patience=0, seed=42,
            fixed_epochs=True,
        )
        self.assertEqual(fitted.selected_epoch, 2)
        self.assertIsNone(fitted.validation_auprc)

    def test_full_stack_uses_every_source_row_for_oof_meta_training(self):
        rows = [{"group_id": f"P{group}", "label": str(label), "site": group * 2 + label,
                 "window_51aa": "A" * 25 + "K" + "A" * 25}
                for group in range(20) for label in (0, 1)]
        selected = {f"LSTM_fold_{fold}": {"selected_epoch": 1} for fold in range(5)}
        selected["META"] = {"selected_epoch": 2}
        head = lambda: nn.Sequential(nn.Linear(1, 1), nn.Flatten(0))
        features = lambda _name, items, _resources, _gps: np.asarray(
            [[item["site"]] for item in items], dtype=np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch("run_hybridkla_architecture.BRANCHES", ("LSTM",)), \
                 patch("run_hybridkla_architecture.SequenceLSTM", head), \
                 patch("run_hybridkla_architecture.MetaDNN", head), \
                 patch("run_hybridkla_architecture.branch_arrays", features), \
                 patch("run_hybridkla_architecture.fit_gps_reference", return_value=np.zeros(1)):
                reports, folds, meta = refit_full_stack(
                    SimpleNamespace(seed=42, lstm_batch_size=4, feature_batch_size=4),
                    rows, selected, None, None, "unused", torch.device("cpu"), root, root,
                )
            self.assertEqual(sorted(index for fold in folds for index in fold), list(range(40)))
            self.assertEqual(reports["META"]["trained_epochs"], 2)
            self.assertTrue((root / "META.pth").is_file())
            self.assertIsNotNone(meta)


if __name__ == "__main__":
    unittest.main()
