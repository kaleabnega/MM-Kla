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

from run_hybridkla_architecture import grouped_folds, refit_full_stack, train_esm_fold
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
