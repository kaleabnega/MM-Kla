import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_hybridkla_architecture import grouped_folds, train_esm_fold


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


if __name__ == "__main__":
    unittest.main()
