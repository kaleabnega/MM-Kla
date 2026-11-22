import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from pcbert_kla_clean.features import FEATURE_NAMES
from pcbert_kla_clean.data import KlaRecord, KlaSplit
from pcbert_kla_clean.source_study import sha256
from pcbert_kla_clean.stacked_development import (
    PositionalSequenceBranch, feature_columns, read_branch_reference,
    run, sequence_tokens,
)


class ExtendedStackAblationTests(unittest.TestCase):
    def test_positional_controls_change_only_the_named_module(self):
        tokens = sequence_tokens(["A" * 25 + "K" + "G" * 25] * 2)
        for mode in ("none", "no_position_embedding", "one_lstm_layer", "mean_pool"):
            with self.subTest(mode=mode):
                model = PositionalSequenceBranch(ablation=mode)
                self.assertIsNotNone(model.position)
                self.assertEqual(model.encoder.num_layers, 1 if mode == "one_lstm_layer" else 2)
                self.assertIsNotNone(model.attention)
                output = model(tokens)
                self.assertEqual(tuple(output.shape), (2,))
                self.assertTrue(torch.isfinite(output).all())
                output.sum().backward()
                if mode == "no_position_embedding":
                    self.assertIsNone(model.position.weight.grad)
                if mode == "mean_pool":
                    self.assertIsNone(model.attention.weight.grad)
        with self.assertRaises(ValueError):
            PositionalSequenceBranch(ablation="unknown")

    def test_feature_family_controls_keep_the_other_family(self):
        names = list(FEATURE_NAMES) + [f"cksaap_{i}" for i in range(1600)]
        np.testing.assert_array_equal(feature_columns(names, "none"), np.arange(1627))
        np.testing.assert_array_equal(feature_columns(names, "no_descriptors"), np.arange(27, 1627))
        np.testing.assert_array_equal(feature_columns(names, "no_cksaap"), np.arange(27))
        with self.assertRaises(ValueError):
            feature_columns(names[:-1], "none")

    def test_protbert_reference_checks_seed_and_site_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock = root / "split_audit.json"
            lock.write_text("locked")
            reference = root / "reference"
            reference.mkdir()
            split = SimpleNamespace(
                features=np.zeros((3, 1627)), labels=np.array([0, 1, 1]),
                records=[SimpleNamespace(name="P1", position=1, label=0),
                         SimpleNamespace(name="P2", position=2, label=1),
                         SimpleNamespace(name="P3", position=3, label=1)],
            )
            groups = np.array(["g1", "g2"])
            report = {
                "seed": 123, "lock_sha256": sha256(lock), "training_rows": 2,
                "validation_rows": 1, "feature_dim": 1627,
                "internal_test_scored": False, "protbert_ablation": "none",
                "freeze_protbert": False, "folds": [{} for _ in range(5)],
            }
            (reference / "report.json").write_text(json.dumps(report))
            np.savez_compressed(reference / "oof_branch_scores.npz",
                                scores=np.array([[0.2, 0.3], [0.7, 0.8]]),
                                labels=np.array([0, 1]), groups=groups)
            with (reference / "validation_predictions.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["accession", "position_1based", "label", "protbert_cksaap"])
                writer.writerow(["P3", 3, 1, 0.65])
            oof, validation, _ = read_branch_reference(
                reference, lock, split, 2, np.array([2]), groups, 123, "protbert",
            )
            np.testing.assert_allclose(oof, [0.2, 0.7])
            np.testing.assert_allclose(validation, [0.65])
            with self.assertRaises(ValueError):
                read_branch_reference(reference, lock, split, 2, np.array([2]),
                                      groups, 42, "protbert")
            report["feature_ablation"] = "no_cksaap"
            (reference / "report.json").write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                read_branch_reference(reference, lock, split, 2, np.array([2]),
                                      groups, 123, "protbert")

    def test_component_ablations_require_opposite_branch_reference(self):
        common = dict(lock_path=Path("missing"), output_dir=Path("missing-output"),
                      device=torch.device("cpu"), seed=42)
        with self.assertRaises(ValueError):
            run(**common, positional_ablation="mean_pool")
        with self.assertRaises(ValueError):
            run(**common, feature_ablation="no_cksaap")
        with self.assertRaises(ValueError):
            run(**common, positional_reference_dir=Path("p"),
                protbert_reference_dir=Path("q"))
        with self.assertRaises(ValueError):
            run(**common, positional_ablation="none", score_internal_test=True,
                reference_report=Path("missing"), protbert_reference_dir=Path("p"))

    def test_both_extended_paths_reuse_only_the_opposite_branch(self):
        records = [KlaRecord(f"P{i}", 26, i % 2, "A" * 25 + "K" + "G" * 25)
                   for i in range(24)]
        split = KlaSplit(records, list(FEATURE_NAMES) + [f"c{i}" for i in range(1600)],
                         np.zeros((24, 1627), dtype=np.float32))
        empty = KlaSplit([], split.feature_names, np.empty((0, 1627), dtype=np.float32))
        groups = np.asarray([f"g{i // 2}" for i in range(20)])
        base_oof = np.asarray([0.2 if i % 2 == 0 else 0.8 for i in range(20)])
        base_val = np.asarray([0.2, 0.8, 0.2, 0.8])
        reference = {"folds": [{"protbert_selected_epoch": 2,
                                "protbert_inner_auprc": 0.7,
                                "positional_selected_epoch": 2,
                                "positional_inner_auprc": 0.7} for _ in range(5)]}

        def fake_fit(split, train_idx, score_idx, *args, selection_idx=None, **kwargs):
            if selection_idx is not None:
                return 2, 0.7
            return np.where(split.labels[score_idx] == 1, 0.8, 0.2)

        def fake_pos_fit(tokens, labels, train_idx, score_idx, *args,
                         selection_idx=None, **kwargs):
            if selection_idx is not None:
                return 2, 0.7
            return np.where(labels[score_idx] == 1, 0.8, 0.2)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock = root / "split_audit.json"
            lock.write_text("locked")
            with (patch("pcbert_kla_clean.stacked_development.load_clear_lactyl",
                        return_value=(split, np.arange(20, 24), empty)),
                  patch("pcbert_kla_clean.stacked_development.read_training_groups",
                        return_value=groups),
                  patch("pcbert_kla_clean.stacked_development.read_branch_reference",
                        return_value=(base_oof, base_val, reference)),
                  patch("pcbert_kla_clean.stacked_development.read_positional_reference",
                        return_value=(base_oof, base_val, reference)),
                  patch("pcbert_kla_clean.stacked_development.load_tokenizer",
                        return_value=object()),
                  patch("pcbert_kla_clean.stacked_development.fit_protbert",
                        side_effect=fake_fit) as prot_fit,
                  patch("pcbert_kla_clean.stacked_development.fit_positional",
                        side_effect=fake_pos_fit) as pos_fit):
                positional = run(lock, root / "positional", torch.device("cpu"), 42,
                                 protbert_reference_dir=root / "reference",
                                 positional_ablation="mean_pool")
                self.assertEqual(prot_fit.call_count, 0)
                self.assertEqual(pos_fit.call_count, 10)
                self.assertFalse(positional["internal_test_scored"])
                self.assertEqual(positional["positional_ablation"], "mean_pool")
                self.assertEqual(positional["used_feature_dim"], 1627)
                pos_fit.reset_mock()
                features = run(lock, root / "features", torch.device("cpu"), 42,
                               positional_reference_dir=root / "reference",
                               feature_ablation="no_cksaap")
                self.assertEqual(pos_fit.call_count, 0)
                self.assertEqual(prot_fit.call_count, 10)
                self.assertEqual(features["used_feature_dim"], 27)
                self.assertTrue(all(call.kwargs["feature_ablation"] == "no_cksaap"
                                    for call in prot_fit.call_args_list))


if __name__ == "__main__":
    unittest.main()
