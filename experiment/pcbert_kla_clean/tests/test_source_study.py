import json
import sys
import tempfile
import unittest
from argparse import Namespace
from unittest.mock import patch
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from pcbert_kla_clean.data import load_split
from pcbert_kla_clean.data import KlaRecord, KlaSplit
from pcbert_kla_clean.experiments import train_and_predict_independent
from pcbert_kla_clean.features import FEATURE_NAMES, calculate_features
from pcbert_kla_clean.source_study import (
    load_locked_partitions,
    lock_external_manifest,
    read_rows,
    sha256,
    validate_rows,
)


class SourceStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gao = PROJECT_DIR / "review_data/gao_2020_source_split/source_split_manifest.csv"
        cls.rice = PROJECT_DIR / "review_data/source_study_audit_2026_11_01/rice_homology_screened_candidates.csv"
        cls.lock = PROJECT_DIR / "review_data/locked_source_study_v1/protocol_lock.json"

    def test_locked_partitions_and_labels(self):
        development, validation_idx, internal, external = load_locked_partitions(
            self.gao, self.rice, self.lock
        )
        self.assertEqual((len(development.records) - len(validation_idx), len(validation_idx), len(internal.records), len(external.records)), (3599, 519, 1035, 4773))
        training_idx = np.setdiff1d(np.arange(len(development.records)), validation_idx)
        self.assertEqual((int(development.labels[training_idx].sum()), len(training_idx) - int(development.labels[training_idx].sum())), (145, 3454))
        self.assertEqual(int(external.labels.sum()), 409)
        self.assertEqual(development.feature_names, FEATURE_NAMES)
        self.assertEqual(development.features.shape[1], 27)
        self.assertFalse(set(development.sequences) & set(internal.sequences))
        self.assertFalse(set(development.sequences + internal.sequences) & set(external.sequences))

    def test_feature_vector_matches_original_csv(self):
        baseline = PROJECT_DIR.parents[1] / "baselines/PCBert-Kla-original/data"
        original = load_split(baseline / "train.csv", baseline / "feature_train.csv")
        self.assertEqual(original.feature_names, FEATURE_NAMES)
        for index in (0, 91, 1000, len(original.records) - 1):
            np.testing.assert_allclose(
                calculate_features(original.records[index].sequence),
                original.features[index], rtol=1e-5, atol=1e-3,
            )

    def test_duplicate_policy_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            report = lock_external_manifest(self.gao, self.rice, output)
            self.assertEqual(report["rice_repeated_window_rows_removed"], 20)
            self.assertEqual(report["external_manifest_sha256"], sha256(self.lock.parent / "external_test_manifest.csv"))
            self.assertEqual(report["external_manifest_sha256"], sha256(output / "external_test_manifest.csv"))

    def test_changed_lock_input_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            changed = Path(temporary) / "protocol_lock.json"
            data = json.loads(self.lock.read_text())
            data["external_manifest_sha256"] = "0" * 64
            changed.write_text(json.dumps(data))
            (changed.parent / "external_test_manifest.csv").write_bytes(
                (self.lock.parent / "external_test_manifest.csv").read_bytes()
            )
            with self.assertRaisesRegex(ValueError, "Locked source file changed"):
                load_locked_partitions(self.gao, self.rice, changed)

    def test_invalid_central_residue_rejected(self):
        rows = read_rows(self.lock.parent / "external_test_manifest.csv")
        changed = dict(rows[0])
        changed["window_51aa"] = changed["window_51aa"][:25] + "A" + changed["window_51aa"][26:]
        with self.assertRaisesRegex(ValueError, "Invalid 51-aa lysine window"):
            validate_rows([changed], "test")

    def test_internal_and_external_use_same_selected_model(self):
        import torch

        def split(prefix):
            return KlaSplit(
                records=[
                    KlaRecord(f"{prefix}0", 26, 0, "A" * 25 + "K" + "C" * 25),
                    KlaRecord(f"{prefix}1", 26, 1, "C" * 25 + "K" + "A" * 25),
                ],
                feature_names=["feature"],
                features=np.array([[0.0], [1.0]], dtype=np.float32),
            )

        model = torch.nn.Linear(1, 1)
        seen_models = []

        def fake_predict(current_model, loader, criterion, device):
            seen_models.append(current_model)
            return 0.1, np.array([0, 1]), np.array([0.1, 0.9])

        args = Namespace(
            device="cpu", sequence_format="raw", batch_size=2, max_length=64,
            epochs=1, patience=0, decision_threshold=0.5, monitor_metric="AUPRC",
            calibrate_threshold="none", save_predictions=True, save_models=False,
            architecture="token_gated",
        )
        development, internal, external = split("train"), split("internal"), split("external")
        with tempfile.TemporaryDirectory() as temporary, \
             patch("pcbert_kla_clean.experiments.make_loader", side_effect=lambda records, indices, *rest: (records, indices)), \
             patch("pcbert_kla_clean.experiments.build_model", return_value=model), \
             patch("pcbert_kla_clean.experiments.build_optimizer", return_value=object()), \
             patch("pcbert_kla_clean.experiments.build_scheduler", return_value=None), \
             patch("pcbert_kla_clean.experiments.train_one_epoch", return_value=0.1), \
             patch("pcbert_kla_clean.experiments.evaluate", return_value=(0.1, {"ACC": 1.0, "AUPRC": 1.0, "MCC": 1.0, "AUC": 1.0})), \
             patch("pcbert_kla_clean.experiments.predict", side_effect=fake_predict):
            result = train_and_predict_independent(
                args, development, internal, seed=42, tokenizer=object(),
                device=torch.device("cpu"), output_dir=Path(temporary),
                fixed_validation_indices=np.array([1]),
                extra_test_splits={"rice_external": external},
            )
            self.assertEqual(len(seen_models), 2)
            self.assertTrue(all(seen is model for seen in seen_models))
            self.assertIn("rice_external", result["extra_metrics"])
            self.assertTrue((Path(temporary) / "rice_external_predictions.csv").exists())
            self.assertEqual(json.loads((Path(temporary) / "run_config.json").read_text())["extra_test_rows"], {"rice_external": 2})


if __name__ == "__main__":
    unittest.main()
