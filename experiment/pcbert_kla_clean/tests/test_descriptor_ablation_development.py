import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pcbert_kla_clean.data import KlaRecord, KlaSplit
from pcbert_kla_clean.descriptor_ablation_development import (
    load_contextual_reference, without_descriptors,
)
from pcbert_kla_clean.features import FEATURE_NAMES, cksaap_feature_names
from pcbert_kla_clean.source_study import sha256


class DescriptorAblationTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            KlaRecord("P1", 26, 1, "A" * 25 + "K" + "G" * 25),
            KlaRecord("P2", 26, 0, "G" * 25 + "K" + "A" * 25),
            KlaRecord("P3", 26, 1, "C" * 25 + "K" + "D" * 25),
        ]
        names = list(FEATURE_NAMES) + cksaap_feature_names()
        self.features = np.arange(3 * len(names), dtype=np.float32).reshape(3, -1)
        self.split = KlaSplit(self.records, names, self.features)

    def test_only_27_descriptors_are_removed(self):
        ablated = without_descriptors(self.split)
        self.assertEqual(ablated.feature_names, cksaap_feature_names())
        self.assertEqual(ablated.features.shape, (3, 1600))
        np.testing.assert_array_equal(ablated.features, self.features[:, 27:])
        np.testing.assert_array_equal(ablated.labels, self.split.labels)
        with self.assertRaises(ValueError):
            without_descriptors(KlaSplit(self.records, list(reversed(self.split.feature_names)),
                                         self.features))

    def test_contextual_reference_requires_matching_lock_rows_and_groups(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            lock = directory / "split_audit.json"
            lock.write_text("locked data")
            reference = directory / "reference"
            reference.mkdir()
            report = {
                "seed": 42, "lock_sha256": sha256(lock), "training_rows": 2,
                "validation_rows": 1, "model_name": "Rostlab/prot_bert",
                "encoder_layers": 30, "encoder_frozen": True,
                "feature_dim": self.features.shape[1], "internal_test_scored": False,
            }
            (reference / "report.json").write_text(json.dumps(report))
            np.savez_compressed(reference / "oof_contextual_scores.npz",
                                scores=np.array([0.2, 0.8]), labels=np.array([1, 0]),
                                groups=np.array(["a", "b"]))
            with (reference / "validation_predictions.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["accession", "position_1based", "label",
                                 "contextual_positional", "contextual_two_predictor"])
                writer.writerow(["P3", 26, 1, 0.7, 0.6])
            args = (reference, lock, 42, self.split, np.array([2]), np.array(["a", "b"]))
            oof, validation, full_stack, hashes = load_contextual_reference(*args)
            np.testing.assert_array_equal(oof, [0.2, 0.8])
            np.testing.assert_array_equal(validation, [0.7])
            np.testing.assert_array_equal(full_stack, [0.6])
            self.assertIn("oof_sha256", hashes)
            with self.assertRaises(ValueError):
                load_contextual_reference(reference, lock, 123, *args[3:])
            with self.assertRaises(ValueError):
                load_contextual_reference(reference, lock, 42, self.split,
                                          np.array([2]), np.array(["a", "wrong"]))


if __name__ == "__main__":
    unittest.main()
