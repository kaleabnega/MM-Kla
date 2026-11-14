import unittest

import numpy as np
import torch

from pcbert_kla_clean.contextual_descriptor_development import (
    DescriptorPredictor, descriptor_features, fit_descriptors,
)
from pcbert_kla_clean.data import KlaRecord, KlaSplit
from pcbert_kla_clean.features import FEATURE_NAMES, cksaap_feature_names


class ContextualDescriptorTests(unittest.TestCase):
    def test_only_27_descriptors_enter_predictor(self):
        names = list(FEATURE_NAMES) + cksaap_feature_names()
        features = np.arange(2 * len(names), dtype=np.float32).reshape(2, -1)
        records = [KlaRecord("P1", 26, 1, "A" * 25 + "K" + "G" * 25),
                   KlaRecord("P2", 26, 0, "G" * 25 + "K" + "A" * 25)]
        split = KlaSplit(records, names, features)
        np.testing.assert_array_equal(descriptor_features(split), features[:, :27])
        with self.assertRaises(ValueError):
            descriptor_features(KlaSplit(records, names[27:] + names[:27], features))
        with self.assertRaises(ValueError):
            descriptor_features(KlaSplit(records, list(FEATURE_NAMES), features[:, :27]))

    def test_predictor_accepts_only_27_features_and_trains(self):
        model = DescriptorPredictor()
        logits = model(torch.randn(4, 27))
        self.assertEqual(tuple(logits.shape), (4,))
        logits.sum().backward()
        self.assertIsNotNone(model.network[0].weight.grad)
        with self.assertRaises(ValueError):
            model(torch.randn(4, 28))

    def test_fold_fit_selects_epoch_and_returns_finite_scores(self):
        random = np.random.default_rng(42)
        features = random.normal(size=(30, 27)).astype(np.float32)
        labels = np.tile([0, 1], 15)
        train = np.arange(20)
        held = np.arange(20, 30)
        epoch, ap = fit_descriptors(features, labels, train, held,
                                    torch.device("cpu"), 42, 2, 2,
                                    selection_idx=held)
        self.assertIn(epoch, (1, 2))
        self.assertTrue(0 <= ap <= 1)
        scores = fit_descriptors(features, labels, train, held,
                                 torch.device("cpu"), 42, 2, 2,
                                 fixed_epochs=epoch)
        self.assertEqual(scores.shape, (10,))
        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(((0 <= scores) & (scores <= 1)).all())


if __name__ == "__main__":
    unittest.main()
