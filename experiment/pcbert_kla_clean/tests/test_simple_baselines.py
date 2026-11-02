import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from pcbert_kla_clean.data import KlaRecord, KlaSplit
from pcbert_kla_clean.simple_baselines import feature_logistic_validation


class FeatureLogisticTests(unittest.TestCase):
    def setUp(self):
        labels = [0, 1, 0, 1, 0, 1, 0, 1]
        self.split = KlaSplit(
            records=[KlaRecord(str(i), 26, label, "A" * 25 + "K" + "A" * 25)
                     for i, label in enumerate(labels)],
            feature_names=["f1", "f2"],
            features=np.array([[0, 2], [1, 1], [0.1, 3], [1.1, 0],
                               [0.2, 4], [1.2, 2], [100, -100], [101, -101]], dtype=float),
        )
        self.validation = np.array([6, 7])

    def test_matches_train_only_pipeline(self):
        scores, metrics = feature_logistic_validation(self.split, self.validation)
        expected = make_pipeline(
            MinMaxScaler(), LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        )
        expected.fit(self.split.features[:6], self.split.labels[:6])
        np.testing.assert_allclose(scores, expected.predict_proba(self.split.features[6:])[:, 1])
        self.assertEqual(set(metrics), {"ACC", "AUC", "AUPRC", "F1", "MCC", "Pre", "Rec", "SP"})
        self.assertTrue(np.isfinite(scores).all())

    def test_validation_labels_do_not_affect_fit(self):
        scores, _ = feature_logistic_validation(self.split, self.validation)
        changed = KlaSplit(
            records=self.split.records[:6] + [
                KlaRecord("6", 26, 1, "A" * 25 + "K" + "A" * 25),
                KlaRecord("7", 26, 0, "A" * 25 + "K" + "A" * 25),
            ],
            feature_names=self.split.feature_names,
            features=self.split.features,
        )
        changed_scores, _ = feature_logistic_validation(changed, self.validation)
        np.testing.assert_array_equal(scores, changed_scores)

    def test_bad_indices_rejected(self):
        for indices in (np.array([]), np.array([6, 6]), np.array([8]), np.array([0, 1, 2, 3, 4, 5, 6, 7])):
            with self.subTest(indices=indices):
                with self.assertRaises(ValueError):
                    feature_logistic_validation(self.split, indices)


if __name__ == "__main__":
    unittest.main()
