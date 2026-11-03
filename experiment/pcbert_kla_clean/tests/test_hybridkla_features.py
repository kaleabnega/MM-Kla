import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pcbert_kla_clean.hybridkla_features import (
    FEATURE_DIMS, GPS_AA, STANDARD_AA, encode_features, fit_gps_reference, load_resources,
)


class HybridKlaFeatureTests(unittest.TestCase):
    def make_resources(self, root):
        top10 = root / "top10.txt"
        with top10.open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["name", *STANDARD_AA])
            for index in range(10):
                writer.writerow([str(index), *range(20)])
        blosum = root / "BLOSUM62R.txt"
        pairs = [GPS_AA[i] + GPS_AA[j] for i in range(24) for j in range(i, 24)]
        with blosum.open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["intercept", *pairs])
            writer.writerow(["1", *(["1"] * 300)])
        return load_resources(top10, blosum)

    def test_dimensions_and_composition(self):
        with tempfile.TemporaryDirectory() as temporary:
            resources = self.make_resources(Path(temporary))
            sequences = ["A" * 25 + "K" + "A" * 25,
                         "C" * 25 + "K" + "C" * 25]
            reference = fit_gps_reference(sequences[:1])
            encoded = encode_features(sequences, resources, reference, batch_size=1)
            self.assertEqual({name: values.shape for name, values in encoded.items()},
                             {name: (2, dim) for name, dim in FEATURE_DIMS.items()})
            self.assertEqual(encoded["CKSAAP"].sum(axis=1).tolist(), [50.0, 50.0])
            self.assertEqual(encoded["OBC"].sum(axis=1).tolist(), [51.0, 51.0])
            np.testing.assert_allclose(encoded["PSEAAC"].sum(axis=1), [1.0, 1.0])
            self.assertAlmostEqual(encoded["PSEAAC"][0, STANDARD_AA.index("A")], 50 / 51)
            self.assertTrue(all(np.isfinite(values).all() for values in encoded.values()))

    def test_gps_reference_uses_only_supplied_training_windows(self):
        with tempfile.TemporaryDirectory() as temporary:
            resources = self.make_resources(Path(temporary))
            training = ["A" * 25 + "K" + "A" * 25]
            query = "C" * 25 + "K" + "C" * 25
            reference = fit_gps_reference(training)
            before = encode_features([query], resources, reference)["GPS"]
            after = encode_features([query, training[0]], resources, reference)["GPS"][:1]
            np.testing.assert_array_equal(before, after)

    def test_invalid_window_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "K-centered"):
            fit_gps_reference(["A" * 51])


if __name__ == "__main__":
    unittest.main()
