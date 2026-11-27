import gzip
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from pcbert_kla_clean.positional_global_development import (
    CHUNKS, GLOBAL_CHANNELS, GlobalPositionalSequenceBranch,
    load_verified_proteins, mark_site_chunk, protein_chunks,
)
from pcbert_kla_clean.stacked_development import PositionalSequenceBranch


class PositionalGlobalDevelopmentTests(unittest.TestCase):
    def test_chunk_features_cover_full_sequence_and_preserve_order(self):
        features = protein_chunks("A" * 8 + "R" * 8)
        self.assertEqual(features.shape, (CHUNKS, GLOBAL_CHANNELS))
        np.testing.assert_allclose(features[:, :-1].sum(axis=1), 1)
        self.assertEqual(features[0, 0], 1)
        self.assertEqual(features[-1, 14], 1)  # R in canonical alphabet
        self.assertFalse(np.array_equal(features, protein_chunks("R" * 8 + "A" * 8)))

    def test_unknown_residues_have_explicit_channel(self):
        features = protein_chunks("X" * 16)
        np.testing.assert_allclose(features[:, -2], 1)

    def test_site_marker_targets_the_correct_chunk(self):
        features = protein_chunks("A" * 80)
        marked = mark_site_chunk(features, 80, 21)
        self.assertEqual(int(np.argmax(marked[:, -1])), 2)
        self.assertEqual(marked[:, -1].sum(), 1)
        with self.assertRaises(ValueError):
            mark_site_chunk(features, 80, 81)

    def test_proteome_must_match_locked_site(self):
        record = SimpleNamespace(name="P1", position=26, sequence="A" * 25 + "K" + "A" * 25)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "proteins.fasta.gz"
            with gzip.open(path, "wt") as handle:
                handle.write(">sp|P1|protein\n" + record.sequence + "\n")
            self.assertEqual(load_verified_proteins(path, [record])["P1"], record.sequence)
            bad = SimpleNamespace(name="P1", position=26, sequence="A" * 51)
            with self.assertRaisesRegex(ValueError, "Reference sequence mismatch"):
                load_verified_proteins(path, [bad])

    def test_global_model_uses_protein_features_and_local_control_remains_unchanged(self):
        torch.manual_seed(3)
        model = GlobalPositionalSequenceBranch().eval()
        control = PositionalSequenceBranch().eval()
        tokens = torch.full((2, 51), 0, dtype=torch.long)
        tokens[:, 25] = 8  # K
        protein_a = torch.zeros(2, CHUNKS, GLOBAL_CHANNELS)
        protein_b = protein_a.clone()
        protein_a[:, :, 0] = 1
        protein_b[:, :, 1] = 1
        with torch.no_grad():
            scores_a = model(tokens, protein_a)
            scores_b = model(tokens, protein_b)
            control_scores = control(tokens)
        self.assertEqual(scores_a.shape, (2,))
        self.assertEqual(control_scores.shape, (2,))
        self.assertGreater((scores_a - scores_b).abs().max().item(), 0)


if __name__ == "__main__":
    unittest.main()
