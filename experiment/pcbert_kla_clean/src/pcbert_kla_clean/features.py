from __future__ import annotations

import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
CKSAAP_GAPS = (0, 1, 2, 3)
FEATURE_NAMES = [
    "Molecular Weight Calculation",
    "Isoelectric Point Calculation",
    *AMINO_ACIDS,
    "Helix",
    "Turn",
    "Sheet",
    "Hydrophobicity Calculation",
    "Net Charge Calculation",
]


def cksaap_feature_names() -> list[str]:
    return [
        f"CKSAAP_k{gap}_{first}{second}"
        for gap in CKSAAP_GAPS
        for first in AMINO_ACIDS
        for second in AMINO_ACIDS
    ]


def calculate_cksaap_features(sequences: list[str]) -> np.ndarray:
    """Normalized ordered residue-pair frequencies for gaps 0 through 3."""
    if not sequences or any(len(sequence) != 51 or sequence[25] != "K" for sequence in sequences):
        raise ValueError("CKSAAP requires nonempty 51-residue K-centered windows")
    lookup = np.full(256, -1, dtype=np.int16)
    for index, residue in enumerate(AMINO_ACIDS):
        lookup[ord(residue)] = index
    encoded = np.frombuffer("".join(sequences).encode("ascii"), dtype=np.uint8).reshape(-1, 51)
    residue_ids = lookup[encoded]
    if np.any(residue_ids < 0):
        raise ValueError("CKSAAP requires canonical amino acids")

    row_offsets = np.arange(len(sequences), dtype=np.int64)[:, None] * 400
    features = []
    for gap in CKSAAP_GAPS:
        pair_ids = residue_ids[:, :50 - gap] * 20 + residue_ids[:, gap + 1:]
        counts = np.bincount(
            (pair_ids + row_offsets).ravel(), minlength=len(sequences) * 400
        ).reshape(len(sequences), 400)
        features.append(counts.astype(np.float32) / (50 - gap))
    return np.concatenate(features, axis=1)


def calculate_features(sequence: str) -> np.ndarray:
    analysis = ProteinAnalysis(sequence)
    if hasattr(analysis, "get_amino_acids_percent"):
        composition = analysis.get_amino_acids_percent()
    else:
        composition = {
            residue: percent / 100.0
            for residue, percent in analysis.amino_acids_percent.items()
        }
    return np.asarray(
        [
            analysis.molecular_weight(),
            analysis.isoelectric_point(),
            *(composition[residue] for residue in AMINO_ACIDS),
            *analysis.secondary_structure_fraction(),
            analysis.gravy(),
            analysis.charge_at_pH(7.0),
        ],
        dtype=np.float32,
    )
