from __future__ import annotations

import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
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
