from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

import Bio
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from pcbert_kla_clean.data import load_split


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
        dtype=np.float64,
    )


def audit_split(sequence_file: Path, feature_file: Path) -> dict[str, object]:
    split = load_split(sequence_file, feature_file)
    if split.feature_names != FEATURE_NAMES:
        raise ValueError("Feature columns do not match the expected 27-column order")
    recomputed = np.stack([calculate_features(record.sequence) for record in split.records])
    provided = split.features.astype(np.float64)
    absolute_error = np.abs(recomputed - provided)
    close = np.isclose(recomputed, provided, rtol=1e-5, atol=1e-3)
    return {
        "rows": len(split.records),
        "features": len(FEATURE_NAMES),
        "mismatched_cells": int((~close).sum()),
        "mismatched_rows": int((~close).any(axis=1).sum()),
        "maximum_absolute_error_by_feature": {
            name: float(absolute_error[:, index].max())
            for index, name in enumerate(FEATURE_NAMES)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute PCBert-Kla feature CSVs")
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=PROJECT_DIR.parents[1] / "baselines" / "PCBert-Kla-original",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "outputs" / "review_exact_window" / "feature_audit.json",
    )
    args = parser.parse_args()
    data_root = args.baseline_root / "data"
    report = {
        "biopython_version_used_for_audit": Bio.__version__,
        "reference_implementation": "PCBert-Kla.ipynb: analyze_protein_sequences",
        "comparison_tolerance": {"relative": 1e-5, "absolute": 1e-3},
        "train": audit_split(data_root / "train.csv", data_root / "feature_train.csv"),
        "test": audit_split(data_root / "test.csv", data_root / "feature_test.csv"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
