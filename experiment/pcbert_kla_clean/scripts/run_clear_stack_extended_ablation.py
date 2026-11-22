"""Validation-only ablations of the positional branch or feature inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import torch  # noqa: E402
from pcbert_kla_clean.stacked_development import run  # noqa: E402


MODES = {
    "no_position_embedding": ("positional", "no_position_embedding"),
    "one_lstm_layer": ("positional", "one_lstm_layer"),
    "positional_mean_pool": ("positional", "mean_pool"),
    "no_descriptors": ("features", "no_descriptors"),
    "no_cksaap": ("features", "no_cksaap"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clear-lock", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ablation", required=True, choices=tuple(MODES))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    branch, mode = MODES[args.ablation]
    report = run(
        lock_path=args.clear_lock, output_dir=args.output_dir,
        device=torch.device(args.device), seed=args.seed,
        max_epochs=args.epochs, patience=args.patience,
        protbert_reference_dir=args.reference_dir if branch == "positional" else None,
        positional_reference_dir=args.reference_dir if branch == "features" else None,
        positional_ablation=mode if branch == "positional" else "none",
        feature_ablation=mode if branch == "features" else "none",
    )
    print("CLEAR-Lactyl stacked extended ablation validation:", flush=True)
    print(json.dumps(report["validation_metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
