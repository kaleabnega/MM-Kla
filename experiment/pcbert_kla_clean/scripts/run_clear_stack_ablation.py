"""Validation-only, grouped-OOF MM-Kla component ablation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import torch  # noqa: E402
from pcbert_kla_clean.stacked_development import run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clear-lock", type=Path, required=True)
    parser.add_argument("--positional-reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ablation", required=True, choices=(
        "none", "cls_pool", "mean_pool", "max_pool", "concat_fusion",
        "scalar_gate", "no_gated_fusion", "no_residual", "frozen_encoder",
    ))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    report = run(
        lock_path=args.clear_lock, output_dir=args.output_dir,
        device=torch.device(args.device), seed=args.seed,
        max_epochs=args.epochs, patience=args.patience,
        protbert_ablation="none" if args.ablation == "frozen_encoder" else args.ablation,
        freeze_protbert=args.ablation == "frozen_encoder",
        positional_reference_dir=args.positional_reference_dir,
    )
    print("CLEAR-Lactyl stacked component validation:", flush=True)
    print(json.dumps(report["validation_metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
