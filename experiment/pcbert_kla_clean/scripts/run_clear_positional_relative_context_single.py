"""Evaluate one site-relative-context positional model on CLEAR-Lactyl validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import torch  # noqa: E402
from pcbert_kla_clean.positional_order_context_single import run  # noqa: E402
from pcbert_kla_clean.positional_relative_context import RelativeContextPositionalBranch  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clear-lock", type=Path, required=True)
    parser.add_argument("--proteome", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    report = run(args.clear_lock, args.proteome, args.output_dir,
                 torch.device(args.device), args.seed, args.epochs, args.patience,
                 model_factory=RelativeContextPositionalBranch,
                 model_name="single site-relative-context positional predictor")
    print(json.dumps({"validation_metrics": report["validation_metrics"],
                      "selected_epoch": report["selected_epoch"],
                      "ensemble_size": report["ensemble_size"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
