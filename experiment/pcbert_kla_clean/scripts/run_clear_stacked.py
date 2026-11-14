"""Validation-only CLEAR-Lactyl two-branch experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from pcbert_kla_clean.stacked_development import run  # noqa: E402
import torch  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clear-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    result = run(args.clear_lock, args.output_dir, torch.device(args.device),
                 args.seed, args.epochs, args.patience)
    print("CLEAR-Lactyl stacked validation:")
    print(json.dumps(result["validation_metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
