"""Train the AttentionKla architecture on the locked CLEAR-Lactyl development split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from pcbert_kla_clean.attentionkla_development import run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=2)
    args = parser.parse_args()
    report = run(args.lock, args.output_dir, torch.device(args.device),
                 seed=args.seed, max_epochs=args.epochs, patience=args.patience)
    print(json.dumps({"validation_metrics": report["validation_metrics"],
                      "output_dir": str(args.output_dir)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
