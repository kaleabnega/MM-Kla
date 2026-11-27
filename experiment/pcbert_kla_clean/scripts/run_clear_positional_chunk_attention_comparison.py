"""Paired five-seed validation comparison of chunk-mean and chunk-attention models."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from pcbert_kla_clean.positional_chunk_attention import (  # noqa: E402
    ChunkAttentiveOrderContextPositionalBranch,
)
from pcbert_kla_clean.positional_order_context_development import (  # noqa: E402
    OrderContextPositionalBranch,
)
from pcbert_kla_clean.positional_order_context_single import run as run_single  # noqa: E402
from pcbert_kla_clean.source_study import sha256  # noqa: E402


SEEDS = (42, 123, 777, 2025, 3407)
METRICS = ("AUPRC", "AUC", "MCC", "F1", "Pre", "Rec", "SP", "ACC")
MODELS = (
    ("chunk_mean", OrderContextPositionalBranch),
    ("chunk_attention", ChunkAttentiveOrderContextPositionalBranch),
)


def summarize(reports: list[dict]) -> dict:
    summary = {}
    for name, _ in MODELS:
        rows = [report["validation_metrics"] for report in reports if report["name"] == name]
        if len(rows) != len(SEEDS):
            raise ValueError("Incomplete seed set")
        summary[name] = {
            metric: {"mean": float(np.mean([row[metric] for row in rows])),
                     "std": float(np.std([row[metric] for row in rows], ddof=1))}
            for metric in METRICS
        }
    return summary


def run(args) -> None:
    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        raise RuntimeError("A CUDA compute node is required")
    args.output_root.mkdir(parents=True, exist_ok=False)
    reports = []
    for seed in SEEDS:
        for name, model_factory in MODELS:
            print(f"Running {name} seed={seed}", flush=True)
            report = run_single(
                args.clear_lock, args.proteome,
                args.output_root / f"seed_{seed}" / name,
                torch.device(args.device), seed=seed, max_epochs=30, patience=5,
                model_factory=model_factory,
                model_name=f"single positional predictor with {name}",
            )
            reports.append({"name": name, **report})
            print(f"{name} seed={seed} validation={report['validation_metrics']}", flush=True)
    with (args.output_root / "individual_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("seed", "architecture", "selected_epoch", *METRICS))
        writer.writeheader()
        for report in reports:
            writer.writerow({"seed": report["seed"], "architecture": report["name"],
                             "selected_epoch": report["selected_epoch"],
                             **{metric: report["validation_metrics"][metric] for metric in METRICS}})
    summary = {
        "seeds": list(SEEDS), "lock_sha256": sha256(args.clear_lock),
        "proteome_sha256": sha256(args.proteome),
        "protocol": "paired independent single-model runs; grouped inner epoch selection; fixed 0.5 threshold",
        "heldout_test_scored": False, "metrics": summarize(reports),
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clear-lock", type=Path, required=True)
    parser.add_argument("--proteome", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
