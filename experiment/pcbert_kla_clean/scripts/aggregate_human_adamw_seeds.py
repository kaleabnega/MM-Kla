"""Combine validation-only seed bundles from separate Kaggle runs."""

import argparse
import json
import math
from pathlib import Path

import pandas as pd


SEEDS = {42, 123, 777}
ARCHITECTURES = {"baseline", "token_gated", "hybridkla_architecture_aligned"}
METRICS = ("AUPRC", "AUC", "MCC", "F1", "Pre", "Rec", "SP", "ACC")


def aggregate(paths, model_architecture="token_gated"):
    if model_architecture not in {"token_gated", "token_gated_revised"}:
        raise ValueError("Unsupported MM-Kla architecture")
    architectures = {"baseline", model_architecture, "hybridkla_architecture_aligned"}
    if len(paths) != len(SEEDS):
        raise ValueError(f"Expected exactly {len(SEEDS)} seed bundles")
    bundles = [json.loads(Path(path).read_text()) for path in paths]
    if {bundle.get("seed") for bundle in bundles} != SEEDS:
        raise ValueError("Bundles must contain exactly seeds 42, 123, and 777")

    reference = bundles[0]
    for bundle in bundles:
        if bundle.get("schema_version") != 1:
            raise ValueError("Unsupported seed bundle schema")
        for key in ("git_commit", "audit_sha256", "manifest_sha256", "settings"):
            if not bundle.get(key) or bundle[key] != reference.get(key):
                raise ValueError(f"Seed bundles have missing or inconsistent {key}")
        records = bundle.get("records")
        if not isinstance(records, list) or len(records) != len(architectures):
            raise ValueError(f"Seed {bundle['seed']} needs three model records")
        if {record.get("architecture") for record in records} != architectures:
            raise ValueError(f"Seed {bundle['seed']} has missing or duplicate architectures")
        for record in records:
            if record.get("seed") != bundle["seed"]:
                raise ValueError("Record seed differs from bundle seed")
            if not isinstance(record.get("selected_epoch"), int) or record["selected_epoch"] < 1:
                raise ValueError("Invalid selected epoch")
            for metric in METRICS:
                value = record.get(metric)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"Invalid {metric} in seed {bundle['seed']}")
                if metric == "MCC":
                    valid = -1 <= value <= 1
                else:
                    valid = 0 <= value <= 1
                if not valid:
                    raise ValueError(f"Out-of-range {metric} in seed {bundle['seed']}")

    rows = pd.DataFrame(record for bundle in bundles for record in bundle["records"])
    rows = rows.sort_values(["seed", "architecture"]).reset_index(drop=True)
    paired = rows.pivot(index="seed", columns="architecture", values="AUPRC")
    paired["MM-Kla minus PCBert-Kla"] = paired[model_architecture] - paired["baseline"]
    paired["MM-Kla minus HybridKla-aligned"] = paired[model_architecture] - paired["hybridkla_architecture_aligned"]
    summary = rows.groupby("architecture")[list(METRICS)].agg(["mean", "std"])
    return rows, paired, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundles", nargs=3, required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-architecture", choices=["token_gated", "token_gated_revised"], default="token_gated")
    args = parser.parse_args()
    rows, paired, summary = aggregate(args.bundles, model_architecture=args.model_architecture)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output_dir / "validation_results.csv", index=False)
    paired.to_csv(args.output_dir / "validation_paired_auprc.csv")
    summary.to_csv(args.output_dir / "validation_mean_std.csv")
    print(rows.to_string(index=False))
    print("\nPaired validation AUPRC:\n" + paired.to_string())
    print("\nMean and sample standard deviation:\n" + summary.to_string())
    print("\nSaved combined validation tables to", args.output_dir)


if __name__ == "__main__":
    main()
