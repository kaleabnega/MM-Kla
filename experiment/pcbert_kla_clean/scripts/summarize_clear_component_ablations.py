"""Summarize matched, validation-only CLEAR-Lactyl component controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


MODES = ("none", "cls_pool", "mean_pool", "max_pool", "concat_fusion",
         "scalar_gate", "no_residual", "frozen_encoder")
SEEDS = (42, 123, 777)
METRICS = ("AUPRC", "AUC", "MCC", "F1", "Pre", "Rec", "SP", "ACC")


def summarize(root: Path, allow_partial: bool = False) -> pd.DataFrame:
    rows = []
    missing = []
    lock_hashes = set()
    for seed in SEEDS:
        for mode in MODES:
            run_dir = root / mode / f"seed_{seed}" / "results" / "clear_lactyl" / "runs"
            candidates = list(run_dir.glob("*/validation_metrics.json"))
            if not candidates:
                missing.append(f"{mode}/seed_{seed}")
                continue
            if len(candidates) != 1:
                raise ValueError(f"Expected one completed run in {run_dir}")
            metrics = json.loads(candidates[0].read_text())
            config = json.loads(candidates[0].with_name("run_config.json").read_text())
            if (config["run"] != "clear-lactyl-development" or
                    config["test_rows"] != 0 or
                    config["architecture"] != "token_gated" or
                    config["model_seed"] != seed or
                    config["training_rows"] != 20599 or
                    config["validation_rows"] != 2944 or
                    config["decision_threshold_used"] != 0.5 or
                    config["monitor_metric"] != "AUPRC" or
                    config["model_name"] != "Rostlab/prot_bert" or
                    config["encoder_layers"] != 4 or
                    not config["add_cksaap"] or
                    config["optimizer"] != "adamw" or
                    config["learning_rate"] != 2e-5 or
                    config["head_learning_rate"] != 1e-4 or
                    config["epochs"] != 30 or
                    config["patience"] != 5 or
                    config["positive_weighting"] != "none" or
                    config["ablation"] != ("none" if mode == "frozen_encoder" else mode) or
                    config["freeze_encoder"] != (mode == "frozen_encoder")):
                raise ValueError(f"Protocol mismatch in {run_dir}")
            lock_hashes.add(config["clear_lock_sha256"])
            rows.append({"seed": seed, "mode": mode,
                         "selected_epoch": config["selected_epoch"],
                         **{key: metrics[key] for key in METRICS}})
    if len(lock_hashes) > 1:
        raise ValueError("Runs use different locked datasets")
    if missing and not allow_partial:
        raise FileNotFoundError("Missing completed runs: " + ", ".join(missing))
    if missing:
        print("Incomplete runs:", ", ".join(missing))
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    control = frame.loc[frame["mode"] == "none"].set_index("seed")
    for key in METRICS:
        frame[f"delta_{key}"] = frame.apply(
            lambda row: row[key] - control.loc[row["seed"], key]
            if row["seed"] in control.index else float("nan"), axis=1,
        )
    return frame.sort_values(["seed", "mode"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    frame = summarize(args.output_root, allow_partial=args.allow_partial)
    if frame.empty:
        print("No completed runs")
        return
    print(frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nBy variant, mean and sample standard deviation across available seeds:")
    print(frame.groupby("mode")[list(METRICS)].agg(["mean", "std"]).to_string(
        float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
