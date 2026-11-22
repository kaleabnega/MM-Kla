"""Validate and summarize the two-seed, full-stack CLEAR-Lactyl ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


OLD_MODES = ("none", "cls_pool", "mean_pool", "max_pool", "concat_fusion",
             "scalar_gate", "no_gated_fusion", "no_residual", "frozen_encoder")
POSITIONAL_MODES = ("no_position_embedding", "one_lstm_layer", "positional_mean_pool")
FEATURE_MODES = ("no_descriptors", "no_cksaap")
MODES = OLD_MODES + POSITIONAL_MODES + FEATURE_MODES
METRICS = ("AUPRC", "AUC", "MCC", "F1", "Pre", "Rec", "SP", "ACC")


def read_metrics(root: Path, reference_paths: dict[int, Path]) -> dict[int, dict[str, dict]]:
    results = {}
    for seed in (42, 123):
        reference = json.loads((reference_paths[seed] / "report.json").read_text())
        if reference["seed"] != seed or reference["internal_test_scored"]:
            raise ValueError(f"Invalid seed-{seed} reference")
        results[seed] = {}
        for mode in MODES:
            report = json.loads((root / mode / f"seed_{seed}" / "results" / "report.json").read_text())
            expected_pos = ("mean_pool" if mode == "positional_mean_pool" else mode) if mode in POSITIONAL_MODES else "none"
            expected_feature = mode if mode in FEATURE_MODES else "none"
            expected_prot = mode if mode in OLD_MODES and mode != "frozen_encoder" else "none"
            expected_dim = 1600 if mode == "no_descriptors" else 27 if mode == "no_cksaap" else 1627
            if (report["seed"] != seed or report["lock_sha256"] != reference["lock_sha256"] or
                    report["internal_test_scored"] or
                    report["training_rows"] != reference["training_rows"] or
                    report["validation_rows"] != reference["validation_rows"] or
                    len(report["folds"]) != 5 or report["feature_dim"] != 1627 or
                    report.get("used_feature_dim", report["feature_dim"]) != expected_dim or
                    report.get("positional_ablation", "none") != expected_pos or
                    report.get("feature_ablation", "none") != expected_feature or
                    report["protbert_ablation"] != expected_prot or
                    report["freeze_protbert"] != (mode == "frozen_encoder") or
                    bool(report.get("protbert_reference_dir")) != (mode in POSITIONAL_MODES) or
                    bool(report.get("positional_reference_dir")) != (mode not in POSITIONAL_MODES)):
                raise ValueError(f"Protocol mismatch: {mode}, seed {seed}")
            results[seed][mode] = report["validation_metrics"]["oof_logistic_combination"]
        for metric in ("AUPRC", "AUC"):
            actual = results[seed]["none"][metric]
            expected = reference["validation_metrics"]["oof_logistic_combination"][metric]
            if abs(actual - expected) > 0.001:
                raise ValueError(f"Seed-{seed} control did not reproduce {metric}")
    if (json.loads((reference_paths[42] / "report.json").read_text())["lock_sha256"] !=
            json.loads((reference_paths[123] / "report.json").read_text())["lock_sha256"]):
        raise ValueError("Seed references use different dataset locks")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed-42-reference", type=Path, required=True)
    parser.add_argument("--seed-123-reference", type=Path, required=True)
    args = parser.parse_args()
    rows = read_metrics(args.output_root, {
        42: args.seed_42_reference, 123: args.seed_123_reference,
    })
    print("Locked CLEAR-Lactyl validation only; five grouped folds per run; threshold 0.5.")
    print("mode".ljust(23), "seed", *(metric.rjust(8) for metric in METRICS), "delta_AP".rjust(9))
    for mode in MODES:
        for seed in (42, 123):
            metrics = rows[seed][mode]
            delta = metrics["AUPRC"] - rows[seed]["none"]["AUPRC"]
            print(mode.ljust(23), str(seed).rjust(4),
                  *(f"{metrics[key]:8.4f}" for key in METRICS), f"{delta:+9.4f}")
    print("Across-seed validation AUPRC (mean and sample standard deviation):")
    for mode in MODES:
        values = np.asarray([rows[seed][mode]["AUPRC"] for seed in (42, 123)])
        deltas = np.asarray([rows[seed][mode]["AUPRC"] - rows[seed]["none"]["AUPRC"]
                             for seed in (42, 123)])
        print(f"{mode:23} {values.mean():.4f} +/- {values.std(ddof=1):.4f}  "
              f"paired delta {deltas.mean():+.4f} (seed range {deltas.min():+.4f}, {deltas.max():+.4f})")


if __name__ == "__main__":
    main()
