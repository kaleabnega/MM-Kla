"""Summarize validation-only grouped-OOF component controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MODES = ("none", "cls_pool", "mean_pool", "max_pool", "concat_fusion",
         "scalar_gate", "no_gated_fusion", "no_residual", "frozen_encoder")
METRICS = ("AUPRC", "AUC", "MCC", "F1", "Pre", "Rec", "SP", "ACC")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads((args.reference_dir / "report.json").read_text())
    rows = []
    for mode in MODES:
        report = json.loads((args.output_root / mode / "seed_42" / "results" / "report.json").read_text())
        if (report["seed"] != 42 or report["lock_sha256"] != reference["lock_sha256"] or
                report["internal_test_scored"] or report["training_rows"] != reference["training_rows"] or
                report["validation_rows"] != reference["validation_rows"] or
                len(report["folds"]) != 5 or
                report["protbert_ablation"] != ("none" if mode == "frozen_encoder" else mode) or
                report["freeze_protbert"] != (mode == "frozen_encoder")):
            raise ValueError(f"Protocol mismatch: {mode}")
        rows.append((mode, report["validation_metrics"]["oof_logistic_combination"]))
    control = rows[0][1]
    original = reference["validation_metrics"]["oof_logistic_combination"]
    for metric in ("AUPRC", "AUC"):
        if abs(control[metric] - original[metric]) > 0.001:
            raise ValueError(f"Rerun of unchanged stack did not reproduce {metric}")
    print("All rows: locked CLEAR-Lactyl validation, seed 42; never the internal test.")
    print("mode".ljust(19), *(metric.rjust(8) for metric in METRICS), "delta_AP".rjust(9))
    for mode, metrics in rows:
        print(mode.ljust(19), *(f"{metrics[key]:8.4f}" for key in METRICS),
              f"{metrics['AUPRC'] - control['AUPRC']:+9.4f}")


if __name__ == "__main__":
    main()
