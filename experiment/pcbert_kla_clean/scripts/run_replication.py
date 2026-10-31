from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from pcbert_kla_clean.data import (  # noqa: E402
    describe_split,
    load_split,
    train_test_overlap_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean PCBert-Kla replication runner")
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=PROJECT_DIR.parents[1] / "baselines" / "PCBert-Kla-original",
    )
    parser.add_argument(
        "--run",
        choices=[
            "data-check",
            "review-audit",
            "window-diagnostic",
            "candidate-group-audit",
            "candidate-group-diagnostic",
            "cv",
            "independent",
            "ensemble-independent",
        ],
        default="data-check",
    )
    parser.add_argument("--model-name", default="Rostlab/prot_bert")
    parser.add_argument(
        "--sequence-format",
        choices=["auto", "protbert_spaced", "prott5_spaced", "raw"],
        default="auto",
        help=(
            "How to format amino-acid windows before tokenization. Use auto for "
            "ProtBert/ProtT5 spaced residues and raw sequences for ESM-style models."
        ),
    )
    parser.add_argument(
        "--architecture",
        choices=["baseline", "token_gated", "hybrid_gated"],
        default="baseline",
        help=(
            "Model architecture. 'baseline' preserves PCBert-Kla; 'token_gated' "
            "uses site-aware token attention and gated physicochemical fusion; "
            "'hybrid_gated' preserves CLS context and adds site-aware pooling."
        ),
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ensemble-seeds", default="42,123,2025,3407,777")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--splitter", choices=["kfold", "stratified"], default="kfold")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--sgd-momentum", type=float, default=0.0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--scheduler", choices=["none", "linear"], default="none")
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Train only the task-specific head while using the PLM as a feature extractor.",
    )
    parser.add_argument("--fusion-dim", type=int, default=256)
    parser.add_argument("--attention-dim", type=int, default=256)
    parser.add_argument("--arch-dropout", type=float, default=0.2)
    parser.add_argument(
        "--ablation",
        choices=["none", "no_gated_fusion", "no_physicochemical", "no_sequence"],
        default="none",
        help=(
            "Token-gated ablation mode. Use 'none' for the full model, "
            "'no_gated_fusion' for ungated averaging of sequence/features, "
            "'no_physicochemical' for sequence-only, and 'no_sequence' for "
            "physicochemical-features-only."
        ),
    )
    parser.add_argument(
        "--site-token-index",
        type=int,
        default=None,
        help=(
            "Token index of the central lysine after tokenization. Defaults to 26 "
            "for CLS-based models and 25 for T5-style encoder models."
        ),
    )
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument(
        "--monitor-metric",
        choices=["ACC", "AUC", "AUPRC", "F1", "MCC", "Pre", "Rec", "SP"],
        default=None,
    )
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--review-split-seed", type=int, default=42)
    parser.add_argument("--review-test-fraction", type=float, default=0.2)
    parser.add_argument("--review-validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--candidate-mapping-csv",
        type=Path,
        default=PROJECT_DIR.parents[1]
        / "experiment-1/structure_aware_kla/results/botrytis_b0510_window_mapping/all_window_matches.csv",
    )
    parser.add_argument(
        "--homology-clusters-csv",
        type=Path,
        default=None,
        help="Optional complete accession,cluster_id CSV from a documented full-protein clustering run.",
    )
    parser.add_argument(
        "--homology-hits-tsv",
        type=Path,
        default=None,
        help="Optional complete MMseqs2 all-vs-all search TSV for cross-partition hit verification.",
    )
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument(
        "--calibrate-threshold",
        choices=["none", "ACC", "Rec", "Pre", "MCC", "F1", "SP"],
        default="none",
        help="Choose a decision threshold on the validation subset before testing.",
    )
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-steps", type=int, default=901)
    parser.add_argument(
        "--save-predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save validation/test probability CSV files for independent runs.",
    )
    parser.add_argument("--save-models", action="store_true")
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device != "auto":
        return device

    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def run_data_check(train_split, test_split) -> None:
    report = {
        "train": describe_split(train_split),
        "test": describe_split(test_split),
        "train_test_overlap": train_test_overlap_report(train_split, test_split),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.monitor_metric is None:
        args.monitor_metric = "AUPRC" if args.run in {"window-diagnostic", "candidate-group-diagnostic"} else "ACC"

    train_split = load_split(
        args.baseline_root / "data" / "train.csv",
        args.baseline_root / "data" / "feature_train.csv",
    )
    test_split = load_split(
        args.baseline_root / "data" / "test.csv",
        args.baseline_root / "data" / "feature_test.csv",
    )

    if args.run == "data-check":
        run_data_check(train_split, test_split)
        return

    fixed_validation_indices = None
    if args.run in {"review-audit", "window-diagnostic", "candidate-group-audit", "candidate-group-diagnostic"}:
        from pcbert_kla_clean.review_dataset import (
            combine_development_partitions,
            prepare_review_partitions,
            save_review_partitions,
        )

        split_kwargs = dict(
            seed=args.review_split_seed,
            test_fraction=args.review_test_fraction,
            validation_fraction=args.review_validation_fraction,
        )
        if args.run in {"candidate-group-audit", "candidate-group-diagnostic"}:
            from pcbert_kla_clean.candidate_groups import prepare_candidate_group_partitions

            partitions = prepare_candidate_group_partitions(
                train_split,
                test_split,
                mapping_csv=args.candidate_mapping_csv,
                homology_csv=args.homology_clusters_csv,
                homology_hits_tsv=args.homology_hits_tsv,
                **split_kwargs,
            )
            args.output_dir = args.output_dir / "review_candidate_groups"
        else:
            partitions = prepare_review_partitions(train_split, test_split, **split_kwargs)
            args.output_dir = args.output_dir / "review_exact_window"
        partitions.report["source_file_sha256"] = {
            filename: hashlib.sha256(
                (args.baseline_root / "data" / filename).read_bytes()
            ).hexdigest()
            for filename in (
                "train.csv",
                "test.csv",
                "feature_train.csv",
                "feature_test.csv",
            )
        }
        save_review_partitions(partitions, args.output_dir)
        print(json.dumps(partitions.report, indent=2, sort_keys=True))
        if args.run in {"review-audit", "candidate-group-audit"}:
            return
        train_split, fixed_validation_indices = combine_development_partitions(partitions)
        test_split = partitions.test

    from pcbert_kla_clean.backbones import infer_site_token_index

    if args.site_token_index is None:
        args.site_token_index = infer_site_token_index(args.model_name)
    args.device = resolve_device(args.device)
    if args.run in {"window-diagnostic", "candidate-group-diagnostic"}:
        run_settings = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        run_settings["dataset_audit_sha256"] = hashlib.sha256(
            (args.output_dir / "dataset_audit.json").read_bytes()
        ).hexdigest()
        run_settings["window_inventory_sha256"] = hashlib.sha256(
            (args.output_dir / "window_inventory.csv").read_bytes()
        ).hexdigest()
        run_id = hashlib.sha256(
            json.dumps(run_settings, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]
        args.output_dir = args.output_dir / "runs" / run_id
        print(f"Diagnostic run output: {args.output_dir}")
    from pcbert_kla_clean.experiments import (
        run_cv,
        run_ensemble_independent,
        run_independent,
    )

    if args.run == "cv":
        run_cv(args, train_split)
    elif args.run == "independent":
        run_independent(args, train_split, test_split)
    elif args.run in {"window-diagnostic", "candidate-group-diagnostic"}:
        run_independent(
            args,
            train_split,
            test_split,
            fixed_validation_indices=fixed_validation_indices,
            result_label=(
                "Candidate-protein-group diagnostic test:"
                if args.run == "candidate-group-diagnostic"
                else "Exact-window-disjoint diagnostic test:"
            ),
        )
    elif args.run == "ensemble-independent":
        run_ensemble_independent(args, train_split, test_split)
    else:
        raise ValueError(args.run)


if __name__ == "__main__":
    main()
