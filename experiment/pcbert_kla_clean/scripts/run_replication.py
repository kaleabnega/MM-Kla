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
            "source-study-check",
            "source-study-development",
            "source-study",
            "human-study-check",
            "human-study-development",
            "human-study",
            "clear-lactyl-check",
            "clear-lactyl-development",
            "hybridkla-source-check",
            "hybridkla-source-development",
            "hybridkla-source",
            "hybridkla-study-check",
            "hybridkla-study-development",
            "hybridkla-study",
            "pcbert-revised-check",
            "pcbert-revised-development",
            "pcbert-revised",
        ],
        default="data-check",
    )
    parser.add_argument("--model-name", default="Rostlab/prot_bert")
    parser.add_argument(
        "--source-lock",
        type=Path,
        default=PROJECT_DIR / "review_data/locked_source_study_v1/protocol_lock.json",
    )
    parser.add_argument(
        "--human-lock", type=Path,
        default=PROJECT_DIR / "outputs/kladb_human_locked/split_audit.json",
    )
    parser.add_argument(
        "--clear-lock", type=Path,
        default=PROJECT_DIR / "outputs/clear_lactyl_locked/split_audit.json",
    )
    parser.add_argument(
        "--hybridkla-source-lock", type=Path,
        default=PROJECT_DIR / "outputs/hybridkla_source_locked/split_audit.json",
    )
    parser.add_argument(
        "--hybridkla-study-lock", type=Path,
        default=PROJECT_DIR / "outputs/hybridkla_study_locked/split_audit.json",
    )
    parser.add_argument(
        "--pcbert-revised-lock", type=Path,
        default=PROJECT_DIR / "review_data/pcbert_revised_v1/split_audit.json",
    )
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
        choices=["baseline", "token_gated", "token_gated_bilstm", "token_gated_revised", "token_gated_conv_concat", "hybrid_gated"],
        default="baseline",
        help=(
            "Model architecture. 'baseline' preserves PCBert-Kla; 'token_gated' "
            "uses site-aware token attention and gated physicochemical fusion; "
            "'token_gated_bilstm' adds central-lysine raw-residue BiLSTM context; "
            "'token_gated_revised' adds K-centered convolutions of widths 3, 5, and 9; "
            "'token_gated_conv_concat' replaces its gate with concatenation fusion; "
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
    parser.add_argument(
        "--head-learning-rate", type=float, default=None,
        help="Optional learning rate for non-encoder parameters; defaults to --learning-rate.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--positive-weighting",
        choices=["none", "train_ratio"],
        default="none",
        help="Weight positive examples in training BCE by training negatives/positives.",
    )
    parser.add_argument("--sgd-momentum", type=float, default=0.0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--scheduler", choices=["none", "linear"], default="none")
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument(
        "--add-cksaap", action="store_true",
        help="Add normalized CKSAAP gaps 0-3 to the 27 CLEAR-Lactyl sequence descriptors.",
    )
    parser.add_argument(
        "--train-last-encoder-layers", type=int, default=None,
        help="Keep --encoder-layers pretrained layers, but update only the last N; token_gated only.",
    )
    parser.add_argument(
        "--encoder-adaptation", choices=["full", "lora"], default="full",
        help="Train all encoder weights or LoRA query/key/value adapters.",
    )
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.3)
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Train only the task-specific head while using the PLM as a feature extractor.",
    )
    parser.add_argument("--fusion-dim", type=int, default=256)
    parser.add_argument("--attention-dim", type=int, default=256)
    parser.add_argument("--arch-dropout", type=float, default=0.2)
    parser.add_argument("--residue-embedding-dim", type=int, default=64)
    parser.add_argument("--lstm-hidden-dim", type=int, default=64)
    parser.add_argument("--lstm-layers", type=int, default=1)
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
    if args.encoder_adaptation == "lora" and args.architecture != "token_gated":
        raise ValueError("LoRA encoder adaptation currently supports --architecture token_gated only")
    if args.train_last_encoder_layers is not None and args.architecture != "token_gated":
        raise ValueError("--train-last-encoder-layers requires --architecture token_gated")
    if args.add_cksaap and args.run not in {"clear-lactyl-check", "clear-lactyl-development"}:
        raise ValueError("--add-cksaap currently supports CLEAR-Lactyl runs only")
    if args.monitor_metric is None:
        args.monitor_metric = "AUPRC" if args.run in {"window-diagnostic", "candidate-group-diagnostic", "source-study", "source-study-development", "human-study", "human-study-development", "clear-lactyl-development", "hybridkla-source", "hybridkla-source-development", "hybridkla-study", "hybridkla-study-development", "pcbert-revised", "pcbert-revised-development"} else "ACC"

    if args.run in {"clear-lactyl-check", "clear-lactyl-development"}:
        from pcbert_kla_clean.clear_lactyl import load_clear_lactyl
        from pcbert_kla_clean.source_study import sha256

        development, validation_indices, internal_test = load_clear_lactyl(
            args.clear_lock, add_cksaap=args.add_cksaap,
        )
        report = {
            "training_rows": len(development.records) - len(validation_indices),
            "validation_rows": len(validation_indices),
            "internal_test_rows": len(internal_test.records),
            "feature_dim": development.features.shape[1],
            "clear_lock_sha256": sha256(args.clear_lock),
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        if args.run == "clear-lactyl-check":
            return
        if args.calibrate_threshold != "none" or args.monitor_metric != "AUPRC":
            raise ValueError("CLEAR-Lactyl development requires validation AUPRC without threshold calibration")
        from pcbert_kla_clean.backbones import infer_site_token_index, load_tokenizer
        from pcbert_kla_clean.experiments import train_and_predict_independent
        import torch

        if args.site_token_index is None:
            args.site_token_index = infer_site_token_index(args.model_name)
        args.device = resolve_device(args.device)
        config = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items() if key != "output_dir"
        }
        run_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
        output_dir = args.output_dir / "clear_lactyl" / "runs" / run_id
        result = train_and_predict_independent(
            args, development, None, seed=args.seed,
            tokenizer=load_tokenizer(args.model_name, args.cache_dir),
            device=torch.device(args.device), output_dir=output_dir,
            fixed_validation_indices=validation_indices,
        )
        print("\nCLEAR-Lactyl validation (development only):")
        print(json.dumps({"loss": result["validation_loss"],
                          **result["validation_metrics"]}, indent=2))
        return

    if args.run in {"pcbert-revised-check", "pcbert-revised-development", "pcbert-revised"}:
        from pcbert_kla_clean.pcbert_revised import load_pcbert_revised

        development, validation_indices, internal_test = load_pcbert_revised(
            args.pcbert_revised_lock
        )
        report = {
            "training_rows": len(development.records) - len(validation_indices),
            "validation_rows": len(validation_indices),
            "internal_test_rows": len(internal_test.records),
            "revised_lock_sha256": hashlib.sha256(args.pcbert_revised_lock.read_bytes()).hexdigest(),
            "scope": "retrospective candidate-homology diagnostic; no verified parent IDs or external study",
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        if args.run == "pcbert-revised-check":
            return
        if args.ablation != "none" or args.calibrate_threshold != "none":
            raise ValueError("Primary revised comparison requires no ablation or threshold calibration")
        if args.monitor_metric != "AUPRC" or args.decision_threshold != 0.5:
            raise ValueError("Primary revised comparison requires validation AUPRC and threshold 0.5")
        from pcbert_kla_clean.backbones import infer_site_token_index, load_tokenizer
        from pcbert_kla_clean.experiments import train_and_predict_independent
        import torch

        if args.site_token_index is None:
            args.site_token_index = infer_site_token_index(args.model_name)
        args.device = resolve_device(args.device)
        args.revised_lock_sha256 = report["revised_lock_sha256"]
        config = {key: str(value) if isinstance(value, Path) else value
                  for key, value in vars(args).items() if key != "output_dir"}
        run_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
        output_dir = args.output_dir / "pcbert_revised" / "runs" / run_id
        development_only = args.run == "pcbert-revised-development"
        result = train_and_predict_independent(
            args, development, None if development_only else internal_test, seed=args.seed,
            tokenizer=load_tokenizer(args.model_name, args.cache_dir),
            device=torch.device(args.device), output_dir=output_dir,
            fixed_validation_indices=validation_indices,
        )
        if development_only:
            print("\nRevised PCBert-Kla validation (development only):")
            print(json.dumps({"loss": result["validation_loss"],
                              **result["validation_metrics"]}, indent=2))
        else:
            print("\nRevised PCBert-Kla internal test:")
            print(json.dumps({"loss": result["loss"], **result["metrics"]}, indent=2))
        return

    if args.run in {"hybridkla-study-check", "hybridkla-study-development", "hybridkla-study"}:
        from pcbert_kla_clean.hybridkla_study import load_hybridkla_study
        from pcbert_kla_clean.source_study import sha256

        development, validation_indices, internal_test, external_test = load_hybridkla_study(
            args.hybridkla_study_lock
        )
        report = {
            "training_rows": len(development.records) - len(validation_indices),
            "validation_rows": len(validation_indices),
            "internal_test_rows": len(internal_test.records),
            "external_test_rows": len(external_test.records),
            "hybridkla_study_lock_sha256": sha256(args.hybridkla_study_lock),
            "scope": "human S1 to mouse S7; S7 study independence is paper-designated, not per-site verified",
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        if args.run == "hybridkla-study-check":
            return
        if args.ablation != "none" or args.calibrate_threshold != "none":
            raise ValueError("Primary HybridKla study comparison requires no ablation or threshold calibration")
        if args.monitor_metric != "AUPRC" or args.decision_threshold != 0.5:
            raise ValueError("Primary comparison requires validation AUPRC and threshold 0.5")
        from pcbert_kla_clean.backbones import infer_site_token_index, load_tokenizer
        from pcbert_kla_clean.experiments import train_and_predict_independent
        import torch

        if args.site_token_index is None:
            args.site_token_index = infer_site_token_index(args.model_name)
        args.device = resolve_device(args.device)
        args.hybridkla_study_lock_sha256 = report["hybridkla_study_lock_sha256"]
        config = {key: str(value) if isinstance(value, Path) else value
                  for key, value in vars(args).items() if key != "output_dir"}
        run_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
        output_dir = args.output_dir / "hybridkla_study" / "runs" / run_id
        development_only = args.run == "hybridkla-study-development"
        result = train_and_predict_independent(
            args, development, None if development_only else internal_test, seed=args.seed,
            tokenizer=load_tokenizer(args.model_name, args.cache_dir),
            device=torch.device(args.device), output_dir=output_dir,
            fixed_validation_indices=validation_indices,
            extra_test_splits=None if development_only else {"external_mouse": external_test},
        )
        if development_only:
            print("\nHybridKla study validation (development only):")
            print(json.dumps({"loss": result["validation_loss"],
                              **result["validation_metrics"]}, indent=2))
        else:
            print("\nHybridKla study internal test:")
            print(json.dumps({"loss": result["loss"], **result["metrics"]}, indent=2))
            print("\nHybridKla study mouse external test:")
            print(json.dumps(result["extra_metrics"]["external_mouse"], indent=2))
        return

    if args.run in {"hybridkla-source-check", "hybridkla-source-development", "hybridkla-source"}:
        from pcbert_kla_clean.hybridkla_source_study import load_hybridkla_source_study
        from pcbert_kla_clean.source_study import sha256

        development, validation_indices, internal_test = load_hybridkla_source_study(
            args.hybridkla_source_lock
        )
        report = {"training_rows": len(development.records) - len(validation_indices),
                  "validation_rows": len(validation_indices),
                  "internal_test_rows": len(internal_test.records),
                  "hybridkla_source_lock_sha256": sha256(args.hybridkla_source_lock),
                  "scope": "provisional internal benchmark; no external test or verified source provenance"}
        print(json.dumps(report, indent=2, sort_keys=True))
        if args.run == "hybridkla-source-check":
            return
        if args.ablation != "none" or args.calibrate_threshold != "none":
            raise ValueError("HybridKla source primary comparison requires no ablation or threshold calibration")
        if args.monitor_metric != "AUPRC" or args.decision_threshold != 0.5:
            raise ValueError("HybridKla source comparison requires validation AUPRC and threshold 0.5")
        from pcbert_kla_clean.backbones import infer_site_token_index, load_tokenizer
        from pcbert_kla_clean.experiments import train_and_predict_independent
        import torch

        if args.site_token_index is None:
            args.site_token_index = infer_site_token_index(args.model_name)
        args.device = resolve_device(args.device)
        args.hybridkla_source_lock_sha256 = report["hybridkla_source_lock_sha256"]
        config = {key: str(value) if isinstance(value, Path) else value
                  for key, value in vars(args).items() if key != "output_dir"}
        run_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
        output_dir = args.output_dir / "hybridkla_source" / "runs" / run_id
        development_only = args.run == "hybridkla-source-development"
        result = train_and_predict_independent(
            args, development, None if development_only else internal_test, seed=args.seed,
            tokenizer=load_tokenizer(args.model_name, args.cache_dir),
            device=torch.device(args.device), output_dir=output_dir,
            fixed_validation_indices=validation_indices,
        )
        if development_only:
            print("\nHybridKla source validation (development only):")
            print(json.dumps({"loss": result["validation_loss"],
                              **result["validation_metrics"]}, indent=2))
        else:
            print("\nHybridKla source internal test:")
            print(json.dumps({"loss": result["loss"], **result["metrics"]}, indent=2))
        return

    if args.run in {"human-study-check", "human-study-development", "human-study"}:
        from pcbert_kla_clean.human_study import load_human_study
        from pcbert_kla_clean.source_study import sha256

        development, validation_indices, internal_test, external_test = load_human_study(
            args.human_lock
        )
        report = {
            "training_rows": len(development.records) - len(validation_indices),
            "validation_rows": len(validation_indices),
            "internal_test_rows": len(internal_test.records),
            "external_test_rows": len(external_test.records),
            "human_lock_sha256": sha256(args.human_lock),
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        if args.run == "human-study-check":
            return
        if args.ablation != "none" or args.calibrate_threshold != "none":
            raise ValueError("Human primary comparison requires no ablation or threshold calibration")
        if args.monitor_metric != "AUPRC" or args.decision_threshold != 0.5:
            raise ValueError("Human primary comparison requires validation AUPRC and threshold 0.5")
        from pcbert_kla_clean.backbones import infer_site_token_index, load_tokenizer
        from pcbert_kla_clean.experiments import train_and_predict_independent
        import torch

        if args.site_token_index is None:
            args.site_token_index = infer_site_token_index(args.model_name)
        args.device = resolve_device(args.device)
        args.human_lock_sha256 = report["human_lock_sha256"]
        config = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items() if key != "output_dir"
        }
        run_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
        output_dir = args.output_dir / "human_study" / "runs" / run_id
        development_only = args.run == "human-study-development"
        result = train_and_predict_independent(
            args, development, None if development_only else internal_test, seed=args.seed,
            tokenizer=load_tokenizer(args.model_name, args.cache_dir),
            device=torch.device(args.device), output_dir=output_dir,
            fixed_validation_indices=validation_indices,
            extra_test_splits=None if development_only else {"external_study": external_test},
        )
        if development_only:
            print("\nHuman validation (development only):")
            print(json.dumps({"loss": result["validation_loss"], **result["validation_metrics"]}, indent=2))
        else:
            print("\nHuman internal test:")
            print(json.dumps({"loss": result["loss"], **result["metrics"]}, indent=2))
            print("\nHuman study-held-out test:")
            print(json.dumps(result["extra_metrics"]["external_study"], indent=2))
        return

    if args.run in {"source-study-check", "source-study-development", "source-study"}:
        from pcbert_kla_clean.source_study import load_locked_partitions, sha256

        gao_path = PROJECT_DIR / "review_data/gao_2020_source_split/source_split_manifest.csv"
        rice_path = PROJECT_DIR / "review_data/source_study_audit_2026_11_01/rice_homology_screened_candidates.csv"
        development, validation_indices, internal_test, rice_external = load_locked_partitions(
            gao_path, rice_path, args.source_lock
        )
        report = {
            "training_rows": len(development.records) - len(validation_indices),
            "validation_rows": len(validation_indices),
            "internal_test_rows": len(internal_test.records),
            "external_rice_rows": len(rice_external.records),
            "source_lock_sha256": sha256(args.source_lock),
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        if args.run == "source-study-check":
            return
        if args.ablation != "none":
            raise ValueError("Source-study primary comparison requires --ablation none")
        if args.calibrate_threshold != "none":
            raise ValueError("Source-study primary comparison uses the predeclared fixed threshold")
        if args.monitor_metric != "AUPRC" or args.decision_threshold != 0.5:
            raise ValueError("Source-study primary comparison requires validation AUPRC and threshold 0.5")
        from pcbert_kla_clean.backbones import infer_site_token_index, load_tokenizer
        from pcbert_kla_clean.experiments import train_and_predict_independent
        import torch

        if args.site_token_index is None:
            args.site_token_index = infer_site_token_index(args.model_name)
        args.device = resolve_device(args.device)
        args.source_lock_sha256 = report["source_lock_sha256"]
        config = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "output_dir"
        }
        config["source_lock_sha256"] = report["source_lock_sha256"]
        run_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
        output_dir = args.output_dir / "source_study" / "runs" / run_id
        print(f"Source-study run output: {output_dir}")
        development_only = args.run == "source-study-development"
        result = train_and_predict_independent(
            args, development, None if development_only else internal_test, seed=args.seed,
            tokenizer=load_tokenizer(args.model_name, args.cache_dir),
            device=torch.device(args.device), output_dir=output_dir,
            fixed_validation_indices=validation_indices,
            extra_test_splits=None if development_only else {"rice_external": rice_external},
        )
        if development_only:
            print("\nGao validation (development only):")
            print(json.dumps(
                {"loss": result["validation_loss"], **result["validation_metrics"]},
                indent=2, sort_keys=True,
            ))
            return
        print("\nGao internal test:")
        print(json.dumps({"loss": result["loss"], **result["metrics"]}, indent=2, sort_keys=True))
        print("\nRice study-held-out test (putative negatives):")
        print(json.dumps(result["extra_metrics"]["rice_external"], indent=2, sort_keys=True))
        return

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
