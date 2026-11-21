"""Recompute locked CLEAR-Lactyl results and manuscript figures from saved scores."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


ROOT = Path(__file__).resolve().parents[2] / "pcbert_kla_clean"
KEY = ["accession", "position_1based", "label"]
SCORES = {
    "PCBert-Kla": ("pcbert_kla_predictions.csv", "score"),
    "MM-Kla (original token gate)": ("original_mm_kla_predictions.csv", "score"),
    "DeepKla-aligned": ("deepkla_aligned_predictions.csv", "score"),
    "HybridKla-aligned": ("hybridkla_aligned_predictions.csv", "score"),
    "TransKla": ("test_predictions.csv", "score"),
    "MM-Kla (ProtBERT branch)": ("internal_test_predictions.csv", "protbert_cksaap"),
    "MM-Kla (positional branch)": ("internal_test_predictions.csv", "positional_sequence"),
    "MM-Kla (score average)": ("internal_test_predictions.csv", "simple_average"),
    "MM-Kla (OOF combination)": ("internal_test_predictions.csv", "oof_logistic_combination"),
}
COLORS = {
    "PCBert-Kla": "#546574",
    "HybridKla-aligned": "#b65732",
    "TransKla": "#94652c",
    "MM-Kla (OOF combination)": "#167c80",
    "MM-Kla (positional branch)": "#3867ad",
    "MM-Kla (ProtBERT branch)": "#8d639e",
}
MAIN = [
    "PCBert-Kla", "DeepKla-aligned", "HybridKla-aligned", "TransKla",
    "MM-Kla (original token gate)", "MM-Kla (ProtBERT branch)",
    "MM-Kla (positional branch)", "MM-Kla (OOF combination)",
]


def metrics(y, score):
    pred = score >= 0.5
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "AUPRC": float(average_precision_score(y, score)),
        "AUROC": float(roc_auc_score(y, score)),
        "MCC": float(matthews_corrcoef(y, pred)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)),
        "accuracy": float(accuracy_score(y, pred)),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }


def load_scores(input_dir, manifest_path):
    manifest = pd.read_csv(manifest_path)
    manifest = manifest.loc[manifest.partition == "internal_test", KEY + ["window_51aa", "homology_group"]]
    if manifest.duplicated(KEY[:2]).any():
        raise ValueError("Duplicate test accession-position in locked manifest")
    manifest = manifest.sort_values(KEY[:2]).reset_index(drop=True)
    if len(manifest) != 5888 or manifest.label.sum() != 2911:
        raise ValueError("Unexpected locked test counts")
    arrays = {}
    provenance = {}
    for name, (filename, column) in SCORES.items():
        path = input_dir / filename
        frame = pd.read_csv(path)
        if frame.duplicated(KEY[:2]).any():
            raise ValueError(f"Duplicate accession-position in {path}")
        frame = frame.sort_values(KEY[:2]).reset_index(drop=True)
        if not frame[KEY].equals(manifest[KEY]):
            raise ValueError(f"Row IDs or labels differ from locked manifest: {path}")
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"Invalid score in {path}:{column}")
        arrays[name] = values
        provenance[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    return manifest, arrays, provenance


def grouped_bootstrap(y, scores, groups, repetitions, seed):
    group_codes = pd.factorize(groups)[0]
    members = [np.flatnonzero(group_codes == g) for g in range(group_codes.max() + 1)]
    rng = np.random.default_rng(seed)
    targets = ["PCBert-Kla", "HybridKla-aligned", "TransKla", "MM-Kla (positional branch)"]
    base = scores["MM-Kla (OOF combination)"]
    distributions = {name: [] for name in targets}
    for _ in range(repetitions):
        draw = rng.integers(0, len(members), size=len(members))
        idx = np.concatenate([members[g] for g in draw])
        if len(np.unique(y[idx])) != 2:
            continue
        base_ap = average_precision_score(y[idx], base[idx])
        for name in targets:
            distributions[name].append(base_ap - average_precision_score(y[idx], scores[name][idx]))
    return {
        name: {
            "delta": float(average_precision_score(y, base) - average_precision_score(y, scores[name])),
            "percentile_95_ci": [float(x) for x in np.percentile(values, [2.5, 97.5])],
            "bootstrap_replicates": len(values),
        }
        for name, values in distributions.items()
    }


def plot_curves(y, scores, out):
    names = ["PCBert-Kla", "HybridKla-aligned", "TransKla", "MM-Kla (OOF combination)"]
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 4.2))
    for name in names:
        fpr, tpr, _ = roc_curve(y, scores[name])
        precision, recall, _ = precision_recall_curve(y, scores[name])
        axes[0].plot(fpr, tpr, label=f"{name} ({roc_auc_score(y, scores[name]):.3f})", lw=1.8,
                     color=COLORS[name])
        axes[1].plot(recall, precision, label=f"{name} ({average_precision_score(y, scores[name]):.3f})",
                     lw=1.8, color=COLORS[name])
    axes[0].plot([0, 1], [0, 1], ls="--", lw=1, color="0.6")
    axes[1].axhline(y.mean(), ls="--", lw=1, color="0.6", label=f"Prevalence ({y.mean():.3f})")
    for ax, title, xlabel, ylabel in zip(axes, ["ROC", "Precision-recall"],
        ["False-positive rate", "Recall"], ["True-positive rate", "Precision"]):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1.01))
        ax.grid(alpha=0.15)
        ax.legend(loc="lower right" if title == "ROC" else "lower left", fontsize=7.2, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "heldout_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_components(y, scores, out):
    names = ["MM-Kla (ProtBERT branch)", "MM-Kla (positional branch)",
             "MM-Kla (score average)", "MM-Kla (OOF combination)"]
    labels = ["ProtBERT + descriptors", "Positional predictor", "Score average", "OOF combination"]
    fig, ax = plt.subplots(figsize=(7.1, 3.2))
    values = [average_precision_score(y, scores[name]) for name in names]
    bars = ax.barh(labels[::-1], values[::-1], color=["#8d639e", "#3867ad", "#7a8990", "#167c80"][::-1])
    ax.set(xlabel="Held-out average precision (AUPRC)", xlim=(0, 1))
    for bar, value in zip(bars, values[::-1]):
        ax.text(value + 0.012, bar.get_y() + bar.get_height()/2, f"{value:.3f}", va="center", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "component_scores.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_confusion(y, scores, out):
    names = ["PCBert-Kla", "HybridKla-aligned", "TransKla", "MM-Kla (OOF combination)"]
    fig, axes = plt.subplots(1, 4, figsize=(10.6, 2.7), constrained_layout=True)
    for ax, name in zip(axes, names):
        cm = confusion_matrix(y, scores[name] >= 0.5, labels=[0, 1])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=3000)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > 1600 else "#1c2935", fontsize=9)
        ax.set(title=name, xticks=[0, 1], yticks=[0, 1], xlabel="Predicted")
        ax.set_xticklabels(["Negative", "Kla"], fontsize=7)
        ax.set_yticklabels(["Negative", "Kla"], fontsize=7)
    axes[0].set_ylabel("Reference label")
    fig.savefig(out / "confusion_matrices.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_sequence_context(manifest, out):
    residues = "ACDEFGHIKLMNPQRSTVWY"
    windows = manifest.window_51aa.to_numpy()
    labels = manifest.label.to_numpy()
    array = np.array([list(w) for w in windows])
    if array.shape[1] != 51 or not (array[:, 25] == "K").all():
        raise ValueError("Unexpected sequence window format")
    pos = np.stack([(array[labels == 1] == aa).mean(axis=0) for aa in residues])
    neg = np.stack([(array[labels == 0] == aa).mean(axis=0) for aa in residues])
    diff = pos - neg
    scale = np.quantile(abs(diff[:, np.arange(51) != 25]), 0.99)
    fig, ax = plt.subplots(figsize=(11, 4.4))
    image = ax.imshow(diff, cmap="RdBu_r", aspect="auto", vmin=-scale, vmax=scale)
    ax.set(yticks=np.arange(20), yticklabels=list(residues), xticks=np.arange(0, 51, 5),
           xticklabels=np.arange(-25, 26, 5), xlabel="Position relative to central K",
           ylabel="Amino acid")
    ax.axvline(25, color="black", lw=0.8)
    cb = fig.colorbar(image, ax=ax, pad=0.01, shrink=0.85)
    cb.set_label("Frequency: Kla minus putative negative")
    fig.tight_layout()
    fig.savefig(out / "sequence_context.pdf", bbox_inches="tight")
    plt.close(fig)
    return diff


def write_table(results, out):
    columns = ["AUPRC", "AUROC", "MCC", "F1", "precision", "recall", "specificity", "accuracy"]
    with (out / "heldout_table.tex").open("w") as file:
        file.write("\\begin{tabular}{lrrrrrrrr}\\toprule\n")
        file.write("Method & AUPRC & AUROC & MCC & F1 & Pre & Rec & SP & ACC\\\\\\midrule\n")
        for name in MAIN:
            vals = " & ".join(f"{results[name][col]:.3f}" for col in columns)
            file.write(f"{name.replace('MM-Kla (OOF combination)', 'MM-Kla (two-predictor)')} & {vals}\\\\\n")
        file.write("\\bottomrule\\end{tabular}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).parent / "input")
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "outputs/clear_lactyl_locked/model_manifest.csv")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "revision/figures_generated")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest, scores, hashes = load_scores(args.input_dir, args.manifest)
    y = manifest.label.to_numpy(dtype=int)
    results = {name: metrics(y, score) for name, score in scores.items()}
    bootstrap = grouped_bootstrap(y, scores, manifest.homology_group, args.bootstrap, seed=20261121)
    plot_curves(y, scores, args.output_dir)
    plot_components(y, scores, args.output_dir)
    plot_confusion(y, scores, args.output_dir)
    diff = plot_sequence_context(manifest, args.output_dir)
    write_table(results, args.output_dir)
    summary = {
        "source": "locked CLEAR-Lactyl internal held-out test", "rows": len(y),
        "positives": int(y.sum()), "negatives": int(len(y)-y.sum()),
        "homology_groups": int(manifest.homology_group.nunique()),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "prediction_sha256": hashes, "metrics_at_0_5": results,
        "paired_group_bootstrap_auprc": bootstrap,
        "sequence_context_max_abs_frequency_difference": float(abs(diff).max()),
    }
    (args.output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"rows": len(y), "metrics": results, "paired_auprc": bootstrap}, indent=2))


if __name__ == "__main__":
    main()
