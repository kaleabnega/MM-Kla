"""Train and evaluate the published TransKla architecture on locked CLEAR-Lactyl."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
from pcbert_kla_clean.metrics import compute_binary_metrics  # noqa: E402


AUTHOR_COMMIT = "15ab472180cab74682f56f9f018b844cec264cc6"
AUTHOR_SOURCE_SHA256 = "3f0ac7070bbc94b82b08c063c9884a7d022974fcff0ca45012bd0b1483d75581"
AUTHOR_URL = "https://github.com/abduljabbar-repo/TransKla"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_author(source: Path):
    if sha256(source) != AUTHOR_SOURCE_SHA256:
        raise ValueError("TransKla source differs from the pinned author version")
    spec = importlib.util.spec_from_file_location("transkla_author", source)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot import the pinned TransKla source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_proteins(path: Path, required: set[str]) -> dict[str, str]:
    proteins: dict[str, str] = {}
    accession = None
    chunks: list[str] = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                if accession in required:
                    if accession in proteins:
                        raise ValueError(f"Duplicate reference accession: {accession}")
                    proteins[accession] = "".join(chunks).upper()
                identifier = line[1:].split()[0]
                accession = identifier.split("|")[1] if "|" in identifier else identifier
                chunks = []
            else:
                chunks.append(line.strip())
    if accession in required:
        if accession in proteins:
            raise ValueError(f"Duplicate reference accession: {accession}")
        proteins[accession] = "".join(chunks).upper()
    missing = required - proteins.keys()
    if missing:
        raise ValueError(f"Reference proteome lacks {len(missing)} locked accessions")
    return proteins


def load_lock(lock: Path, proteome: Path):
    audit = json.loads(lock.read_text())
    if not audit.get("protocol", "").startswith("CLEAR-Lactyl protein-"):
        raise ValueError("Not the locked CLEAR-Lactyl benchmark")
    manifest = lock.with_name("model_manifest.csv")
    manifest_hash = sha256(manifest)
    if manifest_hash != audit["output_sha256"]["model_manifest.csv"]:
        raise ValueError("CLEAR-Lactyl manifest checksum mismatch")
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    partitions: dict[str, list[dict[str, str]]] = defaultdict(list)
    protein_partition: dict[str, str] = {}
    group_partition: dict[str, str] = {}
    seen_windows: set[str] = set()
    for row in rows:
        part = row["partition"]
        if part not in {"train", "validation", "internal_test"}:
            raise ValueError(f"Unexpected partition: {part}")
        accession = row["accession"]
        group = row["homology_group"]
        window = row["window_51aa"]
        if not accession or not group or row["label"] not in {"0", "1"}:
            raise ValueError("Invalid locked record")
        if protein_partition.setdefault(accession, part) != part:
            raise ValueError("Protein crosses locked partitions")
        if group_partition.setdefault(group, part) != part:
            raise ValueError("Homology group crosses locked partitions")
        if len(window) != 51 or window[25] != "K" or window in seen_windows:
            raise ValueError("Duplicate or invalid locked 51-residue window")
        seen_windows.add(window)
        partitions[part].append(row)
    for part in ("train", "validation", "internal_test"):
        part_rows = partitions[part]
        expected = audit["partitions"][part]
        if (len(part_rows) != expected["rows"] or
                sum(row["label"] == "1" for row in part_rows) != expected["positives"]):
            raise ValueError(f"Locked {part} counts mismatch")

    proteins = load_proteins(proteome, set(protein_partition))
    for row in rows:
        position = int(row["position_1based"])
        protein = proteins[row["accession"]]
        window = row["window_51aa"]
        if protein[position - 26:position + 25] != window:
            raise ValueError(f"Reference sequence mismatch at {row['accession']}:{position}")
        if window[5:46][20] != "K":
            raise ValueError("TransKla 41-residue input is not lysine-centered")
    return partitions, proteins, manifest_hash


def write_inputs(output_dir: Path, partitions, proteins, names: tuple[str, ...]):
    data_dir = output_dir / "data"
    data_dir.mkdir()
    for part in names:
        with (data_dir / f"{part}.fasta").open("w", encoding="utf-8") as handle:
            for row in partitions[part]:
                handle.write(
                    f">{row['accession']}|pos={row['position_1based']}|{row['label']}\n"
                    f"{row['window_51aa'][5:46]}\n"
                )
    accessions = {row["accession"] for part in names for row in partitions[part]}
    with (data_dir / "proteins.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Protein_accession", "Sequence"])
        for accession in sorted(accessions):
            writer.writerow([accession, proteins[accession]])
    return data_dir


def make_model(author, device: torch.device):
    return author.TransKla(
        d_model=256, nhead=4, n_layers=6, d_emb=128, use_global=True,
        global_tokens=8, dropout=0.2, droppath=0.1,
    ).to(device)


def make_loader(author, data_dir: Path, part: str, batch_size: int, workers: int,
                augment: bool):
    protein_map = author.load_protein_map(str(data_dir / "proteins.csv"))
    if not protein_map:
        raise ValueError("TransKla full-protein context was disabled")
    return author.make_loader(
        str(data_dir / f"{part}.fasta"), protein_map, batch_size, augment,
        workers, "simple", 8, augment=augment, mask_prob=0.08 if augment else 0.0,
    )


def write_predictions(path: Path, rows, scores):
    if len(rows) != len(scores):
        raise ValueError("Prediction count differs from locked partition")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["accession", "position_1based", "label", "score"])
        for row, score in zip(rows, scores):
            writer.writerow([row["accession"], row["position_1based"], row["label"], float(score)])


def develop(args, author, partitions, proteins, manifest_hash, device):
    author.set_seed(args.seed)
    data_dir = write_inputs(args.output_dir, partitions, proteins, ("train", "validation"))
    train_loader = make_loader(author, data_dir, "train", args.batch_size, args.workers, True)
    val_loader = make_loader(author, data_dir, "validation", args.batch_size, args.workers, False)
    model = make_model(author, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    scheduler = author.WarmupCosineLR(optimizer, warmup_epochs=5,
                                      max_epochs=args.epochs, min_lr=1e-6)
    ema = author.ModelEMA(model, decay=0.995, device=device)
    best_auprc, selected_epoch, stale = -1.0, 0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        started = time.monotonic()
        train_loss, _, _ = author.run_epoch(model, train_loader, device,
                                             optimizer=optimizer, scaler=scaler, ema=ema)
        current = {key: value.detach().clone() for key, value in model.state_dict().items()}
        ema.load_to(model, device=device)
        val_loss, val_y, val_score = author.run_epoch(model, val_loader, device)
        model.load_state_dict(current)
        val_metrics = compute_binary_metrics(val_y, val_score)
        scheduler.step()
        if val_metrics["AUPRC"] > best_auprc + 0.002:
            best_auprc = val_metrics["AUPRC"]
            selected_epoch = epoch
            stale = 0
            torch.save({key: value.detach().cpu().clone()
                        for key, value in ema.shadow.items()}, args.output_dir / "transkla_best.pt")
        else:
            stale += 1
        record = {"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss,
                  "validation_metrics_fixed_0_5": val_metrics,
                  "elapsed_seconds": time.monotonic() - started}
        history.append(record)
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(f"epoch={epoch} val_AUPRC={val_metrics['AUPRC']:.5f} "
              f"best_epoch={selected_epoch} stale={stale} seconds={record['elapsed_seconds']:.1f}",
              flush=True)
        if stale >= args.patience:
            break
    if not selected_epoch:
        raise RuntimeError("No validation checkpoint selected")
    state = torch.load(args.output_dir / "transkla_best.pt", map_location=device,
                       weights_only=True)
    model.load_state_dict(state, strict=True)
    val_loss, val_y, val_score = author.run_epoch(model, val_loader, device)
    threshold, _ = author.best_mcc_threshold(val_y, val_score)
    report = {
        "model": "TransKla author architecture retrained on locked CLEAR-Lactyl",
        "author_url": AUTHOR_URL, "author_commit": AUTHOR_COMMIT,
        "author_source_sha256": AUTHOR_SOURCE_SHA256,
        "lock_sha256": sha256(args.lock), "manifest_sha256": manifest_hash,
        "proteome_sha256": sha256(args.proteome), "seed": args.seed,
        "training_rows": len(partitions["train"]),
        "validation_rows": len(partitions["validation"]),
        "selected_epoch": selected_epoch, "epochs_run": len(history),
        "selection_metric": "validation AUPRC", "selection_min_delta": 0.002,
        "validation_mcc_threshold": threshold,
        "validation_loss": val_loss,
        "validation_metrics_fixed_0_5": compute_binary_metrics(val_y, val_score),
        "validation_metrics_calibrated": compute_binary_metrics(val_y, val_score, threshold),
        "batch_size": args.batch_size, "max_epochs": args.epochs,
        "patience": args.patience, "global_context": "full reference protein, 8 chunks",
        "local_window": 41,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "test_scored": False,
    }
    write_predictions(args.output_dir / "validation_predictions.csv",
                      partitions["validation"], val_score)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected_epoch": selected_epoch,
                      "validation_metrics_fixed_0_5": report["validation_metrics_fixed_0_5"],
                      "validation_mcc_threshold": threshold}, indent=2), flush=True)


def heldout(args, author, partitions, proteins, manifest_hash, device):
    development_dir = args.development_dir
    report = json.loads((development_dir / "report.json").read_text())
    if (report["test_scored"] or report["author_source_sha256"] != AUTHOR_SOURCE_SHA256 or
            report["lock_sha256"] != sha256(args.lock) or
            report["manifest_sha256"] != manifest_hash or
            report["proteome_sha256"] != sha256(args.proteome)):
        raise ValueError("Development report does not match the locked test inputs")
    data_dir = write_inputs(args.output_dir, partitions, proteins, ("internal_test",))
    loader = make_loader(author, data_dir, "internal_test", report["batch_size"],
                         args.workers, False)
    model = make_model(author, device)
    state = torch.load(development_dir / "transkla_best.pt", map_location=device,
                       weights_only=True)
    model.load_state_dict(state, strict=True)
    loss, labels, scores = author.run_epoch(model, loader, device)
    threshold = float(report["validation_mcc_threshold"])
    result = {
        "model": report["model"], "seed": report["seed"],
        "lock_sha256": report["lock_sha256"],
        "development_report_sha256": sha256(development_dir / "report.json"),
        "checkpoint_sha256": sha256(development_dir / "transkla_best.pt"),
        "test_rows": len(partitions["internal_test"]), "test_loss": loss,
        "selected_epoch": report["selected_epoch"],
        "validation_metrics_fixed_0_5": report["validation_metrics_fixed_0_5"],
        "test_metrics_fixed_0_5": compute_binary_metrics(labels, scores),
        "validation_mcc_threshold": threshold,
        "test_metrics_calibrated": compute_binary_metrics(labels, scores, threshold),
    }
    write_predictions(args.output_dir / "test_predictions.csv",
                      partitions["internal_test"], scores)
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("develop", "heldout"), required=True)
    parser.add_argument("--author-source", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--proteome", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--development-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.mode == "heldout" and args.development_dir is None:
        parser.error("--development-dir is required for heldout mode")
    if min(args.epochs, args.patience, args.batch_size) < 1 or args.workers < 0:
        parser.error("Invalid training configuration")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    author = load_author(args.author_source)
    partitions, proteins, manifest_hash = load_lock(args.lock, args.proteome)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.mode == "develop":
        develop(args, author, partitions, proteins, manifest_hash, device)
    else:
        heldout(args, author, partitions, proteins, manifest_hash, device)


if __name__ == "__main__":
    main()
