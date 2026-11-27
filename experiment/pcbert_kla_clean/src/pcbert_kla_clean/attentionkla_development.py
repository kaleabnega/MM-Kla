"""Grouped CLEAR-Lactyl validation of freshly trained AttentionKla architecture.

The architecture follows the authors' ESM-2 sequence classifier with LoRA on
query/key/value projections. Their Kla-trained adapters are never loaded.
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pcbert_kla_clean.clear_lactyl import load_clear_lactyl
from pcbert_kla_clean.metrics import compute_binary_metrics
from pcbert_kla_clean.source_study import sha256
from pcbert_kla_clean.stacked_development import grouped_splits, read_training_groups
from pcbert_kla_clean.utils import set_seed


BASE_MODEL = "facebook/esm2_t33_650M_UR50D"
BASE_REVISION = "08e4846e537177426273712802403f7ba8261b6c"
AUTHOR_SOURCE_COMMIT = "024c5fa77308d64363be61e6767ad5c7c7e5742c"
MAX_LENGTH = 53
EFFECTIVE_BATCH = 64
MICROBATCH = 8
EVAL_BATCH = 32
LEARNING_RATE = 1e-3
LORA_RANK = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.3


def validate_tokenized_windows(tokenizer, windows: list[str], input_ids: torch.Tensor,
                               attention_mask: torch.Tensor) -> None:
    if len(windows) != len(input_ids) or input_ids.shape != attention_mask.shape:
        raise ValueError("Tokenized arrays do not match the sequence rows")
    if input_ids.shape[1] != MAX_LENGTH or not torch.all(attention_mask.sum(dim=1) == MAX_LENGTH):
        raise ValueError("Every 51-residue window must produce 53 ESM-2 tokens")
    lysine_id = tokenizer.convert_tokens_to_ids("K")
    if not torch.all(input_ids[:, 26] == lysine_id):
        raise ValueError("The central lysine must remain at ESM-2 token position 26")


def average_fold_logits(fold_logits: list[np.ndarray]) -> np.ndarray:
    stacked = np.stack(fold_logits, axis=0)
    if stacked.ndim != 3 or stacked.shape[2] != 2 or not np.isfinite(stacked).all():
        raise ValueError("Expected finite two-class logits from each fold")
    return stacked.mean(axis=0)


def positive_probabilities(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    return probabilities[:, 1] / probabilities.sum(axis=1)


def build_model(device: torch.device):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification

    base = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, num_labels=2, dtype=torch.bfloat16,
        local_files_only=True,
    )
    config = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=LORA_RANK, lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT, target_modules=["query", "key", "value"],
        bias="none",
    )
    model = get_peft_model(base, config).to(device)
    if not any("lora_" in name and p.requires_grad for name, p in model.named_parameters()):
        raise RuntimeError("LoRA parameters are not trainable")
    if not any("classifier" in name and p.requires_grad for name, p in model.named_parameters()):
        raise RuntimeError("Sequence classification head is not trainable")
    return model


def predict_logits(model, token_ids: torch.Tensor, attention_mask: torch.Tensor,
                   indices: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    logits = []
    loader = DataLoader(TensorDataset(torch.as_tensor(indices)), batch_size=EVAL_BATCH)
    with torch.no_grad():
        for (batch_indices,) in loader:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(input_ids=token_ids[batch_indices].to(device),
                               attention_mask=attention_mask[batch_indices].to(device))
            logits.append(output.logits.float().cpu().numpy())
    return np.concatenate(logits, axis=0)


def train_fold(fold_number: int, train_idx: np.ndarray, stop_idx: np.ndarray,
               validation_idx: np.ndarray, labels: np.ndarray,
               token_ids: torch.Tensor, attention_mask: torch.Tensor,
               output_dir: Path, device: torch.device, seed: int,
               max_epochs: int, patience: int) -> tuple[np.ndarray, dict]:
    fold_seed = seed + 100 * fold_number
    set_seed(fold_seed)
    model = build_model(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=LEARNING_RATE, weight_decay=0.0)
    updates_per_epoch = math.ceil(len(train_idx) / EFFECTIVE_BATCH)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs * updates_per_epoch,
    )
    loader = DataLoader(TensorDataset(torch.as_tensor(train_idx)),
                        batch_size=EFFECTIVE_BATCH, shuffle=True,
                        generator=torch.Generator().manual_seed(fold_seed))
    best_epoch, best_auprc, stale = 0, -1.0, 0
    best_state = None
    history = []
    start = time.monotonic()
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        for (batch_indices,) in loader:
            optimizer.zero_grad(set_to_none=True)
            for micro in batch_indices.split(MICROBATCH):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(input_ids=token_ids[micro].to(device),
                                   attention_mask=attention_mask[micro].to(device))
                    targets = torch.as_tensor(labels[micro.tolist()], device=device)
                    loss = nn.functional.cross_entropy(output.logits.float(), targets,
                                                       reduction="sum")
                (loss / len(batch_indices)).backward()
                total_loss += float(loss.detach())
            optimizer.step()
            scheduler.step()
        stop_logits = predict_logits(model, token_ids, attention_mask, stop_idx, device)
        auprc = float(average_precision_score(labels[stop_idx],
                                               positive_probabilities(stop_logits)))
        if auprc > best_auprc:
            best_epoch, best_auprc, stale = epoch, auprc, 0
            best_state = {name: parameter.detach().cpu().clone()
                          for name, parameter in model.named_parameters()
                          if parameter.requires_grad}
        else:
            stale += 1
        entry = {"epoch": epoch, "train_loss": total_loss / len(train_idx),
                 "selection_auprc": auprc, "learning_rate": scheduler.get_last_lr()[0]}
        history.append(entry)
        print(f"AttentionKla fold={fold_number} epoch={epoch} "
              f"train_loss={entry['train_loss']:.5f} selection_AUPRC={auprc:.5f}",
              flush=True)
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("No selected AttentionKla checkpoint")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in best_state:
                parameter.copy_(best_state[name].to(device))
    fold_dir = output_dir / f"fold_{fold_number}"
    fold_dir.mkdir(parents=True)
    model.save_pretrained(fold_dir / "adapter")
    fold_logits = predict_logits(model, token_ids, attention_mask, validation_idx, device)
    report = {
        "fold": fold_number, "seed": fold_seed,
        "training_rows": len(train_idx), "selection_rows": len(stop_idx),
        "selected_epoch": best_epoch, "selection_auprc": best_auprc,
        "epochs_run": len(history), "wall_seconds": time.monotonic() - start,
        "history": history,
    }
    (fold_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    del model, optimizer, scheduler, best_state
    torch.cuda.empty_cache()
    return fold_logits, report


def run(lock: Path, output_dir: Path, device: torch.device, seed: int = 42,
        max_epochs: int = 15, patience: int = 2) -> dict:
    if device.type != "cuda" or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This AttentionKla training run requires a BF16-capable CUDA GPU")
    if seed < 0 or max_epochs < 1 or patience < 1:
        raise ValueError("Invalid run configuration")
    from transformers import AutoTokenizer

    split, validation_idx, _ = load_clear_lactyl(lock, add_cksaap=False)
    train_count = int(validation_idx[0])
    labels = split.labels.astype(np.int64)
    groups = read_training_groups(lock, split, train_count)
    folds = grouped_splits(labels[:train_count], groups, seed)
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, local_files_only=True,
    )
    encoded = tokenizer(split.sequences, padding="max_length", truncation=True,
                        max_length=MAX_LENGTH, return_tensors="pt")
    token_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    validate_tokenized_windows(tokenizer, split.sequences, token_ids, attention_mask)
    output_dir.mkdir(parents=True, exist_ok=False)
    tokenizer.save_pretrained(output_dir / "tokenizer")
    logits_by_fold = []
    fold_reports = []
    start = time.monotonic()
    for fold_number, (outer_train, stop_idx) in enumerate(folds, 1):
        print(f"Training AttentionKla fold {fold_number}/{len(folds)}", flush=True)
        fold_logits, report = train_fold(
            fold_number, outer_train, stop_idx, validation_idx, labels,
            token_ids, attention_mask, output_dir, device, seed, max_epochs, patience,
        )
        logits_by_fold.append(fold_logits)
        fold_reports.append(report)
        np.save(output_dir / f"fold_{fold_number}" / "validation_logits.npy", fold_logits)
    combined_logits = average_fold_logits(logits_by_fold)
    probabilities = positive_probabilities(combined_logits)
    metrics = compute_binary_metrics(labels[validation_idx], probabilities, 0.5)
    with (output_dir / "validation_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("accession", "position_1based", "label", "attentionkla_score"))
        for row_number, index in enumerate(validation_idx):
            record = split.records[int(index)]
            writer.writerow((record.name, record.position, record.label,
                             float(probabilities[row_number])))
    report = {
        "model": "AttentionKla architecture retrained on locked CLEAR-Lactyl",
        "author_source": "https://github.com/lee190632/AttentionKla",
        "author_source_commit": AUTHOR_SOURCE_COMMIT,
        "base_model": BASE_MODEL, "base_revision": BASE_REVISION,
        "kla_trained_adapters_loaded": False,
        "lock_sha256": sha256(lock), "seed": seed,
        "training_rows": train_count, "validation_rows": len(validation_idx),
        "internal_test_scored": False,
        "protocol": "five protein-grouped training folds; selection by held-fold AUPRC; mean validation logits",
        "max_length": MAX_LENGTH, "effective_batch": EFFECTIVE_BATCH,
        "microbatch": MICROBATCH, "learning_rate": LEARNING_RATE,
        "optimizer": "AdamW", "scheduler": "cosine", "weight_decay": 0.0,
        "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT, "lora_targets": ["query", "key", "value"],
        "max_epochs": max_epochs, "patience": patience,
        "folds": fold_reports, "validation_metrics": metrics,
        "wall_seconds": time.monotonic() - start,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
