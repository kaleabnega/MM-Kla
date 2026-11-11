from __future__ import annotations

from argparse import Namespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from pcbert_kla_clean.metrics import compute_binary_metrics
from pcbert_kla_clean.model import (
    HybridTokenGatedPCBertKla,
    PCBertKla,
    TokenGatedConvConcatPCBertKla,
    TokenGatedPCBertKla,
    TokenGatedRevisedPCBertKla,
)


class PositiveWeightedBCELoss(nn.Module):
    def __init__(self, positive_weight: float):
        super().__init__()
        if not np.isfinite(positive_weight) or positive_weight <= 0:
            raise ValueError("Positive weight must be finite and greater than zero")
        self.positive_weight = float(positive_weight)

    def forward(self, probabilities: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        weights = torch.where(labels == 1, self.positive_weight, 1.0)
        return F.binary_cross_entropy(probabilities, labels, weight=weights)


def make_training_criterion(
    labels: np.ndarray,
    train_indices: np.ndarray,
    weighting: str,
) -> tuple[nn.Module, dict[str, float | int]]:
    training_labels = np.asarray(labels)[train_indices]
    positives = int(np.count_nonzero(training_labels == 1))
    negatives = int(np.count_nonzero(training_labels == 0))
    if positives + negatives != len(training_labels) or not len(training_labels):
        raise ValueError("Training labels must be nonempty and binary")
    if weighting == "none":
        positive_weight = 1.0
        criterion = nn.BCELoss()
    elif weighting == "train_ratio":
        if not positives or not negatives:
            raise ValueError("Ratio weighting requires both training classes")
        positive_weight = negatives / positives
        criterion = PositiveWeightedBCELoss(positive_weight)
    else:
        raise ValueError(f"Unsupported positive weighting: {weighting}")
    return criterion, {
        "training_positive_rows": positives,
        "training_negative_rows": negatives,
        "training_positive_weight": positive_weight,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler=None,
) -> float:
    model.train()
    losses: list[float] = []
    for batch in tqdm(loader, leave=False, desc="train"):
        encoded = {key: value.to(device) for key, value in batch["encoded"].items()}
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(features=features, **encoded)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(float(loss.detach().cpu()))

    return float(np.mean(losses))


def predict(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses: list[float] = []
    labels_all: list[float] = []
    scores_all: list[float] = []

    with torch.no_grad():
        for batch in tqdm(loader, leave=False, desc="eval"):
            encoded = {key: value.to(device) for key, value in batch["encoded"].items()}
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(features=features, **encoded)
            loss = criterion(outputs, labels)
            losses.append(float(loss.detach().cpu()))
            labels_all.extend(labels.detach().cpu().numpy().tolist())
            scores_all.extend(outputs.detach().cpu().numpy().tolist())

    return float(np.mean(losses)), np.asarray(labels_all), np.asarray(scores_all)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[float, dict[str, float]]:
    loss, labels, scores = predict(model, loader, criterion, device)
    metrics = compute_binary_metrics(labels, scores, threshold=threshold)
    return loss, metrics


def build_model(args: Namespace, feature_dim: int, device: torch.device) -> nn.Module:
    if args.architecture == "baseline":
        model = PCBertKla(
            model_name=args.model_name,
            feature_dim=feature_dim,
            encoder_layers=args.encoder_layers,
            freeze_encoder=args.freeze_encoder,
            cache_dir=args.cache_dir,
        )
    elif args.architecture == "token_gated":
        model = TokenGatedPCBertKla(
            model_name=args.model_name,
            feature_dim=feature_dim,
            encoder_layers=args.encoder_layers,
            fusion_dim=args.fusion_dim,
            attention_dim=args.attention_dim,
            dropout=args.arch_dropout,
            site_token_index=args.site_token_index,
            freeze_encoder=args.freeze_encoder,
            ablation=args.ablation,
            cache_dir=args.cache_dir,
            encoder_adaptation=getattr(args, "encoder_adaptation", "full"),
            lora_rank=getattr(args, "lora_rank", 8),
            lora_alpha=getattr(args, "lora_alpha", 32),
            lora_dropout=getattr(args, "lora_dropout", 0.3),
        )
    elif args.architecture == "token_gated_revised":
        if args.ablation != "none":
            raise ValueError("The revised model does not support --ablation")
        model = TokenGatedRevisedPCBertKla(
            model_name=args.model_name,
            feature_dim=feature_dim,
            encoder_layers=args.encoder_layers,
            fusion_dim=args.fusion_dim,
            attention_dim=args.attention_dim,
            dropout=args.arch_dropout,
            site_token_index=args.site_token_index,
            freeze_encoder=args.freeze_encoder,
            cache_dir=args.cache_dir,
        )
    elif args.architecture == "token_gated_conv_concat":
        if args.ablation != "none":
            raise ValueError("The convolution-concatenation model does not support --ablation")
        model = TokenGatedConvConcatPCBertKla(
            model_name=args.model_name,
            feature_dim=feature_dim,
            encoder_layers=args.encoder_layers,
            fusion_dim=args.fusion_dim,
            attention_dim=args.attention_dim,
            dropout=args.arch_dropout,
            site_token_index=args.site_token_index,
            freeze_encoder=args.freeze_encoder,
            cache_dir=args.cache_dir,
        )
    elif args.architecture == "hybrid_gated":
        model = HybridTokenGatedPCBertKla(
            model_name=args.model_name,
            feature_dim=feature_dim,
            encoder_layers=args.encoder_layers,
            fusion_dim=args.fusion_dim,
            attention_dim=args.attention_dim,
            dropout=args.arch_dropout,
            site_token_index=args.site_token_index,
            freeze_encoder=args.freeze_encoder,
            cache_dir=args.cache_dir,
        )
    else:
        raise ValueError(f"Unsupported architecture: {args.architecture}")
    return model.to(device)


def build_optimizer(args: Namespace, model: nn.Module):
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("No trainable parameters; check --freeze-encoder and architecture")

    if args.optimizer == "sgd":
        return torch.optim.SGD(
            trainable_parameters,
            lr=args.learning_rate,
            momentum=args.sgd_momentum,
            weight_decay=args.weight_decay,
        )
    if args.optimizer == "adamw":
        return torch.optim.AdamW(
            trainable_parameters,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def build_scheduler(args: Namespace, optimizer, steps_per_epoch: int):
    if args.scheduler == "none":
        return None
    if args.scheduler == "linear":
        from transformers import get_linear_schedule_with_warmup

        total_steps = max(1, steps_per_epoch * args.epochs)
        warmup_steps = int(total_steps * args.warmup_ratio)
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    raise ValueError(f"Unsupported scheduler: {args.scheduler}")
