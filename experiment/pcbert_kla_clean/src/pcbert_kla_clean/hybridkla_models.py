"""Independent PyTorch implementations of HybridKla-style prediction branches."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pcbert_kla_clean.hybridkla_features import FEATURE_DIMS, STANDARD_AA


FEATURE_HEADS = {
    "ACF": ((1024, 512), 0.3, True),
    "AAINDEX": ((512, 256), 0.4, True),
    "CKSAAP": ((768, 256), 0.4, True),
    "GPS": ((512, 128), 0.3, False),
    "OBC": ((2048, 512), 0.5, True),
    "PSEAAC": ((64, 32), 0.2, False),
}
SEQUENCE_AA = "KLAEVGSDITR*PQNFYMHCWB"


class FeatureDNN(nn.Module):
    def __init__(self, feature_name: str):
        super().__init__()
        hidden_sizes, dropout, use_batch_norm = FEATURE_HEADS[feature_name]
        layers = []
        input_size = FEATURE_DIMS[feature_name]
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.extend((nn.ReLU(), nn.Dropout(dropout)))
            input_size = hidden_size
        layers.append(nn.Linear(input_size, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features).squeeze(-1)


class SequenceLSTM(nn.Module):
    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 256,
                 layers: int = 6, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(len(SEQUENCE_AA), embedding_dim)
        self.position = nn.Embedding(51, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(),
                                        nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        embedded = self.embedding(tokens) + self.position(positions)[None, :, :]
        states, _ = self.lstm(embedded)
        return self.classifier(states[:, -1]).squeeze(-1)


class EsmEmbeddingHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(hidden_size, 512), nn.LayerNorm(512),
                                    nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 1))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.layers(embeddings).squeeze(-1)


class MetaDNN(nn.Module):
    def __init__(self, input_size: int = 8):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_size, 128), nn.Linear(128, 64),
                                    nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 32),
                                    nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 1))

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        return self.layers(scores).squeeze(-1)


def encode_lstm_sequences(sequences: list[str]) -> np.ndarray:
    if not sequences or any(len(s) != 51 or s[25] != "K" or set(s) - set(STANDARD_AA)
                            for s in sequences):
        raise ValueError("LSTM requires canonical 51-residue K-centered windows")
    index = {aa: i for i, aa in enumerate(SEQUENCE_AA)}
    return np.asarray([[index[aa] for aa in seq] for seq in sequences], dtype=np.int64)


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Ranking AP used only to select checkpoints on the locked validation set."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or labels.ndim != 1 or len(labels) == 0:
        raise ValueError("Labels and scores must be nonempty aligned vectors")
    positives = int(labels.sum())
    if positives == 0 or positives == len(labels) or not np.isfinite(scores).all():
        raise ValueError("Validation needs both classes and finite scores")
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    group_ends = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), len(scores) - 1]
    true_positives = np.cumsum(sorted_labels)[group_ends]
    precision = true_positives / (group_ends + 1)
    recall_increments = np.diff(np.r_[0, true_positives]) / positives
    return float(np.dot(precision, recall_increments))


def predict_scores(model: nn.Module, values: np.ndarray, device: torch.device,
                   batch_size: int = 1024) -> np.ndarray:
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            batch = torch.as_tensor(values[start:start + batch_size], device=device)
            predictions.append(torch.sigmoid(model(batch)).float().cpu().numpy())
    return np.concatenate(predictions).astype(np.float32)


@dataclass
class TrainingResult:
    model: nn.Module
    selected_epoch: int
    validation_auprc: float


def fit_predictor(
    model: nn.Module, train_values: np.ndarray, train_labels: np.ndarray,
    validation_values: np.ndarray, validation_labels: np.ndarray,
    *, device: torch.device, epochs: int, batch_size: int,
    learning_rate: float, patience: int, seed: int,
) -> TrainingResult:
    if len(train_values) != len(train_labels) or len(validation_values) != len(validation_labels):
        raise ValueError("Feature and label rows differ")
    if len(train_values) < 2 or batch_size < 2 or epochs < 1 or patience < 0:
        raise ValueError("Invalid training size or hyperparameters")
    if not set(np.unique(train_labels)).issubset({0, 1}):
        raise ValueError("Training labels must be binary")
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    x = torch.as_tensor(train_values)
    y = torch.as_tensor(train_labels, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True,
                        generator=generator, drop_last=len(x) % batch_size == 1)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    best_state = None
    best_score = -1.0
    selected_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
        scores = predict_scores(model, validation_values, device,
                                batch_size=max(batch_size, 32))
        auprc = average_precision(validation_labels, scores)
        if auprc > best_score:
            best_score = auprc
            best_state = copy.deepcopy(model.state_dict())
            selected_epoch = epoch
        if patience and epoch - selected_epoch >= patience:
            break
    if best_state is None:
        raise RuntimeError("No finite checkpoint was selected")
    model.load_state_dict(best_state)
    return TrainingResult(model=model, selected_epoch=selected_epoch,
                          validation_auprc=best_score)
