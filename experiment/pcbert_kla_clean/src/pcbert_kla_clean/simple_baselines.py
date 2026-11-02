from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from pcbert_kla_clean.data import KlaSplit
from pcbert_kla_clean.metrics import compute_binary_metrics


def feature_logistic_validation(
    development: KlaSplit, validation_indices: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit a feature-only control on Gao train and score Gao validation."""
    indices = np.asarray(validation_indices, dtype=np.int64)
    row_count = len(development.records)
    if (indices.ndim != 1 or not len(indices) or len(np.unique(indices)) != len(indices)
            or np.any(indices < 0) or np.any(indices >= row_count)):
        raise ValueError("Validation indices must be unique, nonempty, and in range")
    train_mask = np.ones(row_count, dtype=bool)
    train_mask[indices] = False
    if not np.any(train_mask):
        raise ValueError("No training rows remain")
    labels = development.labels
    if set(np.unique(labels[train_mask])) != {0, 1}:
        raise ValueError("Training rows must contain both classes")
    if set(np.unique(labels[indices])) != {0, 1}:
        raise ValueError("Validation rows must contain both classes")
    features = np.asarray(development.features)
    if features.shape != (row_count, len(development.feature_names)) or not np.isfinite(features).all():
        raise ValueError("Features must be finite and aligned with records")

    model = make_pipeline(
        MinMaxScaler(),
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000),
    )
    model.fit(features[train_mask], labels[train_mask])
    scores = model.predict_proba(features[indices])[:, 1]
    return scores, compute_binary_metrics(labels[indices], scores, threshold=0.5)
