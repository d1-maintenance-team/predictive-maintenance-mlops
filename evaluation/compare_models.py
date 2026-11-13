"""Compare binary predictive-maintenance model predictions."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def compare_models(
    labels: np.ndarray,
    predictions: Mapping[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Return common metrics for each model's probability predictions."""
    target = np.asarray(labels)
    results: dict[str, dict[str, float]] = {}
    for name, probabilities in predictions.items():
        scores = np.asarray(probabilities)
        if len(scores) != len(target):
            raise ValueError(f"Prediction length mismatch for model: {name}")
        classes = (scores >= 0.5).astype(int)
        results[name] = {
            "accuracy": float(accuracy_score(target, classes)),
            "f1": float(f1_score(target, classes, zero_division=0)),
            "roc_auc": float(roc_auc_score(target, scores)),
        }
    return results

