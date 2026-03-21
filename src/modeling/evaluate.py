from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true,
    y_pred,
    y_proba,
) -> Dict[str, float]:
    """
    Compute core classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted class labels.
    y_proba : array-like
        Predicted probabilities for positive class.

    Returns
    -------
    dict
        Dictionary with evaluation metrics.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }
    return metrics


def compute_overfitting_gap(train_score: float, cv_score: float) -> float:
    """
    Compute overfitting gap as train - validation.
    """
    return float(train_score - cv_score)