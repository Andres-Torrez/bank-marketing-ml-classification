from __future__ import annotations

from typing import Dict

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
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


def compute_confusion_matrix_dict(y_true, y_pred) -> Dict[str, int]:
    """
    Return confusion matrix as a serializable dictionary.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }