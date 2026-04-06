import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics" / "boosting" / "gradient_boosting_tuned_metrics.json"


def load_metrics():
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_final_model_meets_minimum_roc_auc():
    metrics = load_metrics()
    assert metrics["test_metrics"]["roc_auc"] >= 0.78


def test_final_model_controls_overfitting():
    metrics = load_metrics()
    assert metrics["gap_roc_auc"] <= 0.05


def test_final_model_has_minimum_recall():
    metrics = load_metrics()
    assert metrics["test_metrics"]["recall"] >= 0.20