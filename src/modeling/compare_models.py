from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"


SUMMARY_FILES = [
    "baseline_summary.json",
    "tree_models_summary.json",
    "boosting_models_summary.json",
]


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_model_rows(summary_data: Dict) -> List[Dict]:
    rows = []

    for model_name, model_data in summary_data.items():
        cv_metrics = model_data.get("cv_metrics", {})
        test_metrics = model_data.get("test_metrics", {})

        gap_roc_auc = cv_metrics.get("gap_roc_auc", None)
        overfitting_ok = gap_roc_auc is not None and gap_roc_auc <= 0.05

        row = {
            "model_name": model_name,
            "train_roc_auc": cv_metrics.get("train_roc_auc"),
            "cv_roc_auc": cv_metrics.get("cv_roc_auc"),
            "gap_roc_auc": gap_roc_auc,
            "overfitting_ok": overfitting_ok,
            "test_roc_auc": test_metrics.get("roc_auc"),
            "test_f1": test_metrics.get("f1"),
            "test_recall": test_metrics.get("recall"),
            "test_precision": test_metrics.get("precision"),
            "test_accuracy": test_metrics.get("accuracy"),
        }

        rows.append(row)

    return rows


def build_comparison_table() -> pd.DataFrame:
    all_rows = []

    for filename in SUMMARY_FILES:
        filepath = METRICS_DIR / filename
        if filepath.exists():
            summary_data = load_json(filepath)
            rows = extract_model_rows(summary_data)
            all_rows.extend(rows)

    comparison_df = pd.DataFrame(all_rows)

    if comparison_df.empty:
        raise ValueError("No model summary files were found or all were empty.")

    comparison_df = comparison_df.sort_values(
        by=["test_roc_auc", "cv_roc_auc"],
        ascending=False,
    ).reset_index(drop=True)

    return comparison_df


def save_outputs(comparison_df: pd.DataFrame) -> None:
    csv_path = METRICS_DIR / "model_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)

    print("Model comparison table saved to:")
    print(csv_path)
    print("\nTop models:")
    print(comparison_df.head())


def main() -> None:
    comparison_df = build_comparison_table()
    save_outputs(comparison_df)


if __name__ == "__main__":
    main()