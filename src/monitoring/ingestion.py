from __future__ import annotations

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEEDBACK_DIR = PROJECT_ROOT / "data" / "feedback"
PREDICTIONS_LOG_PATH = FEEDBACK_DIR / "predictions_log.csv"
RETRAINING_DATASET_PATH = FEEDBACK_DIR / "retraining_dataset.csv"

TARGET_COL = "actual_label"

FEATURE_COLUMNS = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
]


def load_feedback_log() -> pd.DataFrame:
    if not PREDICTIONS_LOG_PATH.exists():
        raise FileNotFoundError(f"Feedback log not found: {PREDICTIONS_LOG_PATH}")
    return pd.read_csv(PREDICTIONS_LOG_PATH)


def build_retraining_dataset() -> pd.DataFrame:
    """
    Build retraining dataset using only rows with known actual_label.
    """
    df = load_feedback_log()

    if TARGET_COL not in df.columns:
        raise ValueError(f"Column '{TARGET_COL}' not found in feedback log.")

    retrain_df = df[df[TARGET_COL].notna()].copy()

    if retrain_df.empty:
        raise ValueError("No labeled feedback available yet for retraining dataset.")

    retrain_df = retrain_df[FEATURE_COLUMNS + [TARGET_COL]]

    retrain_df = retrain_df.rename(columns={TARGET_COL: "y"})

    retrain_df.to_csv(RETRAINING_DATASET_PATH, index=False)

    return retrain_df


def main() -> None:
    retrain_df = build_retraining_dataset()
    print("Retraining dataset created successfully.")
    print(f"Rows: {len(retrain_df)}")
    print(f"Saved to: {RETRAINING_DATASET_PATH}")


if __name__ == "__main__":
    main()