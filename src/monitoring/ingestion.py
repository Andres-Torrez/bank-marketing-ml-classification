from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.monitoring.database import get_connection, initialize_database


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEEDBACK_DIR = PROJECT_ROOT / "data" / "feedback"
RETRAINING_DATASET_PATH = FEEDBACK_DIR / "retraining_dataset.csv"

TARGET_COL = "actual_label"

FEATURE_COLUMNS = [
    "age",
    "job",
    "marital",
    "education",
    "default_status",
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


def load_feedback_from_db() -> pd.DataFrame:
    initialize_database()
    conn = get_connection()

    query = "SELECT * FROM prediction_logs"
    df = pd.read_sql_query(query, conn)

    conn.close()
    return df


def build_retraining_dataset() -> pd.DataFrame:
    """
    Build retraining dataset using only rows with known actual_label.
    """
    df = load_feedback_from_db()

    if TARGET_COL not in df.columns:
        raise ValueError(f"Column '{TARGET_COL}' not found in database records.")

    retrain_df = df[df[TARGET_COL].notna()].copy()

    if retrain_df.empty:
        raise ValueError("No labeled feedback available yet for retraining dataset.")

    retrain_df = retrain_df[FEATURE_COLUMNS + [TARGET_COL]]

    retrain_df = retrain_df.rename(
        columns={
            "default_status": "default",
            TARGET_COL: "y",
        }
    )

    retrain_df.to_csv(RETRAINING_DATASET_PATH, index=False)

    return retrain_df


def main() -> None:
    retrain_df = build_retraining_dataset()
    print("Retraining dataset created successfully from SQLite database.")
    print(f"Rows: {len(retrain_df)}")
    print(f"Saved to: {RETRAINING_DATASET_PATH}")


if __name__ == "__main__":
    main()