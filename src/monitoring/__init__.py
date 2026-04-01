from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEEDBACK_DIR = PROJECT_ROOT / "data" / "feedback"
PREDICTIONS_LOG_PATH = FEEDBACK_DIR / "predictions_log.csv"


def ensure_feedback_dir() -> None:
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)


def log_prediction(user_input: dict, prediction: int, probability: float | None) -> None:
    """
    Save a new prediction event to predictions_log.csv
    """
    ensure_feedback_dir()

    row = {
        **user_input,
        "prediction": prediction,
        "prediction_label": "yes" if prediction == 1 else "no",
        "prediction_probability": probability,
        "timestamp": datetime.utcnow().isoformat(),
        "actual_label": None,
    }

    row_df = pd.DataFrame([row])

    if PREDICTIONS_LOG_PATH.exists():
        existing_df = pd.read_csv(PREDICTIONS_LOG_PATH)
        updated_df = pd.concat([existing_df, row_df], ignore_index=True)
    else:
        updated_df = row_df

    updated_df.to_csv(PREDICTIONS_LOG_PATH, index=False)