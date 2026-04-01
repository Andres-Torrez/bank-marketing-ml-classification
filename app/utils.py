from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "gradient_boosting.joblib"


def load_model():
    """Load the final trained model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def build_input_dataframe(user_input: dict) -> pd.DataFrame:
    """Convert user input dictionary into a single-row DataFrame."""
    return pd.DataFrame([user_input])