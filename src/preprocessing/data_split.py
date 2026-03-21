from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_COL = "y"
LEAKAGE_COLS = ["duration"]


def load_dataset(data_path: str | Path) -> pd.DataFrame:
    """Load Bank Marketing dataset."""
    data_path = Path(data_path)
    df = pd.read_csv(data_path, sep=";")
    return df


def prepare_features_target(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    drop_leakage: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split dataframe into features and target.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    target_col : str
        Target column name.
    drop_leakage : bool
        Whether to drop leakage-related columns such as 'duration'.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Binary target encoded as 0/1.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    X = df.drop(columns=[target_col]).copy()

    if drop_leakage:
        cols_to_drop = [col for col in LEAKAGE_COLS if col in X.columns]
        X = X.drop(columns=cols_to_drop)

    y = df[target_col].map({"no": 0, "yes": 1})

    if y.isnull().any():
        raise ValueError("Target contains unexpected values. Expected only 'yes'/'no'.")

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform stratified train/test split.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test