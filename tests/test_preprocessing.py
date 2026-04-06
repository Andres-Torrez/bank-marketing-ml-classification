from pathlib import Path

from preprocessing.data_split import (
    load_dataset,
    prepare_features_target,
    split_data,
)
from preprocessing.pipeline import get_feature_types


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "bank-full.csv"

print(PROJECT_ROOT)
print(DATA_PATH)
print(DATA_PATH.exists())

def test_prepare_features_target_drops_duration():
    df = load_dataset(DATA_PATH)
    X, y = prepare_features_target(df, drop_leakage=True)

    assert "duration" not in X.columns
    assert "y" not in X.columns
    assert set(y.unique()) == {0, 1}


def test_split_data_is_stratified():
    df = load_dataset(DATA_PATH)
    X, y = prepare_features_target(df, drop_leakage=True)

    X_train, X_test, y_train, y_test = split_data(X, y)

    full_rate = y.mean()
    train_rate = y_train.mean()
    test_rate = y_test.mean()

    assert abs(full_rate - train_rate) < 0.01
    assert abs(full_rate - test_rate) < 0.01


def test_feature_type_detection():
    df = load_dataset(DATA_PATH)
    X, _ = prepare_features_target(df, drop_leakage=True)

    numeric_features, categorical_features = get_feature_types(X)

    assert "age" in numeric_features
    assert "balance" in numeric_features
    assert "job" in categorical_features
    assert "marital" in categorical_features