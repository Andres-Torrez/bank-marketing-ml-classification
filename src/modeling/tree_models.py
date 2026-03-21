from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.modeling.evaluate import (
    compute_classification_metrics,
    compute_confusion_matrix_dict,
    compute_overfitting_gap,
)
from src.preprocessing.data_split import (
    TARGET_COL,
    load_dataset,
    prepare_features_target,
    split_data,
)
from src.preprocessing.pipeline import build_preprocessor


RANDOM_STATE = 42
N_SPLITS = 5

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw"/ "bank-full.csv"
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"


def ensure_directories() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)


def get_tree_models() -> Dict[str, Any]:
    """
    Return tree-based candidate models.
    """
    return {
        "decision_tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            max_depth=6,
            min_samples_split=20,
            min_samples_leaf=10,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def evaluate_model_cv(model_pipeline: Pipeline, X_train, y_train) -> Dict[str, float]:
    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    cv_results = cross_validate(
        estimator=model_pipeline,
        X=X_train,
        y=y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1,
    )

    metrics = {}

    for metric_name in scoring:
        train_mean = float(np.mean(cv_results[f"train_{metric_name}"]))
        val_mean = float(np.mean(cv_results[f"test_{metric_name}"]))
        gap = compute_overfitting_gap(train_mean, val_mean)

        metrics[f"train_{metric_name}"] = train_mean
        metrics[f"cv_{metric_name}"] = val_mean
        metrics[f"gap_{metric_name}"] = gap

    return metrics


def fit_final_model(model_pipeline: Pipeline, X_train, y_train) -> Pipeline:
    model_pipeline.fit(X_train, y_train)
    return model_pipeline


def evaluate_on_test(model_pipeline: Pipeline, X_test, y_test) -> Dict[str, Any]:
    y_pred = model_pipeline.predict(X_test)

    if hasattr(model_pipeline, "predict_proba"):
        y_proba = model_pipeline.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred

    test_metrics = compute_classification_metrics(y_test, y_pred, y_proba)
    conf_matrix = compute_confusion_matrix_dict(y_test, y_pred)

    return {
        "test_metrics": test_metrics,
        "confusion_matrix": conf_matrix,
    }


def save_metrics(metrics: Dict[str, Any], filename: str) -> None:
    output_path = METRICS_DIR / filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)


def save_model(model: Pipeline, filename: str) -> None:
    output_path = MODELS_DIR / filename
    joblib.dump(model, output_path)


def main() -> None:
    ensure_directories()

    df = load_dataset(DATA_PATH)
    X, y = prepare_features_target(df, target_col=TARGET_COL, drop_leakage=True)
    X_train, X_test, y_train, y_test = split_data(X, y)

    preprocessor = build_preprocessor(X_train)
    candidate_models = get_tree_models()

    all_results = {}

    for model_name, estimator in candidate_models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )

        cv_metrics = evaluate_model_cv(pipeline, X_train, y_train)
        fitted_pipeline = fit_final_model(pipeline, X_train, y_train)
        test_results = evaluate_on_test(fitted_pipeline, X_test, y_test)

        all_results[model_name] = {
            "model_name": model_name,
            "drop_leakage": True,
            "dropped_columns": ["duration"],
            "cv_metrics": cv_metrics,
            "test_metrics": test_results["test_metrics"],
            "confusion_matrix": test_results["confusion_matrix"],
            "test_size": 0.20,
            "random_state": RANDOM_STATE,
            "n_splits": N_SPLITS,
        }

        save_metrics(
            all_results[model_name],
            filename=f"{model_name}_metrics.json",
        )
        save_model(
            fitted_pipeline,
            filename=f"{model_name}.joblib",
        )

    summary_path = METRICS_DIR / "tree_models_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)

    print("Tree-based model training and evaluation completed.")
    print(f"Saved metrics to: {METRICS_DIR}")
    print(f"Saved models to: {MODELS_DIR}")


if __name__ == "__main__":
    main()