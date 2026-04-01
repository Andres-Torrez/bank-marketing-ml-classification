from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.modeling.evaluate import (
    compute_classification_metrics,
    compute_confusion_matrix_dict,
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
N_ITER = 20

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data"/ "raw"/ "bank-full.csv"
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"


def ensure_directories() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)


def build_search_pipeline(X_train):
    preprocessor = build_preprocessor(X_train)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]
    )
    return pipeline


def get_param_distributions():
    return {
        "model__n_estimators": [100, 150, 200, 250, 300],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__max_depth": [2, 3, 4, 5],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__min_samples_leaf": [1, 2, 5, 10],
        "model__subsample": [0.7, 0.8, 0.9, 1.0],
    }


def evaluate_on_test(model_pipeline, X_test, y_test):
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


def main() -> None:
    ensure_directories()

    df = load_dataset(DATA_PATH)
    X, y = prepare_features_target(df, target_col=TARGET_COL, drop_leakage=True)
    X_train, X_test, y_train, y_test = split_data(X, y)

    pipeline = build_search_pipeline(X_train)

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=get_param_distributions(),
        n_iter=N_ITER,
        scoring="roc_auc",
        cv=cv,
        verbose=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        return_train_score=True,
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_params = search.best_params_
    best_cv_score = float(search.best_score_)

    train_best_index = search.best_index_
    train_roc_auc = float(search.cv_results_["mean_train_score"][train_best_index])
    gap_roc_auc = float(train_roc_auc - best_cv_score)

    test_results = evaluate_on_test(best_model, X_test, y_test)

    results = {
        "model_name": "gradient_boosting_tuned",
        "drop_leakage": True,
        "dropped_columns": ["duration"],
        "search_type": "RandomizedSearchCV",
        "best_params": best_params,
        "train_roc_auc": train_roc_auc,
        "cv_roc_auc": best_cv_score,
        "gap_roc_auc": gap_roc_auc,
        "overfitting_ok": gap_roc_auc <= 0.05,
        "test_metrics": test_results["test_metrics"],
        "confusion_matrix": test_results["confusion_matrix"],
        "n_iter": N_ITER,
        "n_splits": N_SPLITS,
        "random_state": RANDOM_STATE,
    }

    model_path = MODELS_DIR / "gradient_boosting_tuned.joblib"
    metrics_path = METRICS_DIR / "gradient_boosting_tuned_metrics.json"

    joblib.dump(best_model, model_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print("Tuning completed.")
    print(f"Best params: {best_params}")
    print(f"Best CV ROC-AUC: {best_cv_score:.4f}")
    print(f"Train ROC-AUC: {train_roc_auc:.4f}")
    print(f"Overfitting gap: {gap_roc_auc:.4f}")
    print(f"Test ROC-AUC: {test_results['test_metrics']['roc_auc']:.4f}")
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()