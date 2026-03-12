"""Small helpers for wiring modeling experiments (train/test + metrics)."""

from __future__ import annotations

from typing import Any, Tuple

from src.models.evaluation import (
    BinaryClassificationMetrics,
    print_binary_classification_summary,
)
from src.models.training import standard_train_test_split


def run_binary_experiment(
    model,
    X,
    y,
    *,
    test_size: float = 0.2,
    random_state: int = 5230,
    compute_mse: bool = False,
) -> Tuple[Any, Any, Any, Any, Any, BinaryClassificationMetrics]:
    """
    Run a single binary-classification experiment:

    - split ``X, y`` into train/test
    - fit the provided model
    - compute standard metrics (accuracy, AU-ROC, optional MSE)
    - print a summary

    This is a thin wrapper around logic that previously lived in
    multiple notebooks; it does **not** make any assumptions about how
    features or labels are constructed.
    """

    X_train, X_test, y_train, y_test = standard_train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    # Many classifiers used in the notebooks expose predict_proba.
    y_proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        # Assume binary classification when 2 columns are present.
        if proba.ndim == 2 and proba.shape[1] >= 2:
            y_proba = proba[:, 1]

    metrics = print_binary_classification_summary(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        compute_mse=compute_mse,
    )

    return X_train, X_test, y_train, y_test, model, metrics

