"""Small helpers for wiring modeling experiments (train/test + metrics)."""

from __future__ import annotations

from typing import Any, Tuple

from src.config.settings import RANDOM_STATE, TEST_SIZE
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
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    compute_mse: bool = False,
) -> Tuple[Any, Any, Any, Any, Any, BinaryClassificationMetrics]:
    """Split X,y, fit model, print metrics; return X_train, X_test, y_train, y_test, model, metrics."""
    X_train, X_test, y_train, y_test = standard_train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            y_proba = proba[:, 1]

    metrics = print_binary_classification_summary(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        compute_mse=compute_mse,
    )

    return X_train, X_test, y_train, y_test, model, metrics

