"""Small convenience wrappers around sklearn's LogisticRegression."""

from __future__ import annotations

from typing import Any

from sklearn.linear_model import LogisticRegression


def create_logistic_regression(**kwargs: Any) -> LogisticRegression:
    """Build LogisticRegression with given kwargs."""
    return LogisticRegression(**kwargs)


def fit_logistic_regression(
    X,
    y,
    **kwargs: Any,
) -> LogisticRegression:
    """Create and fit LogisticRegression in one call."""
    model = create_logistic_regression(**kwargs)
    model.fit(X, y)
    return model

