"""
Logistic regression model helpers.

This module encapsulates small convenience wrappers around
``sklearn.linear_model.LogisticRegression`` used by multiple notebooks.
The intent is *not* to hide hyperparameters, but to avoid repeating
boilerplate when constructing and fitting models.
"""

from __future__ import annotations

from typing import Any

from sklearn.linear_model import LogisticRegression


def create_logistic_regression(**kwargs: Any) -> LogisticRegression:
    """
    Create a :class:`LogisticRegression` instance with the provided
    keyword arguments.

    Notebooks can pass exactly the same hyperparameters they used
    previously, while centralising the constructor location so that
    future tweaks are easier to audit.
    """

    return LogisticRegression(**kwargs)


def fit_logistic_regression(
    X,
    y,
    **kwargs: Any,
) -> LogisticRegression:
    """
    Convenience helper that constructs and fits a logistic regression
    model in one call.

    Parameters
    ----------
    X, y
        Training features and labels.
    **kwargs
        Keyword arguments forwarded to :func:`create_logistic_regression`.
    """

    model = create_logistic_regression(**kwargs)
    model.fit(X, y)
    return model

