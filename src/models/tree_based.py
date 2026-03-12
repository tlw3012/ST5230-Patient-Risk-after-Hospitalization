"""Light-weight helpers for tree-based models (currently RandomForestClassifier)."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier


def create_random_forest_classifier(**kwargs: Any) -> RandomForestClassifier:
    """
    Create a :class:`RandomForestClassifier` with the provided
    hyperparameters.

    Notebooks can pass exactly the same keyword arguments they used
    previously; this helper simply centralises construction.
    """

    return RandomForestClassifier(**kwargs)


def fit_random_forest_classifier(
    X,
    y,
    **kwargs: Any,
) -> RandomForestClassifier:
    """
    Convenience helper that constructs and fits a random forest
    classifier in one call.

    Parameters
    ----------
    X, y
        Training features and labels.
    **kwargs
        Keyword arguments forwarded to
        :func:`create_random_forest_classifier`.
    """

    model = create_random_forest_classifier(**kwargs)
    model.fit(X, y)
    return model

