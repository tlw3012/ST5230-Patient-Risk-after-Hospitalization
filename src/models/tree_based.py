"""Light-weight helpers for tree-based models (currently RandomForestClassifier)."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier


def create_random_forest_classifier(**kwargs: Any) -> RandomForestClassifier:
    """Build RandomForestClassifier with given kwargs."""
    return RandomForestClassifier(**kwargs)


def fit_random_forest_classifier(
    X,
    y,
    **kwargs: Any,
) -> RandomForestClassifier:
    """Create and fit RandomForestClassifier in one call."""
    model = create_random_forest_classifier(**kwargs)
    model.fit(X, y)
    return model

