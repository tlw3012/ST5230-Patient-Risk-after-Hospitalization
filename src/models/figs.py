"""Helpers for FIGS models from the `imodels` package."""

from __future__ import annotations

from typing import Any

def create_figs_classifier(random_state: int = 5230, **kwargs: Any):
    """FIGSClassifier with default random_state=5230; pass other args via kwargs."""
    from imodels import FIGSClassifier
    return FIGSClassifier(random_state=random_state, **kwargs)

def create_figs_regressor(random_state: int = 5230, **kwargs: Any):
    """FIGSRegressor with default random_state=5230; pass other args via kwargs."""
    from imodels import FIGSRegressor
    return FIGSRegressor(random_state=random_state, **kwargs)
