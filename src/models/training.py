"""
Generic training helpers used in CP* and P* notebooks.

This module centralises low-risk pieces of model-training boilerplate
such as the standard train/test split configuration. Model-specific
construction lives in modules like ``logistic`` and ``tree_based``.
"""

from __future__ import annotations

from typing import Any, Tuple

from sklearn.model_selection import train_test_split


def standard_train_test_split(
    X,
    y,
    *,
    test_size: float = 0.2,
    random_state: int = 5230,
    stratify: Any = None,
) -> Tuple[Any, Any, Any, Any]:
    """
    Perform a train/test split with the defaults used throughout
    the notebooks.

    This is a thin wrapper around :func:`sklearn.model_selection.
    train_test_split` that simply encodes the project-wide defaults
    (``test_size=0.2``, ``random_state=5230``).
    """

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

