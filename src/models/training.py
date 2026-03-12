"""Generic training helpers (e.g., a standard train/test split)."""

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

