"""Generic training helpers (e.g., a standard train/test split)."""

from __future__ import annotations

from typing import Any, Tuple

from sklearn.model_selection import train_test_split

from src.config.settings import RANDOM_STATE, TEST_SIZE


def standard_train_test_split(
    X,
    y,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    stratify: Any = None,
) -> Tuple[Any, Any, Any, Any]:
    """Train/test split with project defaults from config.settings."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

