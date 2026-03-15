"""Helpers for constructing supervised-learning labels (death, severity, mortality, acuity, etc.)."""

from __future__ import annotations

import pandas as pd

def ensure_binary(series: pd.Series) -> pd.Series:
    """Map to 0/1. Accepts numeric 0/1 or boolean; otherwise cast to int and clip."""
    s = pd.Series(series).astype(int)
    return (s.clip(0, 1)).astype(int)

def ensure_categorical(series: pd.Series) -> pd.Series:
    """Convert to pandas Categorical; useful for ordinal/multiclass labels."""
    return series.astype("category")
