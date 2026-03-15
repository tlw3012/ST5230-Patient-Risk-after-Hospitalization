"""Selection and sampling utilities for admissions / patients."""

from __future__ import annotations

from typing import Iterable, Set

import pandas as pd

def get_common_ids(
    *dfs: pd.DataFrame,
    column: str = "hadm_id",
) -> Set[int]:
    """Intersection of id column across all given DataFrames."""
    if not dfs:
        return set()
    out = set(dfs[0][column].dropna().astype(int))
    for df in dfs[1:]:
        out &= set(df[column].dropna().astype(int))
    return out

def filter_by_ids(
    df: pd.DataFrame,
    ids: Iterable[int],
    column: str = "hadm_id",
) -> pd.DataFrame:
    """Keep rows whose column value is in ids."""
    id_set = set(ids)
    return df[df[column].astype(int).isin(id_set)].copy()
