"""Feature-engineering helpers, especially for text embedding columns."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack


def parse_vector_column(df: pd.DataFrame, column: str = "text_vector") -> pd.DataFrame:
    """Convert column of stringified vectors to np.ndarray (uses eval; prefer literal_eval later)."""
    def _to_array(value: Any) -> np.ndarray:
        return np.array(eval(value))  # noqa: S307

    df[column] = df[column].apply(_to_array)
    return df


def build_sparse_matrix_from_vector_column(df: pd.DataFrame, column: str) -> csr_matrix:
    """Stack rows of df[column] as csr_matrix (vstack of per-row csr)."""
    rows = [csr_matrix(row) for row in df[column]]
    return vstack(rows)


def attach_chiefcomplaint_embeddings(
    combined_df: pd.DataFrame,
    encoded_chiefcomplaints: Sequence[np.ndarray],
    column: str = "chiefcomplaint_vector",
) -> pd.DataFrame:
    """Attach pre-computed chief-complaint vectors to combined_df[column] as arrays."""
    combined_df[column] = list(encoded_chiefcomplaints)
    combined_df[column] = combined_df[column].apply(lambda x: np.array(x))
    return combined_df


def reduce_vector_column_average_pool(
    df: pd.DataFrame,
    column: str,
    out_prefix: str,
    n_slices: int = 3,
) -> pd.DataFrame:
    """Split vector dim into n_slices, mean each slice -> columns {out_prefix}1, {out_prefix}2, ..."""
    vectors = np.vstack(df[column].values)
    dim = vectors.shape[1]
    slice_size = dim // n_slices
    for i in range(n_slices):
        start = i * slice_size
        end = dim if i == n_slices - 1 else (i + 1) * slice_size
        df[f"{out_prefix}{i+1}"] = np.mean(vectors[:, start:end], axis=1)
    return df


def reduce_vector_column_pca(
    df: pd.DataFrame,
    column: str,
    n_components: int,
    out_prefix: str,
) -> pd.DataFrame:
    """PCA on stacked vectors; write components as {out_prefix} or {out_prefix}1,2,..."""
    from sklearn.decomposition import PCA

    vectors = np.vstack(df[column].values)
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(vectors)

    if n_components == 1:
        df[out_prefix] = reduced[:, 0]
    else:
        for i in range(n_components):
            df[f"{out_prefix}{i+1}"] = reduced[:, i]
    return df


