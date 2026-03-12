"""Feature-engineering helpers, especially for text embedding columns."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack


def parse_vector_column(df: pd.DataFrame, column: str = "text_vector") -> pd.DataFrame:
    """
    Convert a column of stringified vectors into NumPy arrays.

    In the notebooks, text embeddings were sometimes stored as string
    representations of Python lists and then recovered with
    ``np.array(eval(x))``. This helper centralizes that pattern so it
    can be more easily audited later.

    Parameters
    ----------
    df:
        Input DataFrame containing the column to parse.
    column:
        Name of the column whose elements should be converted to
        ``np.ndarray`` instances.

    Returns
    -------
    pd.DataFrame
        The same DataFrame instance, with ``df[column]`` updated.

    Notes
    -----
    - This implementation intentionally mirrors the existing notebook
      behavior and therefore still uses ``eval``. A future refactor
      should replace this with a safer parser such as
      ``ast.literal_eval`` once all stored formats are confirmed.
    """

    def _to_array(value: Any) -> np.ndarray:
        # Preserve current semantics exactly: np.array(eval(...)).
        return np.array(eval(value))  # noqa: S307  # TODO: replace eval with a safer parser.

    df[column] = df[column].apply(_to_array)
    return df


def build_sparse_matrix_from_vector_column(df: pd.DataFrame, column: str) -> csr_matrix:
    """
    Build a sparse feature matrix from a vector-valued column.

    This helper reproduces the pattern used throughout the notebooks:

    ``X_sparse = vstack([csr_matrix(row) for row in df[column]])``

    where each element of ``df[column]`` is assumed to be a one-
    dimensional array-like object (typically a NumPy array) of fixed
    length.

    Parameters
    ----------
    df:
        Input DataFrame containing the vector-valued column.
    column:
        Name of the column holding per-row vectors.

    Returns
    -------
    scipy.sparse.csr_matrix
        A sparse matrix whose rows correspond to DataFrame rows and
        whose columns correspond to vector dimensions.
    """

    rows = [csr_matrix(row) for row in df[column]]
    return vstack(rows)


def attach_chiefcomplaint_embeddings(
    combined_df: pd.DataFrame,
    encoded_chiefcomplaints: Sequence[np.ndarray],
    column: str = "chiefcomplaint_vector",
) -> pd.DataFrame:
    """
    Attach pre-computed chief-complaint embeddings to a DataFrame.

    This mirrors the pattern in the notebooks:

    - assign the list of vectors to ``combined_df[column]``
    - convert each element to a NumPy array

    Parameters
    ----------
    combined_df:
        Feature table to enrich.
    encoded_chiefcomplaints:
        Sequence of per-row vectors, already computed elsewhere.
    column:
        Column name to store the vectors under.

    Returns
    -------
    pd.DataFrame
        The same DataFrame instance, updated in-place.
    """

    combined_df[column] = list(encoded_chiefcomplaints)
    combined_df[column] = combined_df[column].apply(lambda x: np.array(x))
    return combined_df


def reduce_vector_column_average_pool(
    df: pd.DataFrame,
    column: str,
    out_prefix: str,
    n_slices: int = 3,
) -> pd.DataFrame:
    """
    Reduce a high-dimensional vector column via simple average pooling.

    This helper reproduces the average-pooling logic used in the
    notebooks for `text_vector` and `chiefcomplaint_vector`:

    - stack all vectors vertically using ``np.vstack``
    - split the feature dimension into ``n_slices`` contiguous segments
    - compute the mean of each segment to obtain scalar features

    The resulting scalar features are written to new columns named
    ``{out_prefix}1``, ``{out_prefix}2``, ...

    Parameters
    ----------
    df:
        Input DataFrame.
    column:
        Name of the vector-valued column.
    out_prefix:
        Prefix for the new scalar feature columns.
    n_slices:
        Number of segments to form along the feature dimension.

    Returns
    -------
    pd.DataFrame
        The same DataFrame instance, updated with new columns.
    """

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
    """
    Reduce a high-dimensional vector column using PCA.

    This helper captures the PCA-based dimensionality reduction used in
    the notebooks, where stacked vectors are fed into a PCA instance and
    the resulting components are written back as new columns.

    Parameters
    ----------
    df:
        Input DataFrame.
    column:
        Name of the vector-valued column.
    n_components:
        Number of principal components to keep.
    out_prefix:
        Prefix for the new component columns. If ``n_components == 1``,
        a single column named ``out_prefix`` is used.

    Returns
    -------
    pd.DataFrame
        The same DataFrame instance, updated with PCA component columns.
    """

    # Local import to avoid making sklearn a hard dependency for users
    # who do not need PCA-based dimensionality reduction.
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


