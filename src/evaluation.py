"""Reusable clustering evaluation utilities for customer segmentation."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def _prepare_k_values(k_range: Iterable[int], minimum_k: int) -> list[int]:
    """Normalize and validate the candidate cluster values."""
    k_values = [int(k) for k in k_range]
    if not k_values:
        raise ValueError("k_range must contain at least one cluster value.")

    invalid_values = [k for k in k_values if k < minimum_k]
    if invalid_values:
        raise ValueError(
            f"Cluster values must be >= {minimum_k}. Invalid values: {invalid_values}"
        )

    return k_values


def _validate_input_matrix(X_scaled: np.ndarray) -> None:
    """Ensure the input is a 2D numeric matrix suitable for K-Means."""
    if not isinstance(X_scaled, np.ndarray):
        raise TypeError("X_scaled must be a NumPy array.")

    if X_scaled.ndim != 2:
        raise ValueError("X_scaled must be a 2D array.")


def evaluate_inertia(X_scaled: np.ndarray, k_range: Iterable[int]) -> pd.DataFrame:
    """Evaluate K-Means inertia across a range of cluster counts.

    Parameters
    ----------
    X_scaled:
        Scaled feature matrix used for clustering.
    k_range:
        Iterable of cluster counts to evaluate.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with columns ``K`` and ``inertia``.
    """
    _validate_input_matrix(X_scaled)
    k_values = _prepare_k_values(k_range, minimum_k=1)

    results: list[dict[str, float]] = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X_scaled)
        results.append({"K": k, "inertia": float(model.inertia_)})

    return pd.DataFrame(results)


def evaluate_silhouette(X_scaled: np.ndarray, k_range: Iterable[int]) -> pd.DataFrame:
    """Evaluate silhouette scores across a range of cluster counts.

    Parameters
    ----------
    X_scaled:
        Scaled feature matrix used for clustering.
    k_range:
        Iterable of cluster counts to evaluate. Values must be greater than or
        equal to 2 because silhouette score is undefined for a single cluster.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with columns ``K`` and ``silhouette_score``.
    """
    _validate_input_matrix(X_scaled)
    k_values = _prepare_k_values(k_range, minimum_k=2)

    results: list[dict[str, float]] = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        results.append({"K": k, "silhouette_score": float(score)})

    return pd.DataFrame(results)


__all__ = ["evaluate_inertia", "evaluate_silhouette"]