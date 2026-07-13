"""Reusable clustering helpers for customer segmentation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .preprocessing import CLUSTERING_FEATURES


def _validate_scaled_matrix(X_scaled: np.ndarray) -> None:
    """Validate that the input is a 2D NumPy array."""
    if not isinstance(X_scaled, np.ndarray):
        raise TypeError("X_scaled must be a NumPy array.")

    if X_scaled.ndim != 2:
        raise ValueError("X_scaled must be a 2D array.")


def train_kmeans(X_scaled: np.ndarray, n_clusters: int) -> tuple[KMeans, np.ndarray]:
    """Train a K-Means model and return the fitted model and cluster labels.

    Parameters
    ----------
    X_scaled:
        Scaled feature matrix used for clustering.
    n_clusters:
        Number of clusters to fit.

    Returns
    -------
    tuple[sklearn.cluster.KMeans, numpy.ndarray]
        The fitted KMeans model and the predicted cluster labels.
    """
    _validate_scaled_matrix(X_scaled)

    if n_clusters < 1:
        raise ValueError("n_clusters must be at least 1.")

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = model.fit_predict(X_scaled)
    return model, cluster_labels


def get_cluster_centers(model: KMeans, scaler: StandardScaler) -> pd.DataFrame:
    """Convert cluster centers back to the original feature scale.

    Parameters
    ----------
    model:
        A fitted KMeans model.
    scaler:
        The fitted scaler used to transform the clustering features.

    Returns
    -------
    pandas.DataFrame
        Cluster centers in the original feature scale.
    """
    centers = scaler.inverse_transform(model.cluster_centers_)
    return pd.DataFrame(centers, columns=list(CLUSTERING_FEATURES))


__all__ = ["get_cluster_centers", "train_kmeans"]