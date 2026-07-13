"""Reusable preprocessing utilities for customer segmentation."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


CLUSTERING_FEATURES: tuple[str, str] = (
    "Annual Income (k$)",
    "Spending Score (1-100)",
)


def select_clustering_features(
    df: pd.DataFrame,
    features: Sequence[str] = CLUSTERING_FEATURES,
) -> pd.DataFrame:
    """Return a copy of the columns used for customer clustering.

    Parameters
    ----------
    df:
        Input DataFrame containing the raw or partially processed customer data.
    features:
        Column names to retain for clustering.

    Returns
    -------
    pandas.DataFrame
        A copy of the selected feature columns.

    Raises
    ------
    KeyError
        If any required clustering feature is missing from the input DataFrame.
    """
    missing_columns = [column for column in features if column not in df.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise KeyError(f"Missing required clustering feature(s): {missing_text}")

    return df.loc[:, list(features)].copy()


def impute_missing_values(X: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values using the median strategy.

    Median imputation matches the preprocessing approach used in the notebook and
    provides a robust default for numeric features when outliers may be present.

    Parameters
    ----------
    X:
        Selected clustering features.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with missing values filled.
    """
    imputer = SimpleImputer(strategy="median")
    imputed_array = imputer.fit_transform(X)
    return pd.DataFrame(imputed_array, columns=X.columns, index=X.index)


def scale_features(X: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Scale features with StandardScaler.

    Parameters
    ----------
    X:
        Imputed clustering features.

    Returns
    -------
    numpy.ndarray
        Scaled feature matrix suitable for K-Means clustering.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def preprocess_customer_data(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, StandardScaler]:
    """Prepare customer data for K-Means clustering.

    The preprocessing pipeline performs three steps:
    1. Select the final clustering features.
    2. Impute missing values using the median.
    3. Scale the features using StandardScaler.

    Parameters
    ----------
    df:
        Raw customer DataFrame.

    Returns
    -------
    tuple[pandas.DataFrame, numpy.ndarray]
        X:
            Selected feature DataFrame after imputation and before scaling.
        X_scaled:
            Scaled NumPy array ready for K-Means clustering.

    Notes
    -----
    This function assumes the input DataFrame already contains the feature names
    used during exploratory analysis:
    "Annual Income (k$)" and "Spending Score (1-100)".
    """
    X = select_clustering_features(df)
    X = impute_missing_values(X)

    X_scaled, scaler = scale_features(X)

    return X, X_scaled, scaler


__all__ = [
    "CLUSTERING_FEATURES",
    "impute_missing_values",
    "preprocess_customer_data",
    "scale_features",
    "select_clustering_features",
]