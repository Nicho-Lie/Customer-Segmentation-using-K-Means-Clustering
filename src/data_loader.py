"""Utilities for loading raw customer datasets."""

from pathlib import Path

import pandas as pd


def load_customer_data(file_name: str = "customers.csv") -> pd.DataFrame:
    """Load a raw customer dataset from the project's data/raw directory.

    Parameters
    ----------
    file_name:
        The CSV file name stored in the raw data directory.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the loaded customer data.

    Raises
    ------
    FileNotFoundError
        If the requested file does not exist in the raw data directory.
    """
    project_root = Path(__file__).resolve().parents[1]
    data_directory = project_root / "data" / "raw"
    file_path = data_directory / file_name

    if not file_path.exists():
        raise FileNotFoundError(f"Customer data file not found: {file_path}")

    if not file_path.is_file():
        raise FileNotFoundError(f"Expected a file but found a directory: {file_path}")

    return pd.read_csv(file_path)

def load_processed_data(
    file_name: str = "customer_features.csv"
) -> pd.DataFrame:
    """Load processed customer features."""

    project_root = Path(__file__).resolve().parents[1]
    data_directory = project_root / "data" / "processed"
    file_path = data_directory / file_name

    if not file_path.exists():
        raise FileNotFoundError(
            f"Processed data file not found: {file_path}"
        )

    return pd.read_csv(file_path)
