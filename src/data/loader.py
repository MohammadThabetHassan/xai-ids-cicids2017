"""
Data loading utilities for CIC-IDS-2017 CSV files.

Supports memory-efficient chunked loading, merging multiple CSV files,
and handling common data quality issues.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("xai_ids.loader")


def find_csv_files(data_dir: str = "data/raw") -> List[str]:
    """
    Find all CSV files in the specified directory.

    Parameters
    ----------
    data_dir : str
        Directory to search for CSV files.

    Returns
    -------
    List[str]
        Sorted list of CSV file paths.
    """
    csv_files = sorted(
        [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.lower().endswith(".csv")
        ]
    )
    logger.info(f"Found {len(csv_files)} CSV files in {data_dir}")
    for f in csv_files:
        logger.info(f"  - {os.path.basename(f)}")
    return csv_files


def load_single_csv(
    filepath: str,
    chunk_size: Optional[int] = None,
    nrows: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """
    Load a single CSV file with error handling.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    chunk_size : int, optional
        If specified, read in chunks and concatenate.
    nrows : int, optional
        Maximum number of rows to read.

    Returns
    -------
    Optional[pd.DataFrame]
        Loaded DataFrame, or None if loading failed.
    """
    filename = os.path.basename(filepath)
    logger.info(f"Loading {filename}...")

    try:
        if chunk_size:
            chunks = []
            rows_read = 0
            for chunk in pd.read_csv(
                filepath,
                low_memory=False,
                encoding="utf-8",
                on_bad_lines="skip",
                chunksize=chunk_size,
            ):
                chunks.append(chunk)
                rows_read += len(chunk)
                if nrows and rows_read >= nrows:
                    break
            if not chunks:
                logger.warning(f"No data loaded from {filename}")
                return None
            df = pd.concat(chunks, ignore_index=True)
            if nrows:
                df = df.head(nrows)
        else:
            df = pd.read_csv(
                filepath,
                low_memory=False,
                encoding="utf-8",
                on_bad_lines="skip",
                nrows=nrows,
            )

        logger.info(f"Loaded {filename}: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    except Exception as e:
        logger.error(f"Failed to load {filename}: {e}")
        return None


def load_dataset(
    data_dir: str = "data/raw",
    chunk_size: Optional[int] = 50000,
    nrows_per_file: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load and merge all CSV files from the dataset directory.

    Parameters
    ----------
    data_dir : str
        Directory containing raw CSV files.
    chunk_size : int, optional
        Chunk size for memory-efficient reading.
    nrows_per_file : int, optional
        Maximum rows per file (useful for testing).

    Returns
    -------
    pd.DataFrame
        Merged dataset.

    Raises
    ------
    FileNotFoundError
        If no CSV files are found.
    ValueError
        If no data could be loaded.
    """
    csv_files = find_csv_files(data_dir)

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    dataframes = []
    skipped = []

    for filepath in csv_files:
        df = load_single_csv(filepath, chunk_size=chunk_size, nrows=nrows_per_file)
        if df is not None:
            dataframes.append(df)
        else:
            skipped.append(filepath)

    if not dataframes:
        raise ValueError("No data could be loaded from any CSV file")

    if skipped:
        logger.warning(f"Skipped {len(skipped)} files: {skipped}")

    merged = pd.concat(dataframes, ignore_index=True)
    logger.info(f"Merged dataset: {merged.shape[0]} rows, {merged.shape[1]} columns")

    return merged


def load_processed_dataset(
    filepath: str = "data/processed/cleaned_dataset.csv",
) -> pd.DataFrame:
    """
    Load the preprocessed dataset.

    Parameters
    ----------
    filepath : str
        Path to the processed CSV.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Processed dataset not found: {filepath}")

    df = pd.read_csv(filepath, low_memory=False)
    logger.info(f"Loaded processed dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df
