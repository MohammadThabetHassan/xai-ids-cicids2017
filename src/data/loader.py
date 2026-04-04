"""
Data loading utilities for CIC-IDS-2017 CSV files.

Supports memory-efficient chunked loading, merging multiple CSV files,
and handling common data quality issues.
"""

import os
from typing import List, Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("xai_ids.loader")


def find_csv_files(data_dir: str = "data/raw") -> List[str]:
    """
    Find all CSV files in the specified directory.

    Prefers CIC-IDS-2017-V2.csv if present (real data),
    excludes sample_cicids2017.csv (synthetic) unless it's the only option.

    Parameters
    ----------
    data_dir : str
        Directory to search for CSV files.

    Returns
    -------
    List[str]
        Sorted list of CSV file paths (V2 preferred).
    """
    all_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith(".csv")
    ]

    # Prefer V2 file (real data), exclude synthetic sample unless only option
    v2_files = [f for f in all_files if "CIC-IDS-2017-V2" in f]
    sample_files = [f for f in all_files if "sample_cicids" in f.lower()]
    other_files = [f for f in all_files if f not in v2_files and f not in sample_files]

    # Use V2 if available, otherwise other real files, then sample as fallback
    if v2_files:
        csv_files = sorted(v2_files)
        logger.info(f"Using real dataset: {os.path.basename(csv_files[0])}")
    elif other_files:
        csv_files = sorted(other_files)
        logger.info(f"Found {len(csv_files)} CSV files in {data_dir}")
    elif all_files:
        csv_files = sorted(all_files)
        logger.warning("No V2 or real data files found, using all available")
    else:
        csv_files = []

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
    random_sample: bool = False,
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
    random_sample : bool
        If True, sample randomly from large files instead of taking first N rows.

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
        if random_sample and nrows_per_file:
            # Random sample from large file
            import numpy as np

            with open(filepath, "r") as f:
                total_rows = sum(1 for _ in f) - 1  # -1 for header
            if total_rows > nrows_per_file:
                skiprows = np.random.choice(
                    range(1, total_rows + 1), total_rows - nrows_per_file, replace=False
                )
                skiprows = sorted(skiprows)
                df = pd.read_csv(filepath, skiprows=skiprows, low_memory=False)
                logger.info(
                    f"Random sampled {os.path.basename(filepath)}: {df.shape[0]} rows"
                )
            else:
                df = load_single_csv(
                    filepath, chunk_size=chunk_size, nrows=nrows_per_file
                )
        else:
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
