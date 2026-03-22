"""
Data preprocessing pipeline for CIC-IDS-2017.

Handles cleaning, encoding, scaling, and train/test splitting
with deterministic and reproducible operations.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils.logger import get_logger

logger = get_logger("xai_ids.preprocessing")

# Label column name in CIC-IDS-2017
LABEL_COLUMN = " Label"
LABEL_COLUMN_ALT = "Label"

RANDOM_STATE = 42


def identify_label_column(df: pd.DataFrame) -> str:
    """
    Identify the label column in the dataset.

    The CIC-IDS-2017 dataset has inconsistent label column naming
    (sometimes with leading space).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.

    Returns
    -------
    str
        Name of the label column.

    Raises
    ------
    KeyError
        If no label column is found.
    """
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    if "Label" in df.columns:
        return "Label"

    raise KeyError(f"Label column not found. Available columns: {list(df.columns)}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by handling missing values, infinities, and duplicates.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset.
    """
    initial_rows = len(df)
    logger.info(f"Starting data cleaning: {initial_rows} rows")

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Replace infinities with NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    inf_count = initial_rows - len(df.dropna(subset=numeric_cols))
    logger.info(f"Replaced infinite values -> {inf_count} rows affected")

    # Drop rows with NaN values
    df = df.dropna()
    after_nan = len(df)
    logger.info(f"Dropped NaN rows: {initial_rows - after_nan} removed")

    # Drop duplicates
    df = df.drop_duplicates()
    after_dedup = len(df)
    logger.info(f"Dropped duplicates: {after_nan - after_dedup} removed")

    # Reset index
    df = df.reset_index(drop=True)

    logger.info(
        f"Cleaning complete: {initial_rows} -> {len(df)} rows "
        f"({initial_rows - len(df)} removed, "
        f"{(initial_rows - len(df)) / max(initial_rows, 1) * 100:.1f}%)"
    )

    return df


def encode_labels(
    df: pd.DataFrame,
    label_col: str = "Label",
    save_path: Optional[str] = "outputs/models/label_encoder.pkl",
) -> Tuple[pd.DataFrame, LabelEncoder, Dict[str, int]]:
    """
    Encode string labels to integers.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with string labels.
    label_col : str
        Name of the label column.
    save_path : str, optional
        Path to save the fitted LabelEncoder.

    Returns
    -------
    Tuple[pd.DataFrame, LabelEncoder, Dict[str, int]]
        DataFrame with encoded labels, the fitted encoder, and the label mapping.
    """
    logger.info("Encoding labels...")

    le = LabelEncoder()
    df[label_col] = le.fit_transform(df[label_col].astype(str))

    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    logger.info(f"Label mapping ({len(label_mapping)} classes):")
    for name, code in sorted(label_mapping.items(), key=lambda x: x[1]):
        count = (df[label_col] == code).sum()
        logger.info(f"  {code}: {name} ({count} samples)")

    if save_path:
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        joblib.dump(le, save_path)
        logger.info(f"Saved label encoder to {save_path}")

    return df, le, label_mapping


def scale_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    save_path: Optional[str] = "outputs/models/scaler.pkl",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Standardize features using StandardScaler (fit on train only).

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    X_val : pd.DataFrame
        Validation features.
    X_test : pd.DataFrame
        Test features.
    save_path : str, optional
        Path to save the fitted scaler.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]
        Scaled arrays and the fitted scaler.
    """
    logger.info("Scaling features with StandardScaler...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    logger.info(
        f"Scaling complete. "
        f"Train mean: {X_train_scaled.mean():.4f}, "
        f"Train std: {X_train_scaled.std():.4f}"
    )

    if save_path:
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, save_path)
        logger.info(f"Saved scaler to {save_path}")

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def split_data(
    df: pd.DataFrame,
    label_col: str = "Label",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split the dataset into train, validation, and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.
    label_col : str
        Name of the label column.
    test_size : float
        Fraction for test set.
    val_size : float
        Fraction for validation set (from remaining after test split).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    Tuple
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    logger.info(
        f"Splitting data: test={test_size}, val={val_size}, seed={random_state}"
    )

    X = df.drop(columns=[label_col])
    y = df[label_col]

    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Second split: separate validation from training
    val_fraction = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_fraction,
        random_state=random_state,
        stratify=y_temp,
    )

    logger.info(f"Train: {X_train.shape[0]} samples")
    logger.info(f"Val:   {X_val.shape[0]} samples")
    logger.info(f"Test:  {X_test.shape[0]} samples")

    return X_train, X_val, X_test, y_train, y_val, y_test


def run_preprocessing(
    data_dir: str = "data/raw",
    output_dir: str = "data/processed",
    models_dir: str = "outputs/models",
    nrows_per_file: Optional[int] = None,
) -> Dict:
    """
    Execute the full preprocessing pipeline.

    Parameters
    ----------
    data_dir : str
        Directory containing raw CSV files.
    output_dir : str
        Directory for processed data.
    models_dir : str
        Directory for saving preprocessing artifacts.
    nrows_per_file : int, optional
        Limit rows per file (for testing).

    Returns
    -------
    Dict
        Dictionary containing split data, encoder, scaler, and metadata.
    """
    from src.data.loader import load_dataset

    logger.info("=" * 60)
    logger.info("PREPROCESSING PIPELINE")
    logger.info("=" * 60)

    # Load data
    df = load_dataset(data_dir, nrows_per_file=nrows_per_file)

    # Clean
    df = clean_data(df)

    # Identify label column
    label_col = identify_label_column(df)

    # Encode labels
    df, le, label_mapping = encode_labels(
        df,
        label_col=label_col,
        save_path=os.path.join(models_dir, "label_encoder.pkl"),
    )

    # Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, label_col=label_col)

    # Get feature names before scaling
    feature_names = list(X_train.columns)

    # Scale
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X_train,
        X_val,
        X_test,
        save_path=os.path.join(models_dir, "scaler.pkl"),
    )

    # Save processed data info
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    result = {
        "X_train": X_train_scaled,
        "X_val": X_val_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train.values,
        "y_val": y_val.values,
        "y_test": y_test.values,
        "feature_names": feature_names,
        "label_encoder": le,
        "label_mapping": label_mapping,
        "scaler": scaler,
    }

    logger.info("Preprocessing pipeline complete")
    return result
