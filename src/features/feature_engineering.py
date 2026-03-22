"""
Feature engineering utilities for CIC-IDS-2017.

Provides feature selection, extraction, and analysis functions
for network intrusion detection.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from src.utils.logger import get_logger

logger = get_logger("xai_ids.features")


def remove_constant_features(
    df: pd.DataFrame, label_col: str = "Label"
) -> pd.DataFrame:
    """
    Remove features with zero or near-zero variance.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    label_col : str
        Label column to exclude from removal.

    Returns
    -------
    pd.DataFrame
        Dataset with constant features removed.
    """
    feature_cols = [c for c in df.columns if c != label_col]
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns

    variances = df[numeric_cols].var()
    constant_cols = variances[variances < 1e-10].index.tolist()

    if constant_cols:
        logger.info(f"Removing {len(constant_cols)} constant features: {constant_cols}")
        df = df.drop(columns=constant_cols)
    else:
        logger.info("No constant features found")

    return df


def remove_highly_correlated(
    df: pd.DataFrame, threshold: float = 0.95, label_col: str = "Label"
) -> pd.DataFrame:
    """
    Remove one of each pair of highly correlated features.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    threshold : float
        Correlation threshold above which to drop one feature.
    label_col : str
        Label column to exclude.

    Returns
    -------
    pd.DataFrame
        Dataset with correlated features removed.
    """
    feature_cols = [c for c in df.columns if c != label_col]
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns

    corr_matrix = df[numeric_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    if to_drop:
        logger.info(
            f"Removing {len(to_drop)} highly correlated features "
            f"(threshold={threshold}): {to_drop[:10]}..."
        )
        df = df.drop(columns=to_drop)
    else:
        logger.info("No highly correlated feature pairs found")

    return df


def compute_feature_importance_mi(
    X: np.ndarray, y: np.ndarray, feature_names: List[str], n_top: int = 20
) -> List[Tuple[str, float]]:
    """
    Compute mutual information-based feature importance.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Labels.
    feature_names : List[str]
        Feature names.
    n_top : int
        Number of top features to return.

    Returns
    -------
    List[Tuple[str, float]]
        Top features with their MI scores.
    """
    logger.info("Computing mutual information feature importance...")

    mi_scores = mutual_info_classif(X, y, random_state=42)
    feature_importance = sorted(
        zip(feature_names, mi_scores), key=lambda x: x[1], reverse=True
    )

    logger.info(f"Top {n_top} features by mutual information:")
    for name, score in feature_importance[:n_top]:
        logger.info(f"  {name}: {score:.4f}")

    return feature_importance[:n_top]
