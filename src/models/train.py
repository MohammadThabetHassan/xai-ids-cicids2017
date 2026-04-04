"""
Model training module for XAI-IDS.

Trains Logistic Regression, Random Forest, and XGBoost classifiers
on the CIC-IDS-2017 dataset with proper logging and artifact saving.
Supports SMOTE oversampling for imbalanced datasets.
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

from src.utils.logger import get_logger

logger = get_logger("xai_ids.models")

# Model configurations
MODEL_CONFIGS = {
    "logistic_regression": {
        "class": LogisticRegression,
        "params": {
            "max_iter": 1000,
            "random_state": 42,
            "solver": "lbfgs",
            "n_jobs": -1,
            "C": 1.0,
            "class_weight": "balanced",
        },
    },
    "random_forest": {
        "class": RandomForestClassifier,
        "params": {
            "n_estimators": 100,
            "max_depth": 20,
            "random_state": 42,
            "n_jobs": -1,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
        },
    },
    "xgboost": {
        "class": XGBClassifier,
        "params": {
            "n_estimators": 100,
            "max_depth": 8,
            "learning_rate": 0.1,
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
        },
    },
}


def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    min_samples_per_class: int = 200,
    random_state: int = 42,
) -> tuple:
    """Apply SMOTE oversampling for minority classes.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    min_samples_per_class : int
        Minimum number of samples per class after oversampling.
    random_state : int
        Random seed.

    Returns
    -------
    tuple
        (X_resampled, y_resampled)
    """
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        logger.warning("imbalanced-learn not installed; skipping SMOTE")
        return X_train, y_train

    class_counts = np.bincount(y_train.astype(int))
    n_classes = len(class_counts)

    # Determine sampling strategy: minority classes get min_samples_per_class
    sampling_strategy = {}
    for cls in range(n_classes):
        if class_counts[cls] < min_samples_per_class:
            sampling_strategy[cls] = min_samples_per_class

    if not sampling_strategy:
        logger.info("All classes have sufficient samples; skipping SMOTE")
        return X_train, y_train

    logger.info(
        f"Applying SMOTE: {len(sampling_strategy)} classes below "
        f"{min_samples_per_class} samples will be oversampled"
    )

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        k_neighbors=min(5, min(class_counts) - 1) if min(class_counts) > 1 else 1,
    )
    X_res, y_res = smote.fit_resample(X_train, y_train)

    logger.info(
        f"SMOTE: {len(X_train)} -> {len(X_res)} samples "
        f"({len(X_res) - len(X_train)} synthetic samples added)"
    )

    return X_res, y_res


def train_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    save_dir: str = "outputs/models",
    custom_params: Optional[Dict] = None,
    use_balanced_weights: bool = True,
    use_smote: bool = False,
    smote_min_samples: int = 200,
) -> Any:
    """
    Train a single model.

    Parameters
    ----------
    model_name : str
        Name of the model (key in MODEL_CONFIGS).
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    X_val : np.ndarray, optional
        Validation features.
    y_val : np.ndarray, optional
        Validation labels.
    save_dir : str
        Directory to save trained model.
    custom_params : Dict, optional
        Override default parameters.
    use_balanced_weights : bool
        Whether to use balanced class weights (default: True).
    use_smote : bool
        Whether to apply SMOTE oversampling (default: False).
    smote_min_samples : int
        Minimum samples per class for SMOTE (default: 200).

    Returns
    -------
    Any
        Trained model instance.
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}"
        )

    config = MODEL_CONFIGS[model_name]
    params = {**config["params"]}
    if custom_params:
        params.update(custom_params)

    logger.info(f"Training {model_name}...")
    logger.info(f"  Parameters: {params}")
    logger.info(f"  Training samples: {X_train.shape[0]}")

    # Apply SMOTE if requested
    if use_smote:
        X_train, y_train = apply_smote(X_train, y_train, min_samples_per_class=smote_min_samples)
        logger.info(f"  After SMOTE: {X_train.shape[0]} samples")

    model = config["class"](**params)

    start_time = time.time()

    sample_weight = None
    if use_balanced_weights and model_name == "xgboost":
        classes = np.unique(y_train)
        class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
        sample_weight = np.array(
            [class_weights[np.where(classes == y)[0][0]] for y in y_train]
        )
        logger.info(f"  Using balanced sample weights for XGBoost")

    if model_name == "xgboost" and X_val is not None:
        if sample_weight is not None:
            model.fit(
                X_train,
                y_train,
                sample_weight=sample_weight,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
    elif sample_weight is not None:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)

    train_time = time.time() - start_time
    logger.info(f"  Training time: {train_time:.2f} seconds")

    # Training accuracy
    train_acc = model.score(X_train, y_train)
    logger.info(f"  Training accuracy: {train_acc:.4f}")

    if X_val is not None and y_val is not None:
        val_acc = model.score(X_val, y_val)
        logger.info(f"  Validation accuracy: {val_acc:.4f}")

    # Save model
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(save_dir, f"{model_name}.joblib")
    joblib.dump(model, model_path)
    logger.info(f"  Saved model to {model_path}")

    return model
