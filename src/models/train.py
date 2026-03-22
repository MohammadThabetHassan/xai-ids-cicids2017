"""
Model training module for XAI-IDS.

Trains Logistic Regression, Random Forest, and XGBoost classifiers
on the CIC-IDS-2017 dataset with proper logging and artifact saving.
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
            "use_label_encoder": False,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
        },
    },
}


def train_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    save_dir: str = "outputs/models",
    custom_params: Optional[Dict] = None,
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

    model = config["class"](**params)

    start_time = time.time()

    if model_name == "xgboost" and X_val is not None:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
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
    model_path = os.path.join(save_dir, f"{model_name}.pkl")
    joblib.dump(model, model_path)
    logger.info(f"  Saved model to {model_path}")

    return model


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    save_dir: str = "outputs/models",
) -> Dict[str, Any]:
    """
    Train all configured models.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    X_val : np.ndarray, optional
        Validation features.
    y_val : np.ndarray, optional
        Validation labels.
    save_dir : str
        Directory to save models.

    Returns
    -------
    Dict[str, Any]
        Dictionary mapping model names to trained model instances.
    """
    logger.info("=" * 60)
    logger.info("MODEL TRAINING")
    logger.info("=" * 60)

    models = {}

    for model_name in MODEL_CONFIGS:
        try:
            model = train_model(model_name, X_train, y_train, X_val, y_val, save_dir)
            models[model_name] = model
        except Exception as e:
            logger.error(f"Failed to train {model_name}: {e}")

    logger.info(f"Successfully trained {len(models)}/{len(MODEL_CONFIGS)} models")
    return models
