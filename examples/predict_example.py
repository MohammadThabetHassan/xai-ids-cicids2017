"""
Example: Load a trained model and make a prediction.

This script demonstrates how to load a saved model from the
Kaggle notebook or pipeline and make predictions on new data.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_kaggle_model(dataset: str = "CICIDS2017", model_name: str = "xgboost") -> dict:
    """
    Load a trained model from the Kaggle notebook outputs.

    Parameters
    ----------
    dataset : str
        Dataset name: CICIDS2017, UNSWNB15, or CICIDS2018.
    model_name : str
        Model name: xgboost, random_forest, or lightgbm.

    Returns
    -------
    dict
        Dictionary with model, scaler, label_encoder, and feature_names.
    """
    models_dir = PROJECT_ROOT / "models"

    model_map = {
        "xgboost": f"xgboost_{dataset}.joblib",
        "random_forest": f"rf_{dataset}.joblib",
        "lightgbm": f"lightgbm_{dataset}.joblib",
    }

    filename = model_map.get(model_name)
    if not filename:
        raise ValueError(f"Unknown model: {model_name}")

    model_path = models_dir / filename
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    artifacts = joblib.load(model_path)

    result = {"model": artifacts["model"]}

    # Load associated artifacts
    scaler_path = models_dir / f"scaler_{dataset}.joblib"
    if scaler_path.exists():
        result["scaler"] = joblib.load(scaler_path)

    le_path = models_dir / f"label_encoder_{dataset}.joblib"
    if le_path.exists():
        result["label_encoder"] = joblib.load(le_path)

    return result


def predict(artifacts: dict, features: np.ndarray) -> dict:
    """
    Make a prediction using loaded model artifacts.

    Parameters
    ----------
    artifacts : dict
        Dictionary with model, scaler, and label_encoder.
    features : np.ndarray
        Input features (1D or 2D array).

    Returns
    -------
    dict
        Prediction with class name, confidence, and probabilities.
    """
    model = artifacts["model"]
    scaler = artifacts.get("scaler")
    label_encoder = artifacts.get("label_encoder")

    if features.ndim == 1:
        features = features.reshape(1, -1)

    if scaler is not None:
        features = scaler.transform(features)

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    result = {
        "prediction": prediction,
        "confidence": float(probabilities[prediction]),
        "probabilities": {
            int(i): float(p) for i, p in enumerate(probabilities)
        },
    }

    if label_encoder is not None:
        result["predicted_class"] = str(label_encoder.inverse_transform([prediction])[0])
        result["probabilities"] = {
            str(label_encoder.inverse_transform([i])[0]): float(p)
            for i, p in enumerate(probabilities)
        }

    return result


if __name__ == "__main__":
    print("XAI-IDS Model Prediction Example")
    print("=" * 40)

    try:
        # Load model
        artifacts = load_kaggle_model(dataset="CICIDS2017", model_name="xgboost")
        print(f"Loaded XGBoost model for CICIDS2017")

        # Create dummy input (replace with real features)
        n_features = 20  # CICIDS2017 uses 20 selected features
        dummy_features = np.zeros(n_features)

        # Make prediction
        result = predict(artifacts, dummy_features)
        print(f"\nPrediction: {result.get('predicted_class', result['prediction'])}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"\nTop 3 probabilities:")
        sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
        for cls, prob in sorted_probs[:3]:
            print(f"  {cls}: {prob:.4f}")

    except FileNotFoundError as e:
        print(f"\nModel files not found. Run the Kaggle notebook first.")
        print(f"Error: {e}")
    except Exception as e:
        print(f"\nError: {e}")
