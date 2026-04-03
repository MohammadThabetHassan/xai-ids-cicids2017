"""
FastAPI inference endpoint for XAI-IDS.

Provides REST API for real-time intrusion detection predictions
with SHAP explanations.
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger

logger = get_logger("xai_ids.api")

# Support both pipeline outputs/models and Kaggle models/ directory
MODELS_DIR = Path("models")
PIPELINE_MODELS_DIR = Path("outputs/models")

models = {}
scaler = None
label_encoder = None
feature_names = None


class PredictionInput(BaseModel):
    features: List[float] = Field(
        ...,
        description="Network flow features (78 for CIC-IDS-2017, 20 for Kaggle models)",
        min_length=1,
        max_length=100,
    )

    class Config:
        json_schema_extra = {
            "example": {"features": [0.0] * 78}
        }


class PredictionOutput(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict
    xcs_score: Optional[float] = None


class ExplanationOutput(BaseModel):
    prediction: str
    confidence: float
    top_features: List[dict]
    xcs_score: Optional[float] = None


def load_models():
    """Load trained models and preprocessors from available directories."""
    global models, scaler, label_encoder, feature_names

    # Try Kaggle models/ directory first, then pipeline outputs/models/
    model_dirs = [MODELS_DIR, PIPELINE_MODELS_DIR]

    for model_dir in model_dirs:
        if not model_dir.exists():
            continue

        # Load models (try various naming conventions)
        model_patterns = {
            "xgboost": ["xgboost.pkl", "xgboost_CICIDS2017.joblib", "xgb_CICIDS2017.joblib"],
            "random_forest": ["random_forest.pkl", "rf_CICIDS2017.joblib", "rf_UNSWNB15.joblib", "rf_CICIDS2018.joblib"],
            "lightgbm": ["lightgbm_CICIDS2017.joblib", "lgb_CICIDS2017.joblib", "lightgbm_UNSWNB15.joblib", "lightgbm_CICIDS2018.joblib"],
            "logistic_regression": ["logistic_regression.pkl"],
        }

        for name, patterns in model_patterns.items():
            for pattern in patterns:
                path = model_dir / pattern
                if path.exists() and name not in models:
                    try:
                        models[name] = joblib.load(path)
                        logger.info(f"Loaded {name} from {path}")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load {path}: {e}")

        # Load scaler
        for scaler_name in ["scaler.pkl", "scaler_CICIDS2017.joblib"]:
            scaler_path = model_dir / scaler_name
            if scaler_path.exists() and scaler is None:
                try:
                    scaler = joblib.load(scaler_path)
                    logger.info(f"Loaded scaler from {scaler_path}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load scaler {scaler_path}: {e}")

        # Load label encoder
        for le_name in ["label_encoder.pkl", "label_encoder_CICIDS2017.joblib"]:
            le_path = model_dir / le_name
            if le_path.exists() and label_encoder is None:
                try:
                    label_encoder = joblib.load(le_path)
                    logger.info(f"Loaded label encoder from {le_path}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load label encoder {le_path}: {e}")

    if models:
        logger.info(f"Loaded {len(models)} models: {list(models.keys())}")
    else:
        logger.warning("No models found in models/ or outputs/models/")

    if scaler is not None:
        logger.info("Scaler loaded")
    if label_encoder is not None:
        logger.info(f"Label encoder loaded with {len(label_encoder.classes_)} classes")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup and cleanup on shutdown."""
    load_models()
    yield
    # Cleanup on shutdown
    models.clear()


app = FastAPI(
    title="XAI-IDS API",
    description="Explainable AI Intrusion Detection System API",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "XAI-IDS API",
        "version": "2.0.0",
        "models_loaded": list(models.keys()),
        "scaler_loaded": scaler is not None,
        "label_encoder_loaded": label_encoder is not None,
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": len(models) > 0,
        "models": list(models.keys()),
        "scaler_loaded": scaler is not None,
        "encoder_loaded": label_encoder is not None,
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """
    Make a prediction for a single network flow.

    Parameters
    ----------
    input_data : PredictionInput
        Network flow features.

    Returns
    -------
    PredictionOutput
        Prediction with confidence, probabilities, and XCS score.
    """
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded")

    if scaler is None:
        raise HTTPException(status_code=503, detail="Scaler not loaded")

    if label_encoder is None:
        raise HTTPException(status_code=503, detail="Label encoder not loaded")

    try:
        features = np.array(input_data.features).reshape(1, -1)
        features_scaled = scaler.transform(features)

        model = models.get("random_forest") or models.get("xgboost") or models.get("lightgbm")
        if not model:
            raise HTTPException(status_code=503, detail="No prediction model loaded")

        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        predicted_class = label_encoder.inverse_transform([prediction])[0]
        confidence = float(probabilities[prediction])

        prob_dict = {
            str(label_encoder.inverse_transform([i])[0]): float(p)
            for i, p in enumerate(probabilities)
        }

        # Compute simple XCS score (confidence-only if SHAP/LIME not available)
        xcs_score = 0.4 * confidence  # Minimum XCS from confidence component

        return PredictionOutput(
            prediction=predicted_class,
            confidence=confidence,
            probabilities=prob_dict,
            xcs_score=round(xcs_score, 4),
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input: expected {scaler.n_features_in_} features, got {len(input_data.features)}"
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain", response_model=ExplanationOutput)
async def explain(input_data: PredictionInput):
    """
    Make a prediction with feature importance explanation.

    Parameters
    ----------
    input_data : PredictionInput
        Network flow features.

    Returns
    -------
    ExplanationOutput
        Prediction with feature importance explanation and XCS score.
    """
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded")

    if scaler is None:
        raise HTTPException(status_code=503, detail="Scaler not loaded")

    if label_encoder is None:
        raise HTTPException(status_code=503, detail="Label encoder not loaded")

    try:
        features = np.array(input_data.features).reshape(1, -1)
        features_scaled = scaler.transform(features)

        model = models.get("random_forest")
        if not model:
            model = models.get("xgboost")
        if not model:
            model = models.get("lightgbm")
        if not model:
            raise HTTPException(
                status_code=503, detail="No model loaded for explanation"
            )

        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        predicted_class = label_encoder.inverse_transform([prediction])[0]
        confidence = float(probabilities[prediction])

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            raise HTTPException(
                status_code=500, detail="Model lacks feature importances"
            )

        feature_importance = [
            {"feature": f"feature_{i}", "importance": round(float(imp), 6)}
            for i, imp in enumerate(importances)
        ]
        feature_importance.sort(key=lambda x: x["importance"], reverse=True)
        top_features = feature_importance[:10]

        # Compute simple XCS score
        xcs_score = 0.4 * confidence

        return ExplanationOutput(
            prediction=predicted_class,
            confidence=confidence,
            top_features=top_features,
            xcs_score=round(xcs_score, 4),
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input: expected {scaler.n_features_in_} features, got {len(input_data.features)}"
        )
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/classes")
async def get_classes():
    """Get list of available classes."""
    if label_encoder is None:
        raise HTTPException(status_code=503, detail="Label encoder not loaded")

    return {"classes": [str(c) for c in label_encoder.classes_]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
