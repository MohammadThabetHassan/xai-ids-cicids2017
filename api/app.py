"""
FastAPI inference endpoint for XAI-IDS.

Provides REST API for real-time intrusion detection predictions
with SHAP explanations.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger

logger = get_logger("xai_ids.api")

app = FastAPI(
    title="XAI-IDS API",
    description="Explainable AI Intrusion Detection System API",
    version="1.0.0",
)

MODELS_DIR = Path("outputs/models")

models = {}
scaler = None
label_encoder = None
feature_names = None


class PredictionInput(BaseModel):
    features: List[float] = Field(
        ..., description="Network flow features (78 features)"
    )

    class Config:
        json_schema_extra = {"example": {"features": [0.0] * 78}}


class PredictionOutput(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict


class ExplanationOutput(BaseModel):
    prediction: str
    confidence: float
    top_features: List[dict]


def load_models():
    """Load trained models and preprocessors."""
    global models, scaler, label_encoder, feature_names

    model_files = {
        "random_forest": MODELS_DIR / "random_forest.pkl",
        "xgboost": MODELS_DIR / "xgboost.pkl",
        "logistic_regression": MODELS_DIR / "logistic_regression.pkl",
    }

    for name, path in model_files.items():
        if path.exists():
            models[name] = joblib.load(path)
            logger.info(f"Loaded {name}")

    scaler_path = MODELS_DIR / "scaler.pkl"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        logger.info("Loaded scaler")

    le_path = MODELS_DIR / "label_encoder.pkl"
    if le_path.exists():
        label_encoder = joblib.load(le_path)
        logger.info("Loaded label encoder")


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    load_models()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "XAI-IDS API",
        "version": "1.0.0",
        "models_loaded": list(models.keys()),
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": len(models) > 0,
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
        Prediction with confidence and probabilities.
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

        model = models.get("random_forest") or models.get("xgboost")
        if not model:
            raise HTTPException(status_code=503, detail="No prediction model loaded")

        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        predicted_class = label_encoder.inverse_transform([prediction])[0]
        confidence = float(probabilities[prediction])

        prob_dict = {
            label_encoder.inverse_transform([i])[0]: float(p)
            for i, p in enumerate(probabilities)
        }

        return PredictionOutput(
            prediction=predicted_class,
            confidence=confidence,
            probabilities=prob_dict,
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
        Prediction with feature importance explanation.
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
            {"feature": f"feature_{i}", "importance": float(imp)}
            for i, imp in enumerate(importances)
        ]
        feature_importance.sort(key=lambda x: x["importance"], reverse=True)
        top_features = feature_importance[:10]

        return ExplanationOutput(
            prediction=predicted_class,
            confidence=confidence,
            top_features=top_features,
        )

    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/classes")
async def get_classes():
    """Get list of available classes."""
    if label_encoder is None:
        raise HTTPException(status_code=503, detail="Label encoder not loaded")

    return {"classes": list(label_encoder.classes_)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
