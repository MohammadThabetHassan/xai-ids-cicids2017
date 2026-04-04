"""
Unit tests for loading saved models and making predictions.

Tests verify that trained model artifacts can be loaded and
produce valid predictions.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestModelLoading:
    """Test loading saved model artifacts."""

    def test_kaggle_models_exist(self):
        """Verify that Kaggle model files exist in models/ directory."""
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        if not os.path.exists(models_dir):
            pytest.skip("models/ directory not found")

        model_files = os.listdir(models_dir)
        assert len(model_files) > 0, "No model files found in models/"

        # Check for expected model patterns
        has_xgb = any("xgb_" in f for f in model_files)
        has_rf = any("rf_" in f or "random_forest" in f for f in model_files)
        has_lgbm = any("lightgbm" in f or "lgb_" in f for f in model_files)

        assert has_xgb or has_rf or has_lgbm, "No trained models found in models/"

    def test_load_kaggle_model_artifacts(self):
        """Test loading a Kaggle model artifact and verifying structure."""
        import joblib

        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        if not os.path.exists(models_dir):
            pytest.skip("models/ directory not found")

        # Find any xgb model file
        model_files = [f for f in os.listdir(models_dir) if f.startswith("xgb_") and f.endswith(".joblib")]
        if not model_files:
            pytest.skip("No xgb model files found")

        model_path = os.path.join(models_dir, model_files[0])
        model = joblib.load(model_path)

        assert hasattr(model, "predict"), "Loaded object has no predict method"
        assert hasattr(model, "predict_proba"), "Loaded object has no predict_proba method"

    def test_load_scaler(self):
        """Test loading a saved scaler."""
        import joblib

        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        if not os.path.exists(models_dir):
            pytest.skip("models/ directory not found")

        scaler_files = [f for f in os.listdir(models_dir) if "scaler" in f and f.endswith(".joblib")]
        if not scaler_files:
            pytest.skip("No scaler files found")

        scaler_path = os.path.join(models_dir, scaler_files[0])
        scaler = joblib.load(scaler_path)

        assert hasattr(scaler, "transform"), "Scaler has no transform method"
        assert hasattr(scaler, "n_features_in_"), "Scaler has no n_features_in_ attribute"


class TestPrediction:
    """Test making predictions with loaded models."""

    def test_predict_with_loaded_model(self):
        """Test loading a model and making a prediction."""
        import joblib

        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        if not os.path.exists(models_dir):
            pytest.skip("models/ directory not found")

        model_files = [f for f in os.listdir(models_dir) if f.startswith("xgb_") and f.endswith(".joblib")]
        if not model_files:
            pytest.skip("No xgb model files found")

        model_path = os.path.join(models_dir, model_files[0])
        model = joblib.load(model_path)

        # Create dummy input matching expected feature count
        n_features = model.n_features_in_ if hasattr(model, "n_features_in_") else 20
        dummy_input = np.zeros((1, n_features))

        # Make prediction
        prediction = model.predict(dummy_input)
        probabilities = model.predict_proba(dummy_input)

        assert prediction is not None
        assert len(prediction) == 1
        assert probabilities.shape == (1, model.n_classes_)
        assert np.isclose(probabilities.sum(), 1.0, atol=1e-6)

    def test_predict_with_scaler(self):
        """Test prediction pipeline with scaler + model."""
        import joblib

        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        if not os.path.exists(models_dir):
            pytest.skip("models/ directory not found")

        # Find matching scaler and model for same dataset
        datasets = ["CICIDS2017", "UNSWNB15", "CICIDS2018"]
        for dataset in datasets:
            scaler_files = [f for f in os.listdir(models_dir) if f"scaler_{dataset}" in f]
            model_files = [f for f in os.listdir(models_dir) if f"xgb_{dataset}" in f]

            if scaler_files and model_files:
                scaler = joblib.load(os.path.join(models_dir, scaler_files[0]))
                model = joblib.load(os.path.join(models_dir, model_files[0]))

                n_features = scaler.n_features_in_
                dummy_input = np.zeros((1, n_features))
                scaled = scaler.transform(dummy_input)
                prediction = model.predict(scaled)

                assert prediction is not None
                assert len(prediction) == 1
                break
        else:
            pytest.skip("No matching scaler+model pair found for any dataset")
