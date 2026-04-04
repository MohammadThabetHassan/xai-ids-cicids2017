"""Regression tests for model performance thresholds.

Ensures that model performance does not silently regress below
acceptable thresholds.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Performance thresholds from config
THRESHOLDS = {
    "min_accuracy_synthetic": 0.55,
    "min_macro_f1_synthetic": 0.30,
    "min_weighted_f1_synthetic": 0.55,
}


class TestPerformanceRegression:
    """Test that model performance stays above minimum thresholds."""

    def test_synthetic_pipeline_accuracy(self):
        """Accuracy on synthetic data should exceed threshold."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score

        from src.data.generate_sample import generate_sample_dataset
        from src.data.preprocessing import clean_data, encode_labels, split_data, scale_features

        # Generate small synthetic dataset
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = generate_sample_dataset(n_samples=2000, output_dir=tmpdir)

            import pandas as pd
            df = pd.read_csv(csv_path)
            df = clean_data(df)
            label_col = "Label"
            df, le, _ = encode_labels(df, label_col=label_col, save_path=None)

            X_train, X_val, X_test, y_train, y_val, y_test = split_data(
                df, label_col=label_col
            )

            feature_names = list(X_train.columns)
            X_train_s, X_val_s, X_test_s, scaler = scale_features(
                X_train, X_val, X_test, save_path=None
            )

            model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)

            acc = accuracy_score(y_test, y_pred)
            assert acc >= THRESHOLDS["min_accuracy_synthetic"], (
                f"Accuracy {acc:.4f} below threshold {THRESHOLDS['min_accuracy_synthetic']}"
            )

    def test_model_predict_proba_sums_to_one(self):
        """Model probabilities should sum to 1."""
        from sklearn.ensemble import RandomForestClassifier

        rng = np.random.RandomState(42)
        X = rng.randn(200, 10)
        y = rng.choice([0, 1, 2], size=200)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        proba = model.predict_proba(X[:10])
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
