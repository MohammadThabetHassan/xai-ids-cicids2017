"""Tests for temporal drift detection."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.drift import detect_feature_drift, simulate_temporal_drift


class TestFeatureDrift:
    """Test feature drift detection."""

    def test_no_drift_identical_distributions(self):
        """Identical distributions should show minimal drift."""
        rng = np.random.RandomState(42)
        X_ref = rng.randn(500, 5)
        X_cur = rng.randn(500, 5)

        result = detect_feature_drift(X_ref, X_cur, alpha=0.05)
        assert "features" in result
        assert "n_drifted" in result
        assert "drift_rate" in result
        assert 0 <= result["drift_rate"] <= 1.0
        assert len(result["features"]) == 5

    def test_drift_shifted_distribution(self):
        """Shifted distribution should detect drift."""
        rng = np.random.RandomState(42)
        X_ref = rng.randn(500, 5)
        X_cur = rng.randn(500, 5) + 3.0

        result = detect_feature_drift(X_ref, X_cur, alpha=0.05)
        assert result["n_drifted"] > 0

    def test_feature_names_in_output(self):
        """Feature names should appear in output."""
        rng = np.random.RandomState(42)
        X_ref = rng.randn(100, 3)
        X_cur = rng.randn(100, 3)
        names = ["feat_a", "feat_b", "feat_c"]

        result = detect_feature_drift(X_ref, X_cur, feature_names=names)
        output_names = [f["name"] for f in result["features"]]
        assert output_names == names


class TestTemporalDrift:
    """Test temporal drift simulation."""

    def test_returns_valid_structure(self):
        """Should return dict with expected keys."""
        from sklearn.ensemble import RandomForestClassifier

        rng = np.random.RandomState(42)
        X = rng.randn(500, 5)
        y = rng.choice([0, 1], size=500)

        model = RandomForestClassifier(n_estimators=5, random_state=42)
        result = simulate_temporal_drift(model, X, y, n_splits=5)

        assert "splits" in result
        assert "train_accuracies" in result
        assert "test_accuracies" in result
        assert "test_f1s" in result
        assert "feature_drift_per_split" in result
        assert len(result["splits"]) > 0

    def test_accuracy_values_valid(self):
        """All accuracy values should be in [0, 1]."""
        from sklearn.ensemble import RandomForestClassifier

        rng = np.random.RandomState(42)
        X = rng.randn(500, 5)
        y = rng.choice([0, 1], size=500)

        model = RandomForestClassifier(n_estimators=5, random_state=42)
        result = simulate_temporal_drift(model, X, y, n_splits=5)

        for acc in result["train_accuracies"]:
            assert 0 <= acc <= 1
        for acc in result["test_accuracies"]:
            assert 0 <= acc <= 1
