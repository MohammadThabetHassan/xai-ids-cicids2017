"""Tests for adversarial robustness evaluation."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.adversarial import (
    compute_xcs_on_adversarial,
    evaluate_adversarial_robustness,
)


class TestAdversarialRobustness:
    """Test adversarial robustness evaluation."""

    def test_returns_dict_on_missing_art(self):
        """If ART is not installed, should return error dict gracefully."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 10).astype(np.float32)
        y = rng.choice([0, 1], size=100)

        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)

        result = evaluate_adversarial_robustness(model, X, y)
        assert isinstance(result, dict)
        assert "epsilons" in result or "error" in result

    def test_compute_xcs_on_adversarial(self):
        """XCS comparison between clean and adversarial samples."""
        rng = np.random.RandomState(42)
        X = rng.randn(200, 10).astype(np.float32)
        y = rng.choice([0, 1], size=200)

        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        X_adv = X + rng.randn(*X.shape).astype(np.float32) * 0.1

        result = compute_xcs_on_adversarial(model, X, X_adv, y, n_samples=20)
        assert "mean_xcs_clean" in result
        assert "mean_xcs_adversarial" in result
        assert "xcs_drop" in result
        assert 0 <= result["mean_xcs_clean"] <= 1
        assert 0 <= result["mean_xcs_adversarial"] <= 1
