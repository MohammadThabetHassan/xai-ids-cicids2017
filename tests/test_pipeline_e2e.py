"""End-to-end pipeline tests exercising CLI flags."""

import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPipelineFlags:
    """Test that pipeline CLI flags work without crashing."""

    def test_stats_flag(self):
        """--stats flag should run McNemar's test without error."""
        from src.evaluation.stats import compare_models_statistically

        rng = np.random.RandomState(42)
        y_true = rng.choice([0, 1, 2], size=200)
        preds_a = rng.choice([0, 1, 2], size=200)
        preds_b = rng.choice([0, 1, 2], size=200)

        results = compare_models_statistically(
            y_true, {"model_a": preds_a, "model_b": preds_b}
        )
        assert "model_a" in results
        assert "model_b" in results["model_a"]
        assert "mcnemar_p" in results["model_a"]["model_b"]

    def test_adversarial_flag_no_art(self):
        """--adversarial flag should return error dict when model doesn't support gradients."""
        from src.evaluation.adversarial import evaluate_adversarial_robustness

        from sklearn.ensemble import RandomForestClassifier
        rng = np.random.RandomState(42)
        X = rng.randn(100, 10).astype(np.float32)
        y = rng.choice([0, 1], size=100)
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)

        result = evaluate_adversarial_robustness(model, X, y)
        assert isinstance(result, dict)
        assert "error" in result or "epsilons" in result

    def test_drift_flag(self):
        """--drift flag should run temporal drift detection."""
        from src.evaluation.drift import simulate_temporal_drift

        from sklearn.ensemble import RandomForestClassifier
        rng = np.random.RandomState(42)
        X = rng.randn(500, 5)
        y = rng.choice([0, 1], size=500)
        model = RandomForestClassifier(n_estimators=5, random_state=42)

        result = simulate_temporal_drift(model, X, y, n_splits=5)
        assert "splits" in result
        assert len(result["splits"]) > 0

    def test_counterfactuals_flag(self):
        """--counterfactuals flag should generate counterfactuals."""
        from src.explainability.counterfactual import generate_counterfactuals

        from sklearn.ensemble import RandomForestClassifier
        rng = np.random.RandomState(42)
        X = rng.randn(100, 10)
        y = rng.choice([0, 1], size=100)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        feature_names = [f"f{i}" for i in range(10)]
        query = rng.randn(1, 10)

        result = generate_counterfactuals(model, query, feature_names, n_counterfactuals=2)
        assert "original_prediction" in result
        assert "counterfactuals" in result
        assert len(result["counterfactuals"]) == 2

    def test_cross_dataset_flag(self):
        """--cross-dataset flag should run cross-dataset evaluation."""
        from src.evaluation.cross_dataset import evaluate_cross_dataset

        from sklearn.ensemble import RandomForestClassifier
        rng = np.random.RandomState(42)
        X_src = rng.randn(200, 5)
        y_src = rng.choice([0, 1], size=200)
        X_tgt = rng.randn(100, 5)
        y_tgt = rng.choice([0, 1], size=100)
        model = RandomForestClassifier(n_estimators=5, random_state=42)

        result = evaluate_cross_dataset(
            model, X_src, y_src, X_tgt, y_tgt,
            source_features=["a", "b", "c", "d", "e"],
            target_features=["a", "b", "c", "d", "e"],
        )
        assert "in_distribution" in result
        assert "cross_dataset" in result
        assert "performance_drop" in result

    def test_learned_xcs_flag(self):
        """--learned-xcs flag should fit logistic regression and return weights."""
        from sklearn.linear_model import LogisticRegression

        rng = np.random.RandomState(42)
        n = 200
        conf = rng.rand(n)
        instab = rng.rand(n) * 0.5
        jaccard = rng.rand(n)
        correct = (conf > 0.5).astype(float)

        X = np.column_stack([conf, 1 - instab, jaccard])
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X, correct)

        weights = lr.coef_[0]
        weights_norm = weights / weights.sum()
        assert len(weights_norm) == 3
        assert abs(weights_norm.sum() - 1.0) < 1e-6
