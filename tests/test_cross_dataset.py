"""Tests for cross-dataset generalization."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.cross_dataset import evaluate_cross_dataset, map_features


class TestMapFeatures:
    """Test feature mapping between datasets."""

    def test_shared_features_mapped(self):
        """Should extract shared features by name."""
        X = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        source = ["a", "b", "c", "d"]
        target = ["b", "d", "e"]

        X_mapped, shared = map_features(X, source, target)
        assert shared == ["b", "d"]
        assert X_mapped.shape == (2, 2)
        np.testing.assert_array_equal(X_mapped[:, 0], [2, 6])
        np.testing.assert_array_equal(X_mapped[:, 1], [4, 8])

    def test_no_shared_features_fallback(self):
        """Should fallback to positional mapping when no shared features."""
        X = np.array([[1, 2, 3], [4, 5, 6]])
        source = ["a", "b", "c"]
        target = ["x", "y"]

        X_mapped, shared = map_features(X, source, target)
        assert X_mapped.shape == (2, 2)

    def test_all_shared(self):
        """When all features match, should return all."""
        X = np.array([[1, 2], [3, 4]])
        source = ["a", "b"]
        target = ["a", "b"]

        X_mapped, shared = map_features(X, source, target)
        assert shared == ["a", "b"]
        np.testing.assert_array_equal(X_mapped, X)


class TestCrossDatasetEval:
    """Test cross-dataset evaluation."""

    def test_returns_valid_structure(self):
        """Should return dict with expected keys."""
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
            source_name="src",
            target_name="tgt",
        )

        assert "source_dataset" in result
        assert "target_dataset" in result
        assert "in_distribution" in result
        assert "cross_dataset" in result
        assert "performance_drop" in result
        assert 0 <= result["in_distribution"]["accuracy"] <= 1
        assert 0 <= result["cross_dataset"]["accuracy"] <= 1

    def test_performance_drop_calculated(self):
        """Should compute accuracy and F1 drop."""
        from sklearn.ensemble import RandomForestClassifier

        rng = np.random.RandomState(42)
        X_src = rng.randn(200, 5)
        y_src = rng.choice([0, 1], size=200)
        X_tgt = rng.randn(100, 5) + 2.0
        y_tgt = rng.choice([0, 1], size=100)

        model = RandomForestClassifier(n_estimators=5, random_state=42)
        result = evaluate_cross_dataset(
            model, X_src, y_src, X_tgt, y_tgt,
            source_features=["a", "b", "c", "d", "e"],
            target_features=["a", "b", "c", "d", "e"],
        )

        drop = result["performance_drop"]
        assert "accuracy_drop" in drop
        assert "f1_drop" in drop
        assert "accuracy_drop_pct" in drop
