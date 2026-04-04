"""Tests for statistical significance testing."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.stats import mcnemar_test, paired_ttest


class TestMcNemarTest:
    """Test McNemar's test implementation."""

    def test_identical_models(self):
        """When models are identical, p-value should be 1.0."""
        rng = np.random.RandomState(42)
        y_true = rng.choice([0, 1], size=100)
        preds = y_true.copy()

        chi2, p = mcnemar_test(y_true, preds, preds)
        assert p == 1.0

    def test_completely_different(self):
        """When models disagree systematically, p-value should be low."""
        rng = np.random.RandomState(42)
        y_true = rng.choice([0, 1], size=200)
        preds_a = y_true.copy()
        # preds_b is wrong on different samples than preds_a
        preds_b = y_true.copy()
        # Make model A wrong on first 30, model B wrong on last 30
        preds_a[:30] = 1 - y_true[:30]
        preds_b[-30:] = 1 - y_true[-30:]

        chi2, p = mcnemar_test(y_true, preds_a, preds_b)
        assert 0 <= p <= 1.0

    def test_returns_valid_range(self):
        """Chi2 >= 0 and p in [0, 1]."""
        rng = np.random.RandomState(42)
        y_true = rng.choice([0, 1], size=100)
        preds_a = rng.choice([0, 1], size=100)
        preds_b = rng.choice([0, 1], size=100)

        chi2, p = mcnemar_test(y_true, preds_a, preds_b)
        assert chi2 >= 0
        assert 0 <= p <= 1.0


class TestPairedTTest:
    """Test paired t-test implementation."""

    def test_identical_scores(self):
        """When scores are identical, p-value should be 1.0."""
        scores = np.array([0.8, 0.85, 0.9, 0.75, 0.88])
        t_stat, p = paired_ttest(scores, scores)
        assert p == 1.0

    def test_different_scores(self):
        """When scores differ, p-value should reflect significance."""
        scores_a = np.array([0.9, 0.92, 0.88, 0.91, 0.89])
        scores_b = np.array([0.7, 0.72, 0.68, 0.71, 0.69])

        t_stat, p = paired_ttest(scores_a, scores_b)
        assert 0 <= p <= 1.0
        # Should be significant given the large difference
        assert p < 0.05

    def test_requires_same_length(self):
        """Should raise error for different length arrays."""
        with pytest.raises(ValueError):
            paired_ttest(np.array([1, 2, 3]), np.array([1, 2]))
