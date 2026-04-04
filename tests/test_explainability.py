"""Tests for SHAP and LIME explainability."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSHAP:
    """Test SHAP explanation computation."""

    def test_shap_import(self):
        shap = pytest.importorskip("shap")
        assert shap is not None

    def test_shap_on_synthetic_data(self):
        shap = pytest.importorskip("shap")
        from sklearn.ensemble import RandomForestClassifier

        rng = np.random.RandomState(42)
        X = rng.randn(100, 10)
        y = rng.choice([0, 1], size=100)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X[:5])

        assert sv is not None
        if isinstance(sv, list):
            assert len(sv) == 2  # binary classes
            assert sv[0].shape[1] == 10
        else:
            assert sv.shape[1] == 10


class TestLIME:
    """Test LIME explanation computation."""

    def test_lime_import(self):
        lime = pytest.importorskip("lime")
        assert lime is not None

    def test_lime_on_synthetic_data(self):
        pytest.importorskip("lime")
        import lime.lime_tabular
        from sklearn.ensemble import RandomForestClassifier

        rng = np.random.RandomState(42)
        X = rng.randn(100, 10)
        y = rng.choice([0, 1], size=100)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        feature_names = [f"feat_{i}" for i in range(10)]
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X, feature_names=feature_names, mode="classification"
        )

        exp = explainer.explain_instance(
            X[0], model.predict_proba, num_features=5, num_samples=200
        )

        assert exp is not None
        as_map = exp.as_map()
        assert len(as_map) > 0


class TestCounterfactual:
    """Test counterfactual explanation generation."""

    def test_counterfactual_import(self):
        from src.explainability.counterfactual import generate_counterfactuals
        assert generate_counterfactuals is not None

    def test_counterfactual_fallback(self):
        from sklearn.ensemble import RandomForestClassifier

        from src.explainability.counterfactual import generate_counterfactuals

        rng = np.random.RandomState(42)
        X = rng.randn(100, 10)
        y = rng.choice([0, 1], size=100)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        feature_names = [f"feat_{i}" for i in range(10)]
        query = rng.randn(1, 10)

        result = generate_counterfactuals(
            model, query, feature_names, n_counterfactuals=2
        )

        assert "original_prediction" in result
        assert "counterfactuals" in result
        assert len(result["counterfactuals"]) == 2
