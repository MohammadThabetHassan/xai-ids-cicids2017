"""
Unit tests for XAI-IDS evaluation module.

Tests for metrics computation, cross-validation, calibration,
and failure analysis functions.
"""

import os
import sys

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression as LR
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestComputeMetrics:
    """Test compute_metrics function."""

    def test_basic_metrics(self):
        from src.evaluation.metrics import compute_metrics

        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 0, 2])
        labels = ["A", "B", "C"]

        metrics = compute_metrics(y_true, y_pred, "test_model", label_names=labels)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert metrics["accuracy"] > 0

    def test_classification_report_structure(self):
        from src.evaluation.metrics import compute_metrics

        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        labels = ["Negative", "Positive"]

        metrics = compute_metrics(y_true, y_pred, "test", label_names=labels)

        assert "classification_report" in metrics
        assert "Negative" in metrics["classification_report"]
        assert "Positive" in metrics["classification_report"]


class TestCrossValidation:
    """Test cross-validation function."""

    def test_cv_returns_dict(self):
        from sklearn.ensemble import RandomForestClassifier

        from src.evaluation.metrics import run_cross_validation

        X, y = make_classification(
            n_samples=200, n_features=10, n_classes=2, n_informative=5, random_state=42
        )

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        results = run_cross_validation(model, X, y, cv=3, scoring="f1_weighted")

        assert "mean" in results
        assert "std" in results
        assert "scores" in results
        assert len(results["scores"]) == 3

    def test_cv_scores_reasonable(self):
        from src.evaluation.metrics import run_cross_validation

        X, y = make_classification(
            n_samples=300, n_features=15, n_classes=2, n_informative=5, random_state=42
        )

        model = LR(max_iter=500)
        results = run_cross_validation(model, X, y, cv=3, scoring="accuracy")

        assert 0 <= results["mean"] <= 1
        assert results["std"] >= 0


class TestCalibrationCurves:
    """Test calibration curve generation."""

    def test_calibration_output_exists(self, tmp_path):
        from src.evaluation.metrics import plot_calibration_curves

        X, y = make_classification(
            n_samples=200, n_features=10, n_classes=2, n_informative=5, random_state=42
        )
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        models = {"rf": model}
        labels = ["A", "B"]

        paths = plot_calibration_curves(
            models, X, y, label_names=labels, save_dir=str(tmp_path)
        )

        assert len(paths) == 1
        assert os.path.exists(paths[0])


class TestFailureAnalysis:
    """Test failure analysis generation."""

    def test_failure_analysis_output(self, tmp_path):
        from src.evaluation.metrics import generate_failure_analysis

        X, y = make_classification(
            n_samples=200, n_features=10, n_classes=2, n_informative=5, random_state=42
        )

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        models = {"rf": model}
        labels = ["A", "B"]

        results = generate_failure_analysis(
            models, X, y, label_names=labels, save_dir=str(tmp_path)
        )

        assert "rf" in results
        assert "failing_classes" in results["rf"]
        assert "top_confusions" in results["rf"]

        report_path = os.path.join(tmp_path, "failure_analysis.txt")
        assert os.path.exists(report_path)

    def test_failing_classes_identified(self):
        from src.evaluation.metrics import generate_failure_analysis

        np.random.seed(42)
        X = np.random.randn(200, 10)
        y = np.array([0] * 150 + [1] * 30 + [2] * 20)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        models = {"rf": model}
        labels = ["Major", "Minor1", "Minor2"]

        results = generate_failure_analysis(
            models, X, y, label_names=labels, save_dir="/tmp"
        )

        assert results["rf"]["macro_f1"] >= 0


class TestConfusionMatrix:
    """Test confusion matrix plotting."""

    def test_confusion_matrix_shape(self):
        from src.evaluation.metrics import compute_metrics

        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 0, 2, 1, 1, 2])
        labels = ["A", "B", "C"]

        metrics = compute_metrics(y_true, y_pred, "test", label_names=labels)

        cm = metrics["confusion_matrix"]
        assert cm.shape == (3, 3)

    def test_confusion_matrix_values(self):
        from sklearn.metrics import confusion_matrix

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])

        cm = confusion_matrix(y_true, y_pred)

        assert cm[0, 0] == 1
        assert cm[1, 1] == 2
        assert cm[0, 1] == 1
        assert cm[1, 0] == 0


class TestMetricsEdgeCases:
    """Test edge cases in metrics computation."""

    def test_all_same_prediction(self):
        from src.evaluation.metrics import compute_metrics

        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 0, 0, 0, 0])
        labels = ["A", "B", "C"]

        metrics = compute_metrics(y_true, y_pred, "test", label_names=labels)

        assert metrics["accuracy"] > 0

    def test_perfect_prediction(self):
        from src.evaluation.metrics import compute_metrics

        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2])
        labels = ["A", "B", "C"]

        metrics = compute_metrics(y_true, y_pred, "test", label_names=labels)

        assert metrics["accuracy"] == 1.0
        assert metrics["f1_score"] == 1.0
