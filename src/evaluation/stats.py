"""
Statistical significance testing for model comparison.

Provides McNemar's test, paired t-test, and cross-validation with
statistical comparison utilities for evaluating whether performance
differences between models are statistically significant.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from src.utils.logger import get_logger

logger = get_logger("xai_ids.evaluation.stats")


def mcnemar_test(
    y_true: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    correction: bool = True,
) -> Tuple[float, float]:
    """Perform McNemar's test to compare two classifiers.

    McNemar's test evaluates whether the disagreement between two
    classifiers is statistically significant. It uses the 2x2
    contingency table of cases where model A is right/wrong and
    model B is right/wrong.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels, shape (n_samples,).
    preds_a : np.ndarray
        Predictions from model A, shape (n_samples,).
    preds_b : np.ndarray
        Predictions from model B, shape (n_samples,).
    correction : bool
        Whether to apply Edwards' continuity correction (default: True).

    Returns
    -------
    Tuple[float, float]
        Chi-squared statistic and p-value.

    Notes
    -----
    H0: Both models have the same error rate.
    A small p-value (< 0.05) rejects H0, indicating one model is
    significantly better than the other.
    """
    y_true = np.asarray(y_true)
    preds_a = np.asarray(preds_a)
    preds_b = np.asarray(preds_b)

    a_correct = (preds_a == y_true)
    b_correct = (preds_b == y_true)

    # Contingency table cells
    n01 = float(np.sum(~a_correct & b_correct))   # a wrong, b right
    n10 = float(np.sum(a_correct & ~b_correct))   # a right, b wrong

    if n01 + n10 == 0:
        logger.warning("Both models agree on all samples; McNemar's test undefined")
        return 0.0, 1.0

    if correction:
        # Edwards' continuity correction
        chi2 = (abs(n01 - n10) - 1.0) ** 2 / (n01 + n10)
    else:
        chi2 = (n01 - n10) ** 2 / (n01 + n10)

    p_value = 1.0 - stats.chi2.cdf(chi2, df=1)

    logger.info(
        f"McNemar's test: chi2={chi2:.4f}, p={p_value:.6f} "
        f"(n01={n01:.0f}, n10={n10:.0f})"
    )

    return chi2, p_value


def paired_ttest(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> Tuple[float, float]:
    """Perform a paired t-test on per-sample scores from two models.

    Parameters
    ----------
    scores_a : np.ndarray
        Per-sample scores (e.g., 0/1 correctness, or per-fold F1)
        from model A, shape (n,).
    scores_b : np.ndarray
        Per-sample scores from model B, shape (n,).

    Returns
    -------
    Tuple[float, float]
        t-statistic and p-value (two-sided).

    Notes
    -----
    H0: The mean difference between scores is zero.
    A small p-value (< 0.05) indicates a significant difference.
    """
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)

    if len(scores_a) != len(scores_b):
        raise ValueError("scores_a and scores_b must have the same length")

    if len(scores_a) < 2:
        raise ValueError("Need at least 2 samples for paired t-test")

    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)

    # Handle NaN from zero-variance differences (identical scores)
    if np.isnan(p_value):
        p_value = 1.0
    if np.isnan(t_stat):
        t_stat = 0.0

    logger.info(f"Paired t-test: t={t_stat:.4f}, p={p_value:.6f}")

    return t_stat, p_value


def compare_models_statistically(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    model_names: Optional[List[str]] = None,
    alpha: float = 0.05,
) -> Dict:
    """Run McNemar's test and paired t-test for all model pairs.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels.
    predictions : Dict[str, np.ndarray]
        Mapping from model name to prediction array.
    model_names : list of str, optional
        Subset of models to compare. Defaults to all.
    alpha : float
        Significance level (default: 0.05).

    Returns
    -------
    Dict
        Nested dict: {model_a: {model_b: {mcnemar_p, ttest_p,
        mcnemar_sig, ttest_sig, acc_a, acc_b}}}
    """
    if model_names is None:
        model_names = list(predictions.keys())

    results = {}

    for i, name_a in enumerate(model_names):
        results[name_a] = {}
        acc_a = np.mean(predictions[name_a] == y_true)

        for name_b in model_names[i + 1:]:
            acc_b = np.mean(predictions[name_b] == y_true)

            # McNemar's test
            _, mcnemar_p = mcnemar_test(y_true, predictions[name_a], predictions[name_b])

            # Paired t-test on per-sample correctness (0 or 1)
            correct_a = (predictions[name_a] == y_true).astype(float)
            correct_b = (predictions[name_b] == y_true).astype(float)
            _, ttest_p = paired_ttest(correct_a, correct_b)

            results[name_a][name_b] = {
                "acc_a": round(float(acc_a), 6),
                "acc_b": round(float(acc_b), 6),
                "mcnemar_p": round(float(mcnemar_p), 6),
                "ttest_p": round(float(ttest_p), 6),
                "mcnemar_sig": bool(mcnemar_p < alpha),
                "ttest_sig": bool(ttest_p < alpha),
            }

    return results


def cross_validate_with_stats(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    random_state: int = 42,
) -> Dict:
    """Run stratified k-fold CV and collect per-fold metrics.

    Parameters
    ----------
    model : sklearn-compatible estimator
        Model with fit() and predict() methods.
    X : np.ndarray
        Features, shape (n_samples, n_features).
    y : np.ndarray
        Labels, shape (n_samples,).
    cv : int
        Number of folds (default: 5).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    Dict
        Contains mean, std, and per-fold arrays for accuracy,
        precision, recall, and f1 (all weighted).
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    fold_metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "predictions": [],
        "y_true": [],
    }

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        fold_metrics["accuracy"].append(accuracy_score(y_test, y_pred))
        fold_metrics["precision"].append(
            precision_score(y_test, y_pred, average="weighted", zero_division=0)
        )
        fold_metrics["recall"].append(
            recall_score(y_test, y_pred, average="weighted", zero_division=0)
        )
        fold_metrics["f1"].append(
            f1_score(y_test, y_pred, average="weighted", zero_division=0)
        )
        fold_metrics["predictions"].append(y_pred)
        fold_metrics["y_true"].append(y_test)

        logger.info(
            f"  Fold {fold_idx + 1}/{cv}: "
            f"acc={fold_metrics['accuracy'][-1]:.4f}, "
            f"f1={fold_metrics['f1'][-1]:.4f}"
        )

    # Convert to numpy arrays
    for key in ["accuracy", "precision", "recall", "f1"]:
        fold_metrics[key] = np.array(fold_metrics[key])

    fold_metrics["mean_accuracy"] = float(np.mean(fold_metrics["accuracy"]))
    fold_metrics["std_accuracy"] = float(np.std(fold_metrics["accuracy"]))
    fold_metrics["mean_f1"] = float(np.mean(fold_metrics["f1"]))
    fold_metrics["std_f1"] = float(np.std(fold_metrics["f1"]))

    return fold_metrics


def compare_models_cv(
    models: Dict[str, object],
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    alpha: float = 0.05,
) -> Dict:
    """Run CV on all models and perform pairwise statistical tests on F1 scores.

    Parameters
    ----------
    models : Dict[str, object]
        Mapping from model name to untrained model instance.
    X : np.ndarray
        Features.
    y : np.ndarray
        Labels.
    cv : int
        Number of CV folds.
    alpha : float
        Significance level.

    Returns
    -------
    Dict
        Per-model CV results + pairwise statistical comparisons.
    """
    cv_results = {}
    model_names = list(models.keys())

    for name, model in models.items():
        logger.info(f"Cross-validating {name}...")
        # Use a fresh copy if possible
        from sklearn.base import clone
        try:
            model_copy = clone(model)
        except Exception:
            model_copy = model
        cv_results[name] = cross_validate_with_stats(model_copy, X, y, cv=cv)

    # Pairwise comparisons on per-fold F1
    comparisons = {}
    for i, name_a in enumerate(model_names):
        for name_b in model_names[i + 1:]:
            f1_a = cv_results[name_a]["f1"]
            f1_b = cv_results[name_b]["f1"]
            _, ttest_p = paired_ttest(f1_a, f1_b)
            comparisons[f"{name_a} vs {name_b}"] = {
                "mean_f1_a": round(float(np.mean(f1_a)), 6),
                "mean_f1_b": round(float(np.mean(f1_b)), 6),
                "ttest_p": round(float(ttest_p), 6),
                "significant": bool(ttest_p < alpha),
            }

    return {
        "cv_results": cv_results,
        "pairwise_comparisons": comparisons,
    }
