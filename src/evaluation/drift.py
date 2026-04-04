"""
Temporal drift detection for XAI-IDS models.

Simulates temporal drift by splitting data into time-based chunks
and measuring how model performance degrades across temporal splits.
Uses Kolmogorov-Smirnov tests for feature distribution drift.
"""

from typing import Dict, List, Optional

import numpy as np

from src.utils.logger import get_logger

logger = get_logger("xai_ids.evaluation.drift")


def detect_feature_drift(
    X_reference: np.ndarray,
    X_current: np.ndarray,
    feature_names: Optional[List[str]] = None,
    alpha: float = 0.05,
) -> Dict:
    """Detect feature distribution drift using KS-test.

    Parameters
    ----------
    X_reference : np.ndarray
        Reference (training) features, shape (n_ref, n_features).
    X_current : np.ndarray
        Current (test) features, shape (n_cur, n_features).
    feature_names : list of str, optional
        Names of features.
    alpha : float
        Significance level for KS-test.

    Returns
    -------
    Dict
        Per-feature KS statistic, p-value, and drift flag.
    """
    from scipy import stats

    n_features = X_reference.shape[1]
    names = feature_names or [f"feature_{i}" for i in range(n_features)]

    results = {
        "features": [],
        "n_drifted": 0,
        "drift_rate": 0.0,
    }

    for i in range(n_features):
        ks_stat, p_value = stats.ks_2samp(X_reference[:, i], X_current[:, i])
        drifted = bool(p_value < alpha)

        results["features"].append({
            "name": names[i],
            "ks_statistic": round(float(ks_stat), 6),
            "p_value": round(float(p_value), 6),
            "drifted": drifted,
            "ref_mean": round(float(np.mean(X_reference[:, i])), 6),
            "cur_mean": round(float(np.mean(X_current[:, i])), 6),
            "ref_std": round(float(np.std(X_reference[:, i])), 6),
            "cur_std": round(float(np.std(X_current[:, i])), 6),
        })

        if drifted:
            results["n_drifted"] += 1

    results["drift_rate"] = results["n_drifted"] / max(n_features, 1)

    logger.info(
        f"Feature drift: {results['n_drifted']}/{n_features} features drifted "
        f"({results['drift_rate']:.1%})"
    )

    return results


def simulate_temporal_drift(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict:
    """Simulate temporal drift by sequential train/test splits.

    Trains on earlier chunks and tests on later chunks to measure
    how performance degrades as the 'time gap' increases.

    Parameters
    ----------
    model : sklearn-compatible classifier
    X : np.ndarray
        Features, shape (n_samples, n_features).
    y : np.ndarray
        Labels, shape (n_samples,).
    n_splits : int
        Number of temporal splits.
    random_state : int
        Random seed.

    Returns
    -------
    Dict
        Per-split accuracy, f1, and feature drift metrics.
    """
    from sklearn.metrics import accuracy_score, f1_score

    rng = np.random.RandomState(random_state)
    n_samples = len(X)
    chunk_size = n_samples // n_splits

    # Shuffle deterministically
    indices = rng.permutation(n_samples)
    X = X[indices]
    y = y[indices]

    results = {
        "splits": [],
        "train_accuracies": [],
        "test_accuracies": [],
        "test_f1s": [],
        "feature_drift_per_split": [],
    }

    for split_idx in range(n_splits - 1):
        # Train on all chunks up to and including current
        train_end = (split_idx + 1) * chunk_size
        X_train = X[:train_end]
        y_train = y[:train_end]

        # Test on next chunk
        test_start = (split_idx + 1) * chunk_size
        test_end = min((split_idx + 2) * chunk_size, n_samples)
        X_test = X[test_start:test_end]
        y_test = y[test_start:test_end]

        if len(X_test) == 0:
            break

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # Feature drift between train and test
        drift = detect_feature_drift(X_train, X_test)

        results["splits"].append({
            "train_size": len(X_train),
            "test_size": len(X_test),
            "train_accuracy": round(float(train_acc), 6),
            "test_accuracy": round(float(test_acc), 6),
            "test_f1": round(float(test_f1), 6),
            "feature_drift_rate": round(float(drift["drift_rate"]), 6),
        })
        results["train_accuracies"].append(float(train_acc))
        results["test_accuracies"].append(float(test_acc))
        results["test_f1s"].append(float(test_f1))
        results["feature_drift_per_split"].append(drift)

        logger.info(
            f"Split {split_idx + 1}: train_acc={train_acc:.4f}, "
            f"test_acc={test_acc:.4f}, f1={test_f1:.4f}, "
            f"drift_rate={drift['drift_rate']:.1%}"
        )

    return results


def plot_temporal_drift(
    drift_results: Dict,
    save_path: str = "plots/temporal_drift_CICIDS2017.png",
):
    """Plot temporal drift: accuracy and feature drift over splits.

    Parameters
    ----------
    drift_results : Dict
        Output from simulate_temporal_drift().
    save_path : str
        Path to save the plot.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    splits = drift_results["splits"]
    if not splits:
        logger.warning("No splits to plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    x = list(range(1, len(splits) + 1))

    # Accuracy over time
    ax1 = axes[0]
    train_accs = [s["train_accuracy"] for s in splits]
    test_accs = [s["test_accuracy"] for s in splits]
    ax1.plot(x, train_accs, "o-", label="Train Accuracy", color="#4CAF50", linewidth=2)
    ax1.plot(x, test_accs, "s-", label="Test Accuracy", color="#2196F3", linewidth=2)
    ax1.set_xlabel("Temporal Split (increasing time gap)")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Model Performance Over Time")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # F1 over time
    ax2 = axes[1]
    f1s = [s["test_f1"] for s in splits]
    ax2.plot(x, f1s, "D-", color="#FF9800", linewidth=2, markersize=8)
    ax2.set_xlabel("Temporal Split")
    ax2.set_ylabel("Weighted F1")
    ax2.set_title("F1 Score Degradation Over Time")
    ax2.grid(True, alpha=0.3)

    # Feature drift rate
    ax3 = axes[2]
    drift_rates = [s["feature_drift_rate"] for s in splits]
    colors = ["#4CAF50" if d < 0.2 else "#FF9800" if d < 0.5 else "#F44336" for d in drift_rates]
    ax3.bar(x, drift_rates, color=colors, edgecolor="black", alpha=0.8)
    ax3.set_xlabel("Temporal Split")
    ax3.set_ylabel("Feature Drift Rate")
    ax3.set_title("Feature Distribution Drift (KS-test)")
    ax3.set_ylim(0, 1.0)
    ax3.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved temporal drift plot to {save_path}")
