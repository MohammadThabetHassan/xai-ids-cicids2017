"""
Cross-dataset generalization experiment for XAI-IDS.

Trains models on one dataset and evaluates on another to measure
domain shift and generalization capability.
"""

from typing import Dict, List, Tuple

import numpy as np

from src.utils.logger import get_logger

logger = get_logger("xai_ids.evaluation.cross_dataset")

# Known shared feature names across datasets
SHARED_FEATURES = {
    "CICIDS2017_UNSWNB15": [
        "Dst Port", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
        "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Fwd Pkt Len Max",
        "Fwd Pkt Len Min", "Fwd Pkt Len Mean", "Fwd Pkt Len Std",
        "Bwd Pkt Len Max", "Bwd Pkt Len Min", "Bwd Pkt Len Mean",
        "Bwd Pkt Len Std", "Flow Byts/s", "Flow Pkts/s",
        "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    ],
}


def map_features(
    X: np.ndarray,
    source_features: List[str],
    target_features: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """Map features from source to target dataset by name matching.

    Parameters
    ----------
    X : np.ndarray
        Source features, shape (n_samples, n_source_features).
    source_features : list of str
        Feature names in source dataset.
    target_features : list of str
        Feature names in target dataset.

    Returns
    -------
    Tuple[np.ndarray, List[str]]
        Mapped features and the names of shared features used.
    """
    source_set = set(source_features)
    target_set = set(target_features)
    shared = sorted(source_set & target_set)

    if not shared:
        # Fallback: use positional mapping up to min dimension
        n = min(X.shape[1], len(target_features))
        return X[:, :n], target_features[:n]

    source_idx = [source_features.index(f) for f in shared]
    X_mapped = X[:, source_idx]

    logger.info(f"Feature mapping: {len(shared)}/{len(target_features)} shared features")
    return X_mapped, shared


def evaluate_cross_dataset(
    model,
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_target: np.ndarray,
    y_target: np.ndarray,
    source_features: List[str],
    target_features: List[str],
    source_name: str = "source",
    target_name: str = "target",
) -> Dict:
    """Evaluate cross-dataset generalization.

    Parameters
    ----------
    model : sklearn-compatible classifier
    X_source : np.ndarray
        Training features (source dataset).
    y_source : np.ndarray
        Training labels (source dataset).
    X_target : np.ndarray
        Test features (target dataset).
    y_target : np.ndarray
        Test labels (target dataset).
    source_features : list of str
    target_features : list of str
    source_name : str
    target_name : str

    Returns
    -------
    Dict
        In-distribution accuracy, cross-dataset accuracy,
        accuracy drop, and feature mapping info.
    """
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    # In-distribution performance (train/test on source)
    model.fit(X_source, y_source)
    y_pred_source = model.predict(X_source)
    id_acc = accuracy_score(y_source, y_pred_source)
    id_f1 = f1_score(y_source, y_pred_source, average="weighted", zero_division=0)

    # Cross-dataset: map features and predict
    X_target_mapped, shared_features = map_features(
        X_target, target_features, source_features
    )

    # Adjust source training data to shared features only
    source_idx = [source_features.index(f) for f in shared_features]
    X_source_shared = X_source[:, source_idx]

    model.fit(X_source_shared, y_source)
    y_pred_cross = model.predict(X_target_mapped)

    cd_acc = accuracy_score(y_target, y_pred_cross)
    cd_f1 = f1_score(y_target, y_pred_cross, average="weighted", zero_division=0)

    acc_drop = id_acc - cd_acc
    f1_drop = id_f1 - cd_f1

    report = classification_report(
        y_target, y_pred_cross, output_dict=True, zero_division=0
    )

    result = {
        "source_dataset": source_name,
        "target_dataset": target_name,
        "n_shared_features": len(shared_features),
        "shared_features": shared_features,
        "in_distribution": {
            "accuracy": round(float(id_acc), 6),
            "f1_weighted": round(float(id_f1), 6),
        },
        "cross_dataset": {
            "accuracy": round(float(cd_acc), 6),
            "f1_weighted": round(float(cd_f1), 6),
        },
        "performance_drop": {
            "accuracy_drop": round(float(acc_drop), 6),
            "f1_drop": round(float(f1_drop), 6),
            "accuracy_drop_pct": round(float(acc_drop / max(id_acc, 1e-8) * 100), 2),
            "f1_drop_pct": round(float(f1_drop / max(id_f1, 1e-8) * 100), 2),
        },
        "per_class_report": report,
    }

    logger.info(
        f"Cross-dataset {source_name} -> {target_name}: "
        f"ID acc={id_acc:.4f}, CD acc={cd_acc:.4f}, drop={acc_drop:.4f}"
    )

    return result


def plot_cross_dataset_generalization(
    results: List[Dict],
    save_path: str = "plots/cross_dataset_generalization.png",
):
    """Plot cross-dataset generalization results.

    Parameters
    ----------
    results : list of Dict
        Output from evaluate_cross_dataset().
    save_path : str
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: ID vs CD accuracy bar chart
    ax1 = axes[0]
    labels = [f"{r['source_dataset']}\n→ {r['target_dataset']}" for r in results]
    id_accs = [r["in_distribution"]["accuracy"] for r in results]
    cd_accs = [r["cross_dataset"]["accuracy"] for r in results]

    x = np.arange(len(labels))
    width = 0.35

    ax1.bar(x - width / 2, id_accs, width, label="In-Distribution", color="#4CAF50", alpha=0.8)
    ax1.bar(x + width / 2, cd_accs, width, label="Cross-Dataset", color="#F44336", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("Accuracy")
    ax1.set_title("In-Distribution vs Cross-Dataset Performance")
    ax1.legend()
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3, axis="y")

    # Right: Accuracy drop by feature count
    ax2 = axes[1]
    n_shared = [r["n_shared_features"] for r in results]
    drops = [r["performance_drop"]["accuracy_drop"] for r in results]
    colors = ["#4CAF50" if d < 0.1 else "#FF9800" if d < 0.3 else "#F44336" for d in drops]
    ax2.bar(range(len(drops)), drops, color=colors, edgecolor="black", alpha=0.8)
    ax2.set_xticks(range(len(drops)))
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("Accuracy Drop")
    ax2.set_title("Performance Degradation (Domain Shift)")
    ax2.grid(True, alpha=0.3, axis="y")

    for i, (d, n) in enumerate(zip(drops, n_shared)):
        ax2.text(i, d + 0.01, f"{d:.3f}\n({n} features)", ha="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved cross-dataset generalization plot to {save_path}")
