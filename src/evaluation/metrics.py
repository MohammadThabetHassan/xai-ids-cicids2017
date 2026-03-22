"""
Model evaluation and metrics computation for XAI-IDS.

Computes accuracy, precision, recall, F1-score, confusion matrices,
and generates publication-quality visualizations.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_predict, cross_val_score, StratifiedKFold

from src.utils.logger import get_logger

logger = get_logger("xai_ids.evaluation")

# Plot styling
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    }
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    label_names: Optional[List[str]] = None,
) -> Dict:
    """
    Compute comprehensive classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    model_name : str
        Name of the model.
    label_names : List[str], optional
        Human-readable label names.

    Returns
    -------
    Dict
        Dictionary with all metrics.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Get actual labels present in both y_true and y_pred
    present_labels = sorted(set(y_true) | set(y_pred))

    # Handle label_names being None or having different length
    if label_names is not None and len(present_labels) > 0:
        max_label = max(present_labels)
        if max_label < len(label_names):
            present_label_names = [label_names[i] for i in present_labels]
        else:
            present_label_names = None
    else:
        present_label_names = None

    if present_label_names:
        report = classification_report(
            y_true,
            y_pred,
            labels=present_labels,
            target_names=present_label_names,
            zero_division=0,
            output_dict=True,
        )

        report_str = classification_report(
            y_true,
            y_pred,
            labels=present_labels,
            target_names=present_label_names,
            zero_division=0,
        )
    else:
        report = classification_report(
            y_true,
            y_pred,
            labels=present_labels,
            zero_division=0,
            output_dict=True,
        )

        report_str = classification_report(
            y_true,
            y_pred,
            labels=present_labels,
            zero_division=0,
        )

    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "classification_report": report,
        "classification_report_str": report_str,
    }

    logger.info(f"\n{model_name} Evaluation Results:")
    logger.info(f"  Accuracy:  {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall:    {recall:.4f}")
    logger.info(f"  F1-Score:  {f1:.4f}")

    return metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    label_names: Optional[List[str]] = None,
    save_dir: str = "outputs/figures",
    normalize: bool = False,
) -> str:
    """
    Plot and save a confusion matrix heatmap.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix.
    model_name : str
        Model name for the title.
    label_names : List[str], optional
        Class labels.
    save_dir : str
        Directory to save the figure.
    normalize : bool
        Whether to normalize the confusion matrix.

    Returns
    -------
    str
        Path to the saved figure.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    if normalize:
        cm_plot = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_plot = np.nan_to_num(cm_plot)
        fmt = ".2f"
        suffix = "_normalized"
        title = f"{model_name} - Normalized Confusion Matrix"
    else:
        cm_plot = cm
        fmt = "d"
        suffix = ""
        title = f"{model_name} - Confusion Matrix"

    n_classes = cm_plot.shape[0]
    fig_size = max(8, n_classes * 0.8)

    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    sns.heatmap(
        cm_plot,
        annot=True if n_classes <= 20 else False,
        fmt=fmt if n_classes <= 20 else "",
        cmap="Blues",
        xticklabels=label_names if label_names else range(n_classes),
        yticklabels=label_names if label_names else range(n_classes),
        ax=ax,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    filename = f"confusion_matrix_{model_name.lower().replace(' ', '_')}{suffix}.png"
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, bbox_inches="tight", dpi=150)
    plt.close(fig)

    logger.info(f"Saved confusion matrix: {filepath}")
    return filepath


def plot_model_comparison(
    metrics_list: List[Dict],
    save_dir: str = "outputs/figures",
) -> str:
    """
    Create a grouped bar chart comparing all models.

    Parameters
    ----------
    metrics_list : List[Dict]
        List of metric dictionaries from compute_metrics().
    save_dir : str
        Directory to save the figure.

    Returns
    -------
    str
        Path to the saved figure.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    model_names = [m["model"] for m in metrics_list]
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score"]
    display_names = ["Accuracy", "Precision", "Recall", "F1-Score"]

    x = np.arange(len(model_names))
    width = 0.18
    offsets = np.arange(len(metrics_to_plot)) - (len(metrics_to_plot) - 1) / 2

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

    for i, (metric, display_name) in enumerate(zip(metrics_to_plot, display_names)):
        values = [m[metric] for m in metrics_list]
        bars = ax.bar(
            x + offsets[i] * width,
            values,
            width,
            label=display_name,
            color=colors[i],
            edgecolor="white",
            linewidth=0.5,
        )
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
            )

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [n.replace("_", " ").title() for n in model_names],
        fontsize=10,
    )
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    filepath = os.path.join(save_dir, "model_comparison.png")
    fig.savefig(filepath, bbox_inches="tight", dpi=150)
    plt.close(fig)

    logger.info(f"Saved model comparison chart: {filepath}")
    return filepath


def save_metrics_csv(
    metrics_list: List[Dict],
    save_path: str = "outputs/results_metrics.csv",
) -> str:
    """
    Save all model metrics to a CSV file.

    Parameters
    ----------
    metrics_list : List[Dict]
        List of metric dictionaries.
    save_path : str
        Output CSV path.

    Returns
    -------
    str
        Path to the saved CSV.
    """
    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)

    rows = []
    for m in metrics_list:
        rows.append(
            {
                "Model": m["model"],
                "Accuracy": round(m["accuracy"], 4),
                "Precision": round(m["precision"], 4),
                "Recall": round(m["recall"], 4),
                "F1-Score": round(m["f1_score"], 4),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)

    logger.info(f"Saved metrics CSV: {save_path}")
    logger.info(f"\n{df.to_string(index=False)}")

    return save_path


def save_classification_reports(
    metrics_list: List[Dict],
    save_dir: str = "outputs/reports",
) -> None:
    """
    Save detailed classification reports to text files.

    Parameters
    ----------
    metrics_list : List[Dict]
        List of metric dictionaries.
    save_dir : str
        Directory to save reports.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for m in metrics_list:
        model_name = m["model"]
        report_path = os.path.join(save_dir, f"classification_report_{model_name}.txt")
        with open(report_path, "w") as f:
            f.write(f"Classification Report: {model_name}\n")
            f.write("=" * 60 + "\n\n")
            f.write(m["classification_report_str"])
        logger.info(f"Saved classification report: {report_path}")


def evaluate_all_models(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: Optional[List[str]] = None,
    figures_dir: str = "outputs/figures",
    reports_dir: str = "outputs/reports",
    metrics_path: str = "outputs/results_metrics.csv",
) -> List[Dict]:
    """
    Evaluate all trained models and generate outputs.

    Parameters
    ----------
    models : Dict[str, Any]
        Dictionary of trained models.
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        Test labels.
    label_names : List[str], optional
        Human-readable label names.
    figures_dir : str
        Directory for figures.
    reports_dir : str
        Directory for reports.
    metrics_path : str
        Path for metrics CSV.

    Returns
    -------
    List[Dict]
        List of metrics dictionaries.
    """
    logger.info("=" * 60)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 60)

    all_metrics = []

    for model_name, model in models.items():
        logger.info(f"\nEvaluating {model_name}...")

        y_pred = model.predict(X_test)

        # Compute metrics
        metrics = compute_metrics(
            y_true=y_test, y_pred=y_pred, model_name=model_name, label_names=label_names
        )
        all_metrics.append(metrics)

        # Plot confusion matrices
        plot_confusion_matrix(
            metrics["confusion_matrix"],
            model_name,
            label_names=label_names,
            save_dir=figures_dir,
            normalize=False,
        )
        plot_confusion_matrix(
            metrics["confusion_matrix"],
            model_name,
            label_names=label_names,
            save_dir=figures_dir,
            normalize=True,
        )

    # Save comparison chart
    plot_model_comparison(all_metrics, save_dir=figures_dir)

    # Save metrics CSV
    save_metrics_csv(all_metrics, save_path=metrics_path)

    # Save classification reports
    save_classification_reports(all_metrics, save_dir=reports_dir)

    return all_metrics


def run_cross_validation(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str = "f1_weighted",
) -> Dict:
    """
    Run stratified k-fold cross-validation.

    Parameters
    ----------
    model : sklearn estimator
        Model to validate.
    X : np.ndarray
        Features.
    y : np.ndarray
        Labels.
    cv : int
        Number of folds (default: 5).
    scoring : str
        Scoring metric.

    Returns
    -------
    Dict
        Cross-validation results.
    """
    logger.info(f"Running {cv}-fold cross-validation...")
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    scores = cross_val_score(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)

    results = {
        "cv_folds": cv,
        "scoring": scoring,
        "scores": scores.tolist(),
        "mean": scores.mean(),
        "std": scores.std(),
        "min": scores.min(),
        "max": scores.max(),
    }

    logger.info(
        f"  {cv}-Fold CV {scoring}: {scores.mean():.4f} (+/- {scores.std():.4f})"
    )
    logger.info(f"  Individual folds: {[f'{s:.4f}' for s in scores]}")

    return results


def plot_precision_recall_curves(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: List[str],
    save_dir: str = "outputs/figures",
) -> List[str]:
    """
    Plot precision-recall curves for each class.

    Parameters
    ----------
    models : Dict[str, Any]
        Dictionary of trained models.
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        Test labels.
    label_names : List[str]
        Human-readable class names.
    save_dir : str
        Directory to save figures.

    Returns
    -------
    List[str]
        Paths to saved figures.
    """
    from sklearn.preprocessing import label_binarize

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    saved_paths = []

    y_test_bin = label_binarize(y_test, classes=range(len(label_names)))

    for model_name, model in models.items():
        logger.info(f"Computing PR curve for {model_name}...")

        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(X_test)
        else:
            continue

        n_classes = len(label_names)
        fig, axes = plt.subplots(3, 5, figsize=(18, 12))
        axes = axes.flatten()

        for i in range(min(n_classes, 15)):
            if y_test_bin.shape[1] > i:
                precision, recall, _ = precision_recall_curve(
                    y_test_bin[:, i], y_scores[:, i]
                )
                f1 = (
                    2
                    * precision.mean()
                    * recall.mean()
                    / (precision.mean() + recall.mean() + 1e-10)
                )

                axes[i].plot(recall, precision, "b-", linewidth=2)
                axes[i].fill_between(recall, precision, alpha=0.3)
                axes[i].set_xlabel("Recall")
                axes[i].set_ylabel("Precision")
                axes[i].set_title(f"{label_names[i]}\nF1: {f1:.2f}")
                axes[i].set_xlim([0, 1])
                axes[i].set_ylim([0, 1])
                axes[i].grid(True, alpha=0.3)

        for i in range(n_classes, 15):
            axes[i].axis("off")

        plt.suptitle(
            f"Precision-Recall Curves: {model_name}", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()

        filepath = os.path.join(
            save_dir, f"pr_curves_{model_name.lower().replace(' ', '_')}.png"
        )
        fig.savefig(filepath, bbox_inches="tight", dpi=150)
        plt.close(fig)

        saved_paths.append(filepath)
        logger.info(f"Saved PR curves: {filepath}")

    return saved_paths


def plot_calibration_curves(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: List[str],
    save_dir: str = "outputs/figures",
    n_bins: int = 10,
) -> List[str]:
    """
    Plot calibration curves for each class.

    Parameters
    ----------
    models : Dict[str, Any]
        Dictionary of trained models.
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        Test labels.
    label_names : List[str]
        Human-readable class names.
    save_dir : str
        Directory to save figures.
    n_bins : int
        Number of bins for calibration curve.

    Returns
    -------
    List[str]
        Paths to saved figures.
    """
    from sklearn.preprocessing import label_binarize

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    saved_paths = []

    y_test_bin = label_binarize(y_test, classes=range(len(label_names)))

    for model_name, model in models.items():
        logger.info(f"Computing calibration for {model_name}...")

        if not hasattr(model, "predict_proba"):
            logger.warning(f"{model_name} lacks predict_proba, skipping calibration")
            continue

        y_scores = model.predict_proba(X_test)

        fig, axes = plt.subplots(3, 5, figsize=(18, 12))
        axes = axes.flatten()

        for i in range(min(len(label_names), 15)):
            if y_test_bin.shape[1] > i:
                try:
                    frac_pos, pred_prob = calibration_curve(
                        y_test_bin[:, i], y_scores[:, i], n_bins=n_bins
                    )
                    axes[i].plot([0, 1], [0, 1], "k--", label="Perfect")
                    axes[i].plot(pred_prob, frac_pos, "o-", label="Model")
                    axes[i].set_xlabel("Mean Predicted Probability")
                    axes[i].set_ylabel("Fraction of Positives")
                    axes[i].set_title(f"{label_names[i]}")
                    axes[i].legend(loc="lower right", fontsize=7)
                    axes[i].grid(True, alpha=0.3)
                    axes[i].set_xlim([0, 1])
                    axes[i].set_ylim([0, 1])
                except Exception as e:
                    axes[i].text(0.5, 0.5, f"Error: {str(e)[:30]}", ha="center")

        for i in range(len(label_names), 15):
            axes[i].axis("off")

        plt.suptitle(
            f"Calibration Curves: {model_name}", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()

        filepath = os.path.join(
            save_dir, f"calibration_{model_name.lower().replace(' ', '_')}.png"
        )
        fig.savefig(filepath, bbox_inches="tight", dpi=150)
        plt.close(fig)

        saved_paths.append(filepath)
        logger.info(f"Saved calibration curves: {filepath}")

    return saved_paths


def generate_failure_analysis(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: List[str],
    save_dir: str = "outputs/reports",
) -> Dict:
    """
    Generate failure analysis report showing which classes are confused with each other.

    Parameters
    ----------
    models : Dict[str, Any]
        Dictionary of trained models.
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        Test labels.
    label_names : List[str]
        Human-readable class names.
    save_dir : str
        Directory to save reports.

    Returns
    -------
    Dict
        Failure analysis results.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    results = {}

    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        report = classification_report(
            y_test, y_pred, target_names=label_names, output_dict=True, zero_division=0
        )

        failing_classes = []
        for i, name in enumerate(label_names):
            if report[name]["recall"] < 0.3:
                failing_classes.append(
                    {
                        "class": name,
                        "recall": report[name]["recall"],
                        "precision": report[name]["precision"],
                        "f1": report[name]["f1-score"],
                    }
                )

        top_confusions = []
        for i in range(len(label_names)):
            for j in range(len(label_names)):
                if i != j and cm[i, j] >= 5:
                    top_confusions.append(
                        {
                            "true": label_names[i],
                            "predicted": label_names[j],
                            "count": int(cm[i, j]),
                        }
                    )
        top_confusions.sort(key=lambda x: x["count"], reverse=True)
        top_confusions = top_confusions[:10]

        results[model_name] = {
            "failing_classes": failing_classes,
            "top_confusions": top_confusions,
            "macro_recall": report["macro avg"]["recall"],
            "macro_f1": report["macro avg"]["f1-score"],
        }

    report_path = os.path.join(save_dir, "failure_analysis.txt")
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("FAILURE ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")

        for model_name, data in results.items():
            f.write(f"Model: {model_name}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Macro Recall: {data['macro_recall']:.4f}\n")
            f.write(f"Macro F1: {data['macro_f1']:.4f}\n\n")

            f.write("Failing Classes (recall < 30%):\n")
            if data["failing_classes"]:
                for fc in data["failing_classes"]:
                    f.write(
                        f"  - {fc['class']}: P={fc['precision']:.2f}, R={fc['recall']:.2f}, F1={fc['f1']:.2f}\n"
                    )
            else:
                f.write("  None\n")

            f.write("\nTop Confusions:\n")
            for conf in data["top_confusions"]:
                f.write(f"  {conf['true']} -> {conf['predicted']}: {conf['count']}\n")

            f.write("\n" + "=" * 60 + "\n\n")

    logger.info(f"Saved failure analysis: {report_path}")
    return results
