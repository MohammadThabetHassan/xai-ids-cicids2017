"""
Explainability module for XAI-IDS.

Implements SHAP (global) and LIME (local) explanations
for the trained intrusion detection models.
"""

import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

from src.utils.logger import get_logger

logger = get_logger("xai_ids.explainability")

# Suppress verbose warnings during SHAP computation
warnings.filterwarnings("ignore", category=UserWarning, module="shap")


def compute_shap_explanations(
    model: Any,
    X_sample: np.ndarray,
    feature_names: List[str],
    model_name: str,
    save_dir: str = "outputs/figures",
    max_display: int = 20,
) -> Optional[shap.Explanation]:
    """
    Compute and plot SHAP global explanations.

    Parameters
    ----------
    model : Any
        Trained model.
    X_sample : np.ndarray
        Sample of data for SHAP analysis (should be ~100-500 rows).
    feature_names : List[str]
        Feature names.
    model_name : str
        Name of the model.
    save_dir : str
        Directory to save SHAP plots.
    max_display : int
        Maximum features to display.

    Returns
    -------
    Optional[shap.Explanation]
        SHAP explanation object, or None on failure.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Computing SHAP explanations for {model_name}...")
    logger.info(f"  Sample size: {X_sample.shape[0]} rows")

    try:
        # Use TreeExplainer for tree-based models, KernelExplainer otherwise
        if hasattr(model, "feature_importances_"):
            logger.info("  Using TreeExplainer")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        else:
            logger.info("  Using KernelExplainer (this may take a while)")
            # Use a small background sample for KernelExplainer
            bg_sample = X_sample[:50]
            explainer = shap.KernelExplainer(model.predict_proba, bg_sample)
            shap_values = explainer.shap_values(X_sample[:100])

        # Compute mean absolute SHAP values per feature
        # Handle both list-of-arrays (multi-class) and single array cases
        sv = np.array(shap_values) if isinstance(shap_values, list) else shap_values
        if sv.ndim == 3:
            # Shape: (n_classes, n_samples, n_features) or (n_samples, n_features, n_classes)
            # Average across classes first, then across samples
            mean_abs_shap = (
                np.abs(sv).mean(axis=(0, 1))
                if sv.shape[0] < sv.shape[1]
                else np.abs(sv).mean(axis=0).mean(axis=-1)
            )
            # Ensure we have one value per feature
            if mean_abs_shap.shape[0] != len(feature_names):
                mean_abs_shap = np.abs(sv).mean(axis=0).mean(axis=0)
        else:
            # 2D: (n_samples, n_features)
            mean_abs_shap = np.abs(sv).mean(axis=0)

        # Flatten to 1D if needed
        mean_abs_shap = mean_abs_shap.ravel()
        if len(mean_abs_shap) != len(feature_names):
            logger.warning(
                f"  SHAP shape mismatch: {len(mean_abs_shap)} vs {len(feature_names)} features"
            )
            mean_abs_shap = mean_abs_shap[: len(feature_names)]

        # SHAP Summary Plot
        logger.info("  Generating SHAP summary plot...")
        feature_importance = sorted(
            zip(feature_names, mean_abs_shap.tolist()), key=lambda x: x[1], reverse=True
        )[:max_display]
        names, values = zip(*feature_importance)

        fig, ax = plt.subplots(figsize=(12, 8))
        plt.title(f"SHAP Summary Plot - {model_name}", fontsize=14, fontweight="bold")
        y_pos = np.arange(len(names))
        ax.barh(y_pos, values, color="#2196F3", edgecolor="white")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Mean |SHAP value|", fontsize=11)
        ax.invert_yaxis()

        plt.tight_layout()
        summary_path = os.path.join(
            save_dir, f"shap_summary_{model_name.lower().replace(' ', '_')}.png"
        )
        fig.savefig(summary_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        logger.info(f"  Saved SHAP summary plot: {summary_path}")

        # SHAP Feature Importance Bar Plot
        logger.info("  Generating SHAP feature importance plot...")
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.title(
            f"SHAP Feature Importance - {model_name}",
            fontsize=14,
            fontweight="bold",
        )

        feature_importance = sorted(
            zip(feature_names, mean_abs_shap.tolist()), key=lambda x: x[1], reverse=True
        )[:max_display]
        names, values = zip(*feature_importance)

        cmap = plt.colormaps.get_cmap("RdYlBu_r")
        colors = cmap(np.linspace(0.2, 0.8, len(names)))
        y_pos = np.arange(len(names))
        ax.barh(y_pos, values, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Mean |SHAP value| (Feature Importance)", fontsize=11)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        importance_path = os.path.join(
            save_dir,
            f"shap_feature_importance_{model_name.lower().replace(' ', '_')}.png",
        )
        fig.savefig(importance_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        logger.info(f"  Saved SHAP feature importance: {importance_path}")

        return shap_values

    except Exception as e:
        logger.error(f"SHAP computation failed for {model_name}: {e}")
        return None


def compute_lime_explanations(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    label_names: List[str],
    model_name: str,
    save_dir: str = "outputs/figures",
    reports_dir: str = "outputs/reports",
) -> None:
    """
    Generate LIME local explanations for selected instances.

    Explains at least one correctly classified and one misclassified sample.

    Parameters
    ----------
    model : Any
        Trained model with predict_proba method.
    X_train : np.ndarray
        Training data (used for LIME background).
    X_test : np.ndarray
        Test data.
    y_test : np.ndarray
        True test labels.
    feature_names : List[str]
        Feature names.
    label_names : List[str]
        Class label names.
    model_name : str
        Name of the model.
    save_dir : str
        Directory for LIME figures.
    reports_dir : str
        Directory for LIME text reports.
    """
    from lime.lime_tabular import LimeTabularExplainer

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    Path(reports_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Computing LIME explanations for {model_name}...")

    try:
        # Create LIME explainer
        explainer = LimeTabularExplainer(
            training_data=X_train,
            feature_names=feature_names,
            class_names=label_names,
            mode="classification",
            random_state=42,
        )

        # Get predictions
        y_pred = model.predict(X_test)

        # Find correctly classified and misclassified samples
        correct_mask = y_pred == y_test
        misclassified_mask = ~correct_mask

        # Explain a correctly classified sample
        if correct_mask.any():
            correct_idx = np.where(correct_mask)[0][0]
            logger.info(
                f"  Explaining correctly classified sample (index {correct_idx})"
            )
            _explain_single_instance(
                explainer,
                model,
                X_test,
                y_test,
                y_pred,
                correct_idx,
                feature_names,
                label_names,
                model_name,
                "correct",
                save_dir,
                reports_dir,
            )

        # Explain a misclassified sample
        if misclassified_mask.any():
            misclassified_idx = np.where(misclassified_mask)[0][0]
            logger.info(
                f"  Explaining misclassified sample (index {misclassified_idx})"
            )
            _explain_single_instance(
                explainer,
                model,
                X_test,
                y_test,
                y_pred,
                misclassified_idx,
                feature_names,
                label_names,
                model_name,
                "misclassified",
                save_dir,
                reports_dir,
            )
        else:
            logger.info("  No misclassified samples found for LIME explanation")

    except Exception as e:
        logger.error(f"LIME computation failed for {model_name}: {e}")


def _explain_single_instance(
    explainer,
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    idx: int,
    feature_names: List[str],
    label_names: List[str],
    model_name: str,
    classification_type: str,
    save_dir: str,
    reports_dir: str,
) -> None:
    """
    Generate LIME explanation for a single instance.
    """
    instance = X_test[idx]
    true_label = int(y_test[idx])
    pred_label = int(y_pred[idx])

    true_name = (
        label_names[true_label] if true_label < len(label_names) else str(true_label)
    )
    pred_name = (
        label_names[pred_label] if pred_label < len(label_names) else str(pred_label)
    )

    logger.info(f"    True: {true_name}, Predicted: {pred_name}")

    # Generate explanation
    exp = explainer.explain_instance(
        instance,
        model.predict_proba,
        num_features=10,
        top_labels=min(3, len(label_names)),
    )

    # Save as figure
    fig = exp.as_pyplot_figure(label=pred_label)
    fig.suptitle(
        f"LIME Explanation - {model_name}\n"
        f"({classification_type.title()}: True={true_name}, Pred={pred_name})",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()

    fig_path = os.path.join(
        save_dir,
        f"lime_{classification_type}_{model_name.lower().replace(' ', '_')}.png",
    )
    fig.savefig(fig_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"    Saved LIME figure: {fig_path}")

    # Save text report
    report_path = os.path.join(
        reports_dir,
        f"lime_{classification_type}_{model_name.lower().replace(' ', '_')}.txt",
    )
    with open(report_path, "w") as f:
        f.write(f"LIME Explanation - {model_name}\n")
        f.write(f"Classification Type: {classification_type}\n")
        f.write(f"True Label: {true_name} ({true_label})\n")
        f.write(f"Predicted Label: {pred_name} ({pred_label})\n")
        f.write("=" * 60 + "\n\n")
        f.write("Feature contributions:\n")
        for feature, weight in exp.as_list(label=pred_label):
            f.write(f"  {feature}: {weight:.4f}\n")

    logger.info(f"    Saved LIME report: {report_path}")


def run_explainability(
    models: Dict[str, Any],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    label_names: List[str],
    figures_dir: str = "outputs/figures",
    reports_dir: str = "outputs/reports",
    shap_sample_size: int = 500,
) -> None:
    """
    Run full explainability analysis on all models.

    Parameters
    ----------
    models : Dict[str, Any]
        Dictionary of trained models.
    X_train : np.ndarray
        Training data.
    X_test : np.ndarray
        Test data.
    y_test : np.ndarray
        Test labels.
    feature_names : List[str]
        Feature names.
    label_names : List[str]
        Class label names.
    figures_dir : str
        Directory for figures.
    reports_dir : str
        Directory for reports.
    shap_sample_size : int
        Number of samples for SHAP analysis.
    """
    logger.info("=" * 60)
    logger.info("EXPLAINABILITY ANALYSIS")
    logger.info("=" * 60)

    # Subsample for SHAP efficiency
    n_shap = min(shap_sample_size, X_test.shape[0])
    rng = np.random.RandomState(42)
    shap_indices = rng.choice(X_test.shape[0], n_shap, replace=False)
    X_shap = X_test[shap_indices]

    for model_name, model in models.items():
        logger.info(f"\n{'=' * 40}")
        logger.info(f"Explaining: {model_name}")
        logger.info(f"{'=' * 40}")

        # SHAP global explanations
        compute_shap_explanations(
            model,
            X_shap,
            feature_names,
            model_name,
            save_dir=figures_dir,
        )

        # LIME local explanations
        if hasattr(model, "predict_proba"):
            compute_lime_explanations(
                model,
                X_train,
                X_test,
                y_test,
                feature_names,
                label_names,
                model_name,
                save_dir=figures_dir,
                reports_dir=reports_dir,
            )
        else:
            logger.warning(f"Skipping LIME for {model_name}: no predict_proba method")

    logger.info("\nExplainability analysis complete")
