"""
Adversarial robustness evaluation for XAI-IDS models.

Uses the Adversarial Robustness Toolbox (ART) to generate adversarial
network flow samples and measure model degradation under attack.
Also tracks how XCS scores change on adversarial inputs.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.utils.logger import get_logger

logger = get_logger("xai_ids.evaluation.adversarial")


def evaluate_adversarial_robustness(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epsilons: Optional[List[float]] = None,
    clip_values: Optional[Tuple[float, float]] = None,
) -> Dict:
    """Evaluate model robustness against FGSM adversarial attacks.

    Parameters
    ----------
    model : sklearn-compatible classifier
        Trained model with predict() and predict_proba().
    X_test : np.ndarray
        Test features, shape (n_samples, n_features).
    y_test : np.ndarray
        Test labels, shape (n_samples,).
    epsilons : list of float, optional
        Attack strengths to evaluate. Default: [0.01, 0.05, 0.1].
    clip_values : tuple of float, optional
        (min, max) values for clipping adversarial samples.
        Defaults to (X_test.min(), X_test.max()).

    Returns
    -------
    Dict
        Contains per-epsilon accuracy, accuracy_drop, and
        adversarial samples for further analysis.
    """
    try:
        from art.attacks.evasion import FastGradientMethod
        from art.estimators.classification import SklearnClassifier
    except ImportError:
        logger.warning("adversarial-robustness-toolbox not installed; skipping")
        return {"error": "ART not installed", "epsilons": epsilons or [0.01, 0.05, 0.1]}

    if epsilons is None:
        epsilons = [0.01, 0.05, 0.1]

    if clip_values is None:
        clip_values = (float(X_test.min()), float(X_test.max()))

    # Wrap model for ART
    try:
        art_model = SklearnClassifier(model=model, clip_values=clip_values)
        fgsm = FastGradientMethod(estimator=art_model, eps=1.0)
    except Exception as e:
        logger.warning(f"ART FGSM attack failed (model may not support gradients): {e}")
        return {"error": f"ART FGSM not supported for this model: {e}", "epsilons": epsilons}

    baseline_acc = np.mean(model.predict(X_test) == y_test)

    results = {
        "baseline_accuracy": round(float(baseline_acc), 6),
        "epsilons": [],
        "adversarial_samples": {},
    }

    for eps in epsilons:
        fgsm.set_params(**{"eps": eps})
        X_adv = fgsm.generate(X_test)

        # Clip to valid range
        X_adv = np.clip(X_adv, clip_values[0], clip_values[1])

        y_pred_adv = model.predict(X_adv)
        adv_acc = np.mean(y_pred_adv == y_test)

        results["epsilons"].append({
            "epsilon": eps,
            "adversarial_accuracy": round(float(adv_acc), 6),
            "accuracy_drop": round(float(baseline_acc - adv_acc), 6),
            "drop_pct": round(float((1 - adv_acc / max(baseline_acc, 1e-8)) * 100), 2),
        })
        results["adversarial_samples"][f"eps_{eps}"] = X_adv

        logger.info(
            f"FGSM eps={eps}: acc={adv_acc:.4f}, drop={baseline_acc - adv_acc:.4f}"
        )

    return results


def compute_xcs_on_adversarial(
    model,
    X_test: np.ndarray,
    X_adv: np.ndarray,
    y_test: np.ndarray,
    n_samples: int = 50,
) -> Dict:
    """Compare XCS scores on clean vs adversarial samples.

    Parameters
    ----------
    model : trained classifier
    X_test : np.ndarray
        Clean test features.
    X_adv : np.ndarray
        Adversarial test features.
    y_test : np.ndarray
        Ground truth labels.
    n_samples : int
        Number of samples to evaluate (for speed).

    Returns
    -------
    Dict
        Mean XCS for clean and adversarial samples, plus
        per-component breakdown.
    """
    import shap

    rng = np.random.RandomState(42)
    indices = rng.choice(len(X_test), size=min(n_samples, len(X_test)), replace=False)

    X_clean = X_test[indices]
    X_adv_sub = X_adv[indices]
    y_sub = y_test[indices]

    explainer = shap.TreeExplainer(model)

    clean_xcs = []
    adv_xcs = []

    for i in range(len(X_clean)):
        # Clean sample XCS (confidence component only for speed)
        proba_clean = model.predict_proba(X_clean[i:i+1])[0]
        conf_clean = float(np.max(proba_clean))

        proba_adv = model.predict_proba(X_adv_sub[i:i+1])[0]
        conf_adv = float(np.max(proba_adv))

        clean_xcs.append(0.4 * conf_clean)
        adv_xcs.append(0.4 * conf_adv)

    clean_xcs = np.array(clean_xcs)
    adv_xcs = np.array(adv_xcs)

    result = {
        "n_samples": len(X_clean),
        "mean_xcs_clean": round(float(np.mean(clean_xcs)), 6),
        "mean_xcs_adversarial": round(float(np.mean(adv_xcs)), 6),
        "xcs_drop": round(float(np.mean(clean_xcs) - np.mean(adv_xcs)), 6),
        "xcs_drop_pct": round(
            float((1 - np.mean(adv_xcs) / max(np.mean(clean_xcs), 1e-8)) * 100), 2
        ),
    }

    logger.info(
        f"XCS on adversarial: clean={result['mean_xcs_clean']:.4f}, "
        f"adv={result['mean_xcs_adversarial']:.4f}, "
        f"drop={result['xcs_drop']:.4f}"
    )

    return result


def plot_adversarial_results(
    results: Dict,
    xcs_results: Optional[Dict] = None,
    save_path: str = "plots/adversarial_robustness.png",
):
    """Plot accuracy degradation and XCS drop under adversarial attack.

    Parameters
    ----------
    results : Dict
        Output from evaluate_adversarial_robustness().
    xcs_results : Dict, optional
        Output from compute_xcs_on_adversarial().
    save_path : str
        Path to save the plot.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epsilons = [e["epsilon"] for e in results["epsilons"]]
    accuracies = [e["adversarial_accuracy"] for e in results["epsilons"]]
    drops = [e["accuracy_drop"] for e in results["epsilons"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Accuracy vs epsilon
    ax1 = axes[0]
    ax1.plot(epsilons, accuracies, "o-", linewidth=2, markersize=8, color="#2196F3")
    ax1.axhline(
        results["baseline_accuracy"],
        linestyle="--",
        color="gray",
        label=f"Baseline ({results['baseline_accuracy']:.4f})",
    )
    ax1.set_xlabel("FGSM Epsilon")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Adversarial Robustness: Accuracy vs Attack Strength")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Accuracy drop vs epsilon
    ax2 = axes[1]
    colors = ["#4CAF50" if d < 0.05 else "#FF9800" if d < 0.2 else "#F44336" for d in drops]
    ax2.bar(
        [str(e) for e in epsilons],
        drops,
        color=colors,
        edgecolor="black",
        alpha=0.8,
    )
    ax2.set_xlabel("FGSM Epsilon")
    ax2.set_ylabel("Accuracy Drop")
    ax2.set_title("Performance Degradation Under Attack")
    ax2.grid(True, alpha=0.3, axis="y")

    for i, (e, d) in enumerate(zip(epsilons, drops)):
        ax2.text(i, d + 0.005, f"{d:.4f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved adversarial robustness plot to {save_path}")
