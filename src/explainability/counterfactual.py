"""
Counterfactual explanations for XAI-IDS predictions.

Uses DiCE (Diverse Counterfactual Explanations) to generate
minimum feature perturbations that would flip a model's prediction.
This provides actionable insights for SOC analysts.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("xai_ids.explainability.counterfactual")


def generate_counterfactuals(
    model,
    X_query: np.ndarray,
    feature_names: List[str],
    class_names: Optional[List[str]] = None,
    n_counterfactuals: int = 3,
    desired_class: str = "opposite",
    method: str = "random",
) -> Dict:
    """Generate counterfactual explanations for a query instance.

    Parameters
    ----------
    model : sklearn-compatible classifier
        Trained model with predict() and predict_proba().
    X_query : np.ndarray
        Single query instance, shape (1, n_features) or (n_features,).
    feature_names : list of str
        Names of the features.
    class_names : list of str, optional
        Human-readable class names.
    n_counterfactuals : int
        Number of counterfactuals to generate.
    desired_class : str
        Target class for counterfactual. "opposite" flips the prediction,
        or specify a class name/index.
    method : str
        Counterfactual generation method: "random", "genetic", or "kdtree".

    Returns
    -------
    Dict
        Contains original prediction, counterfactual features,
        feature changes, and new prediction.
    """
    X_query = np.asarray(X_query).reshape(1, -1)
    original_pred = model.predict(X_query)[0]
    original_proba = model.predict_proba(X_query)[0]

    n_features = X_query.shape[1]

    # Create a DataFrame for the query
    query_df = pd.DataFrame(X_query, columns=feature_names)

    # Determine target class
    if desired_class == "opposite":
        # Find the class with highest probability that isn't the predicted one
        sorted_classes = np.argsort(original_proba)[::-1]
        target_class = sorted_classes[1] if len(sorted_classes) > 1 else sorted_classes[0]
    elif isinstance(desired_class, str) and class_names is not None:
        target_class = class_names.index(desired_class)
    else:
        target_class = int(desired_class)

    # Try DiCE first, fall back to gradient-free approach
    try:
        import dice_ml
        from dice_ml import Dice

        dice_data = dice_ml.Data(
            dataframe=query_df,
            continuous_features=feature_names,
            outcome_name="_target",
        )
        dice_model = dice_ml.Model(model=model, backend="sklearn")
        dice_exp_gen = Dice(dice_data, dice_model, method=method)

        dice_exp = dice_exp_gen.generate_counterfactuals(
            query_df,
            total_CFs=n_counterfactuals,
            desired_class=target_class,
        )

        results = {
            "original_prediction": int(original_pred),
            "original_proba": {
                str(i): round(float(p), 6) for i, p in enumerate(original_proba)
            },
            "target_class": target_class,
            "counterfactuals": [],
            "method": "dice",
        }

        for cf in dice_exp.cf_examples_list[0].final_cfs_df.itertuples():
            cf_features = {feature_names[i]: round(float(cf[i + 1]), 6) for i in range(n_features)}
            changes = {}
            for j, feat in enumerate(feature_names):
                orig_val = float(X_query[0, j])
                cf_val = cf_features[feat]
                if abs(orig_val - cf_val) > 1e-6:
                    changes[feat] = {
                        "original": round(orig_val, 6),
                        "counterfactual": round(cf_val, 6),
                        "delta": round(cf_val - orig_val, 6),
                    }

            results["counterfactuals"].append({
                "features": cf_features,
                "changes": changes,
                "n_features_changed": len(changes),
            })

        return results

    except Exception as e:
        logger.warning(f"DiCE failed ({e}), using fallback counterfactual generation")
        return _generate_counterfactuals_fallback(
            model, X_query, feature_names, class_names, n_counterfactuals, target_class
        )


def _generate_counterfactuals_fallback(
    model,
    X_query: np.ndarray,
    feature_names: List[str],
    class_names: Optional[List[str]],
    n_counterfactuals: int,
    target_class: int,
) -> Dict:
    """Fallback counterfactual generation using gradient-free search.

    Iteratively perturbs features to find minimum changes needed
    to flip the model's prediction.
    """
    original_pred = model.predict(X_query)[0]
    original_proba = model.predict_proba(X_query)[0]

    cf_results = []
    rng = np.random.RandomState(42)

    for _ in range(n_counterfactuals):
        X_cf = X_query.copy()
        feature_order = rng.permutation(len(feature_names))

        for feat_idx in feature_order:
            current_pred = model.predict(X_cf)[0]
            if current_pred == target_class:
                break

            # Try perturbing this feature
            original_val = X_cf[0, feat_idx]
            step = 0.5
            best_delta = None

            for direction in [-1, 1]:
                for magnitude in [step, step * 2, step * 4]:
                    X_cf[0, feat_idx] = original_val + direction * magnitude
                    pred = model.predict(X_cf)[0]
                    if pred == target_class:
                        best_delta = direction * magnitude
                        break
                if best_delta is not None:
                    break

            if best_delta is not None:
                X_cf[0, feat_idx] = original_val + best_delta
            else:
                X_cf[0, feat_idx] = original_val

        cf_pred = model.predict(X_cf)[0]
        cf_proba = model.predict_proba(X_cf)[0]

        changes = {}
        for j, feat in enumerate(feature_names):
            orig_val = float(X_query[0, j])
            cf_val = float(X_cf[0, j])
            if abs(orig_val - cf_val) > 1e-6:
                changes[feat] = {
                    "original": round(orig_val, 6),
                    "counterfactual": round(cf_val, 6),
                    "delta": round(cf_val - orig_val, 6),
                }

        cf_results.append({
            "features": {feature_names[j]: round(float(X_cf[0, j]), 6) for j in range(len(feature_names))},
            "changes": changes,
            "n_features_changed": len(changes),
            "cf_prediction": int(cf_pred),
            "cf_confidence": round(float(np.max(cf_proba)), 6),
        })

    return {
        "original_prediction": int(original_pred),
        "original_proba": {
            str(i): round(float(p), 6) for i, p in enumerate(original_proba)
        },
        "target_class": target_class,
        "counterfactuals": cf_results,
        "method": "fallback_gradient_free",
    }


def generate_counterfactuals_for_classes(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    class_names: Optional[List[str]] = None,
    n_per_class: int = 3,
) -> Dict:
    """Generate counterfactuals for each attack class.

    Parameters
    ----------
    model : trained classifier
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        Test labels.
    feature_names : list of str
    class_names : list of str, optional
    n_per_class : int
        Number of counterfactuals per class.

    Returns
    -------
    Dict
        Counterfactuals organized by class.
    """
    unique_classes = np.unique(y_test)
    results = {}

    for cls in unique_classes:
        cls_indices = np.where(y_test == cls)[0]
        if len(cls_indices) == 0:
            continue

        # Pick first sample from this class
        sample_idx = cls_indices[0]
        X_sample = X_test[sample_idx:sample_idx + 1]

        cf = generate_counterfactuals(
            model,
            X_sample,
            feature_names,
            class_names,
            n_counterfactuals=n_per_class,
        )

        cls_name = class_names[int(cls)] if class_names else str(cls)
        results[cls_name] = cf

        logger.info(
            f"Generated {len(cf['counterfactuals'])} counterfactuals for class {cls_name}"
        )

    return results
