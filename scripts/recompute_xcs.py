"""
Recompute XCS (XAI Confidence Score) with full LIME formula.

This script loads committed model artifacts, generates synthetic test data,
and computes the complete XCS metric:
    XCS = 0.4*Conf + 0.3*(1-SHAP_Instability) + 0.3*Jaccard(SHAP,LIME)

Previous CSVs had jaccard_sl=0 for all samples because LIME never ran
per-sample. This fixes that by actually computing LIME explanations.
"""

import csv
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASETS = ["CICIDS2017", "UNSWNB15", "CICIDS2018"]
N_SAMPLES = 100
NOISE_STD = 0.05
N_NEIGHBOURS = 5
LIME_NUM_FEATURES = 5
LIME_NUM_SAMPLES = 200
XCS_THRESHOLD = 0.3
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_lime_feature_name(condition_str: str, feature_names: list[str]) -> str | None:
    """Extract the raw feature name from a LIME condition string.

    LIME wraps feature names in human-readable conditions like
    ``"Fwd IAT Mean <= 0.05"``.  We find which known feature name appears
    in the condition string.
    """
    for feat in feature_names:
        if feat in condition_str:
            return feat
    return None


def _top_k_feature_names(explanation, k: int, feature_names: list[str]) -> set[str]:
    """Return the set of top-k feature names from a LIME explanation."""
    name_to_weight = {}
    as_map = explanation.as_map()

    for cls_id in as_map:
        for feat_idx, weight in as_map[cls_id]:
            feat_name = feature_names[int(feat_idx)]
            name_to_weight[feat_name] = name_to_weight.get(feat_name, 0.0) + abs(weight)

    sorted_feats = sorted(name_to_weight, key=lambda k: name_to_weight[k], reverse=True)
    return set(sorted_feats[:k])


def jaccard(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


# ---------------------------------------------------------------------------
# Main XCS computation
# ---------------------------------------------------------------------------

def compute_xcs_for_dataset(dataset: str) -> pd.DataFrame:
    """Compute XCS for a synthetic test set of *dataset*."""
    import shap
    import lime.lime_tabular

    rng = np.random.RandomState(RANDOM_SEED)

    # Load artifacts
    model_path = f"models/xgb_{dataset}.joblib"
    scaler_path = f"models/scaler_{dataset}.joblib"
    le_path = f"models/label_encoder_{dataset}.joblib"
    feat_path = f"models/features_{dataset}.joblib"

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    le = joblib.load(le_path)
    feature_names = joblib.load(feat_path)

    n_features = len(feature_names)
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset}  |  Features: {n_features}  |  Classes: {len(le.classes_)}")
    print(f"{'='*60}")

    # Generate synthetic data: random values scaled to reasonable ranges
    X_raw = rng.randn(N_SAMPLES, n_features).astype(np.float32)
    X_scaled = scaler.transform(X_raw)

    # Predict
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)
    confidence = np.max(y_proba, axis=1)

    # SHAP explainer (TreeExplainer works directly on XGBoost)
    explainer_shap = shap.TreeExplainer(model)
    shap_values_orig = explainer_shap.shap_values(X_scaled)

    # Handle multi-class SHAP: shape (n_samples, n_features, n_classes)
    if shap_values_orig.ndim == 3:
        shap_vals = np.array([shap_values_orig[i, :, y_pred[i]] for i in range(N_SAMPLES)])
    elif isinstance(shap_values_orig, list):
        shap_vals = np.array([shap_values_orig[p][i] for i, p in enumerate(y_pred)])
    else:
        shap_vals = shap_values_orig  # binary case (n_samples, n_features)

    # SHAP Instability: perturb input, compute std of |SHAP| across neighbours
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
    mean_abs_shap_total = np.mean(mean_abs_shap)
    if mean_abs_shap_total == 0:
        mean_abs_shap_total = 1e-8

    instability_scores = []
    for i in range(N_SAMPLES):
        neighbour_shaps = []
        for _ in range(N_NEIGHBOURS):
            noise = rng.normal(0, NOISE_STD, X_scaled.shape[1]).astype(np.float32)
            X_noisy = X_scaled[i:i+1] + noise
            sv = explainer_shap.shap_values(X_noisy)
            if sv.ndim == 3:
                sv = sv[0, :, y_pred[i]]
            elif isinstance(sv, list):
                sv = sv[y_pred[i]][0]
            else:
                sv = sv[0]
            neighbour_shaps.append(np.abs(sv))

        neighbour_shaps = np.array(neighbour_shaps)
        std_shap = np.std(neighbour_shaps, axis=0).mean()
        instab_norm = min(std_shap / mean_abs_shap_total, 0.5)
        instability_scores.append(instab_norm)

    instability_scores = np.array(instability_scores)

    # LIME: per-sample explanations for Jaccard
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        X_scaled,
        feature_names=feature_names,
        mode="classification",
    )

    jaccard_scores = []
    for i in range(N_SAMPLES):
        # SHAP top-5 features for this sample
        shap_top5_idx = np.argsort(np.abs(shap_vals[i]))[-LIME_NUM_FEATURES:]
        shap_top5_names = set()
        for j in shap_top5_idx:
            shap_top5_names.add(feature_names[j])

        # LIME explanation
        exp = lime_explainer.explain_instance(
            X_scaled[i],
            model.predict_proba,
            num_features=LIME_NUM_FEATURES,
            num_samples=LIME_NUM_SAMPLES,
        )
        lime_top5_names = _top_k_feature_names(exp, LIME_NUM_FEATURES, feature_names)

        jac = jaccard(shap_top5_names, lime_top5_names)
        jaccard_scores.append(jac)

        if (i + 1) % 20 == 0:
            print(f"  LIME progress: {i+1}/{N_SAMPLES}")

    jaccard_scores = np.array(jaccard_scores)

    # Compute XCS
    xcs = 0.4 * confidence + 0.3 * (1 - instability_scores) + 0.3 * jaccard_scores
    flag_review = xcs < XCS_THRESHOLD

    # Build result DataFrame
    classes = le.classes_
    true_labels = ["Synthetic"] * N_SAMPLES  # synthetic data, no real labels

    rows = []
    for i in range(N_SAMPLES):
        rows.append({
            "sample_id": i,
            "true_class": true_labels[i],
            "pred_class": classes[y_pred[i]],
            "correct": "N/A",
            "confidence": round(float(confidence[i]), 6),
            "shap_instability": round(float(instability_scores[i]), 6),
            "jaccard_sl": round(float(jaccard_scores[i]), 6),
            "xcs": round(float(xcs[i]), 6),
            "flag_review": str(flag_review[i]),
        })

    df = pd.DataFrame(rows)

    # Save
    out_path = Path(f"explanations/xcs_{dataset}_v2.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")

    # Summary
    print(f"\n  Summary for {dataset}:")
    print(f"    Mean XCS:          {df['xcs'].mean():.4f}")
    print(f"    Mean Confidence:   {df['confidence'].mean():.4f}")
    print(f"    Mean SHAP Instab:  {df['shap_instability'].mean():.4f}")
    print(f"    Mean Jaccard:      {df['jaccard_sl'].mean():.4f}")
    print(f"    Flagged (XCS<0.3): {df['flag_review'].value_counts().get('True', 0)} / {N_SAMPLES}")
    non_zero_jac = (df['jaccard_sl'] > 0).sum()
    print(f"    Non-zero Jaccard:  {non_zero_jac} / {N_SAMPLES}")

    return df


def main():
    print("XCS Recomputation — Full Formula with LIME")
    print("=" * 60)

    all_results = {}
    for ds in DATASETS:
        start = time.time()
        df = compute_xcs_for_dataset(ds)
        all_results[ds] = df
        elapsed = time.time() - start
        print(f"  Time: {elapsed:.1f}s")

    # Final summary table
    print(f"\n{'='*60}")
    print("FINAL SUMMARY TABLE")
    print(f"{'='*60}")
    print(f"{'Dataset':<15} {'Mean XCS':>10} {'Mean Conf':>10} {'Mean Jaccard':>13} {'Flagged':>8}")
    print("-" * 60)
    for ds, df in all_results.items():
        flagged = (df['flag_review'] == 'True').sum()
        print(
            f"{ds:<15} {df['xcs'].mean():>10.4f} {df['confidence'].mean():>10.4f} "
            f"{df['jaccard_sl'].mean():>13.4f} {flagged:>8}"
        )
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
