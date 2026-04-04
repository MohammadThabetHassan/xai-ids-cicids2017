"""
Verify model artifact integrity.

Loads model_metadata.json, loads each committed joblib model,
generates dummy samples with the correct feature count,
runs predict() and predict_proba(), verifies shapes are correct,
and prints a pass/fail table.
"""

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np

DATASETS = ["CICIDS2017", "UNSWNB15", "CICIDS2018"]
N_DUMMY = 200

def main():
    print("=" * 60)
    print("Model Artifact Integrity Verification")
    print("=" * 60)

    metadata = json.load(open("model_metadata.json"))
    results = []

    for ds in DATASETS:
        model_path = Path(f"models/xgb_{ds}.joblib")
        scaler_path = Path(f"models/scaler_{ds}.joblib")
        le_path = Path(f"models/label_encoder_{ds}.joblib")
        feat_path = Path(f"models/features_{ds}.joblib")

        checks = {}

        # Check files exist
        for name, p in [("model", model_path), ("scaler", scaler_path),
                        ("label_encoder", le_path), ("features", feat_path)]:
            checks[f"{name}_exists"] = p.exists()

        if not all(checks.values()):
            results.append({"dataset": ds, "status": "SKIP", "detail": "Missing artifacts"})
            continue

        # Load artifacts
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        le = joblib.load(le_path)
        feature_names = joblib.load(feat_path)

        checks["model_has_predict"] = hasattr(model, "predict")
        checks["model_has_predict_proba"] = hasattr(model, "predict_proba")
        checks["scaler_has_transform"] = hasattr(scaler, "transform")

        n_features = len(feature_names)
        n_classes = len(le.classes_)
        checks["feature_count_ok"] = n_features == model.n_features_in_

        # Generate dummy data and predict
        rng = np.random.RandomState(42)
        X_raw = rng.randn(N_DUMMY, n_features).astype(np.float32)
        X_scaled = scaler.transform(X_raw)

        y_pred = model.predict(X_scaled)
        y_proba = model.predict_proba(X_scaled)

        checks["predict_shape_ok"] = y_pred.shape == (N_DUMMY,)
        checks["predict_proba_shape_ok"] = y_proba.shape == (N_DUMMY, n_classes)
        checks["proba_sums_to_1"] = np.allclose(y_proba.sum(axis=1), 1.0, atol=1e-5)

        # Check metadata consistency
        if ds in metadata.get("datasets", {}):
            ds_meta = metadata["datasets"][ds]
            if "results" in ds_meta and "XGBoost" in ds_meta["results"]:
                xgb_meta = ds_meta["results"]["XGBoost"]
                checks["metadata_f1_present"] = "f1" in xgb_meta
                checks["metadata_acc_present"] = "accuracy" in xgb_meta
            else:
                checks["metadata_f1_present"] = False
                checks["metadata_acc_present"] = False
        else:
            checks["metadata_f1_present"] = False
            checks["metadata_acc_present"] = False

        all_pass = all(checks.values())
        status = "PASS" if all_pass else "FAIL"
        failed = [k for k, v in checks.items() if not v]
        detail = ", ".join(failed) if failed else "All checks passed"
        results.append({"dataset": ds, "status": status, "detail": detail})

    # Print table
    print(f"\n{'Dataset':<15} {'Status':>8}  {'Detail'}")
    print("-" * 60)
    for r in results:
        print(f"{r['dataset']:<15} {r['status']:>8}  {r['detail']}")

    all_pass = all(r["status"] == "PASS" for r in results)
    print(f"\n{'='*60}")
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    print(f"{'='*60}")

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
