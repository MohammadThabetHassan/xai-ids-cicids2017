#!/usr/bin/env python3
"""
XAI-IDS: Explainable AI Intrusion Detection System

End-to-end pipeline for training, evaluating, and explaining
machine learning models on the CIC-IDS-2017 dataset.

Usage:
    python run_pipeline.py                    # Full pipeline with sample data
    python run_pipeline.py --download         # Download real CIC-IDS-2017 dataset
    python run_pipeline.py --skip-explain     # Skip explainability (faster)
    python run_pipeline.py --sample-size 5000 # Use smaller sample for testing

Authors:
    Mohammad Thabet Hassan
    Fahad Sadek
    Ahmed Sami

Supervisor:
    Dr. Mehak Khurana
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.logger import setup_logger

logger = setup_logger("xai_ids.pipeline")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="XAI-IDS: Explainable AI Intrusion Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                       Run full pipeline with synthetic data
  python run_pipeline.py --download            Download and use real CIC-IDS-2017 data
  python run_pipeline.py --sample-size 10000   Generate larger synthetic dataset
  python run_pipeline.py --skip-explain        Skip SHAP/LIME (faster execution)
  python run_pipeline.py --models rf xgb       Train only Random Forest and XGBoost
        """,
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the real CIC-IDS-2017 dataset (requires internet)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory containing raw CSV files (default: data/raw)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50000,
        help="Number of samples for synthetic data generation (default: 50000)",
    )
    parser.add_argument(
        "--random-sample",
        action="store_true",
        help="Use random sampling instead of first N rows when loading real data",
    )
    parser.add_argument(
        "--skip-explain",
        action="store_true",
        help="Skip explainability analysis (SHAP and LIME)",
    )
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="Skip SHAP explanations only",
    )
    parser.add_argument(
        "--skip-lime",
        action="store_true",
        help="Skip LIME explanations only",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["lr", "rf", "xgb", "all"],
        default=["all"],
        help="Models to train: lr (Logistic Regression), rf (Random Forest), xgb (XGBoost)",
    )
    parser.add_argument(
        "--shap-samples",
        type=int,
        default=500,
        help="Number of samples for SHAP analysis (default: 500)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Base output directory (default: outputs)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        help="Number of CV folds (0 to skip, default: 0)",
    )
    parser.add_argument(
        "--pr-curves",
        action="store_true",
        help="Generate precision-recall curves",
    )
    parser.add_argument(
        "--calibration",
        action="store_true",
        help="Generate calibration curves",
    )
    parser.add_argument(
        "--failure-analysis",
        action="store_true",
        help="Generate failure analysis report",
    )
    parser.add_argument(
        "--smote",
        action="store_true",
        help="Apply SMOTE oversampling for minority classes",
    )
    parser.add_argument(
        "--smote-min-samples",
        type=int,
        default=200,
        help="Minimum samples per class for SMOTE (default: 200)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Run statistical significance tests (McNemar, paired t-test)",
    )
    parser.add_argument(
        "--adversarial",
        action="store_true",
        help="Evaluate adversarial robustness (FGSM attacks via ART)",
    )
    parser.add_argument(
        "--drift",
        action="store_true",
        help="Run temporal drift detection analysis",
    )
    parser.add_argument(
        "--counterfactuals",
        action="store_true",
        help="Generate counterfactual explanations (DiCE)",
    )
    parser.add_argument(
        "--learned-xcs",
        action="store_true",
        help="Calibrate XCS weights via logistic regression on correctness",
    )

    return parser.parse_args()


def main():
    """Execute the full XAI-IDS pipeline."""
    args = parse_args()
    start_time = time.time()

    logger.info("=" * 70)
    logger.info("  XAI-IDS: Explainable AI Intrusion Detection System")
    logger.info("  CIC-IDS-2017 Dataset")
    logger.info("=" * 70)

    # Set up output directories
    figures_dir = os.path.join(args.output_dir, "figures")
    models_dir = os.path.join(args.output_dir, "models")
    reports_dir = os.path.join(args.output_dir, "reports")
    logs_dir = os.path.join(args.output_dir, "logs")
    metrics_path = os.path.join(args.output_dir, "results_metrics.csv")

    for d in [figures_dir, models_dir, reports_dir, logs_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # ─── Step 1: Data Acquisition ───
    logger.info("\n" + "=" * 50)
    logger.info("STEP 1: Data Acquisition")
    logger.info("=" * 50)

    if args.download:
        from src.data.download import download_dataset

        logger.info("Downloading CIC-IDS-2017 dataset...")
        downloaded = download_dataset(dest_dir=args.data_dir)
        if not downloaded:
            logger.error("Dataset download failed. Falling back to synthetic data.")
            _generate_synthetic(args)
    else:
        # Check if real data exists
        csv_files = (
            [f for f in os.listdir(args.data_dir) if f.endswith(".csv")]
            if os.path.exists(args.data_dir)
            else []
        )

        if csv_files:
            logger.info(f"Found {len(csv_files)} CSV files in {args.data_dir}")
        else:
            _generate_synthetic(args)

    # ─── Step 2: Preprocessing ───
    logger.info("\n" + "=" * 50)
    logger.info("STEP 2: Preprocessing")
    logger.info("=" * 50)

    from src.data.preprocessing import run_preprocessing

    data = run_preprocessing(
        data_dir=args.data_dir,
        models_dir=models_dir,
        nrows_per_file=args.sample_size if args.sample_size < 500000 else None,
        random_sample=args.random_sample,
    )

    # ─── Step 3: Model Training ───
    logger.info("\n" + "=" * 50)
    logger.info("STEP 3: Model Training")
    logger.info("=" * 50)

    from src.models.train import train_model, MODEL_CONFIGS

    model_map = {"lr": "logistic_regression", "rf": "random_forest", "xgb": "xgboost"}
    if "all" in args.models:
        models_to_train = list(MODEL_CONFIGS.keys())
    else:
        models_to_train = [model_map[m] for m in args.models]

    trained_models = {}
    for model_name in models_to_train:
        model = train_model(
            model_name,
            data["X_train"],
            data["y_train"],
            data["X_val"],
            data["y_val"],
            save_dir=models_dir,
            use_smote=args.smote,
            smote_min_samples=args.smote_min_samples,
        )
        trained_models[model_name] = model

    # ─── Step 4: Evaluation ───
    logger.info("\n" + "=" * 50)
    logger.info("STEP 4: Evaluation")
    logger.info("=" * 50)

    from src.evaluation.metrics import evaluate_all_models

    label_names = list(data["label_encoder"].classes_)
    all_metrics = evaluate_all_models(
        trained_models,
        data["X_test"],
        data["y_test"],
        label_names=label_names,
        figures_dir=figures_dir,
        reports_dir=reports_dir,
        metrics_path=metrics_path,
    )

    if args.cv_folds > 0:
        logger.info("\n" + "=" * 50)
        logger.info("Cross-Validation Analysis")
        logger.info("=" * 50)
        from src.evaluation.metrics import run_cross_validation

        for model_name, model in trained_models.items():
            cv_results = run_cross_validation(
                model,
                data["X_train"],
                data["y_train"],
                cv=args.cv_folds,
                scoring="f1_weighted",
            )

    if args.pr_curves:
        logger.info("\n" + "=" * 50)
        logger.info("Precision-Recall Curves")
        logger.info("=" * 50)
        from src.evaluation.metrics import plot_precision_recall_curves

        plot_precision_recall_curves(
            trained_models,
            data["X_test"],
            data["y_test"],
            label_names,
            save_dir=figures_dir,
        )

    if args.calibration:
        logger.info("\n" + "=" * 50)
        logger.info("Calibration Curves")
        logger.info("=" * 50)
        from src.evaluation.metrics import plot_calibration_curves

        plot_calibration_curves(
            trained_models,
            data["X_test"],
            data["y_test"],
            label_names,
            save_dir=figures_dir,
        )

    if args.failure_analysis:
        logger.info("\n" + "=" * 50)
        logger.info("Failure Analysis")
        logger.info("=" * 50)
        from src.evaluation.metrics import generate_failure_analysis

        generate_failure_analysis(
            trained_models,
            data["X_test"],
            data["y_test"],
            label_names,
            save_dir=reports_dir,
        )

    # ─── Step 4b: Statistical Significance Testing ───
    if args.stats and len(trained_models) >= 2:
        logger.info("\n" + "=" * 50)
        logger.info("Statistical Significance Testing")
        logger.info("=" * 50)

        from src.evaluation.stats import compare_models_statistically

        predictions = {}
        for name, model in trained_models.items():
            predictions[name] = model.predict(data["X_test"])

        stats_results = compare_models_statistically(
            data["y_test"], predictions, alpha=0.05
        )

        logger.info("\nPairwise Model Comparisons:")
        logger.info("-" * 70)
        for name_a, comparisons in stats_results.items():
            for name_b, result in comparisons.items():
                sig_marker = "*" if result["mcnemar_sig"] else " "
                logger.info(
                    f"  {name_a} vs {name_b}: acc={result['acc_a']:.4f} vs "
                    f"{result['acc_b']:.4f}, McNemar p={result['mcnemar_p']:.6f}{sig_marker}"
                )

    # ─── Step 4c: Adversarial Robustness ───
    if args.adversarial:
        logger.info("\n" + "=" * 50)
        logger.info("Adversarial Robustness Evaluation")
        logger.info("=" * 50)

        from src.evaluation.adversarial import (
            compute_xcs_on_adversarial,
            evaluate_adversarial_robustness,
            plot_adversarial_results,
        )

        primary_model = trained_models.get("xgboost") or list(trained_models.values())[0]
        adv_results = evaluate_adversarial_robustness(
            primary_model, data["X_test"], data["y_test"]
        )
        if "error" not in adv_results:
            plot_adversarial_results(
                adv_results,
                save_path=os.path.join(figures_dir, "adversarial_robustness.png"),
            )
            logger.info(f"\nAdversarial Results:")
            logger.info(f"  Baseline accuracy: {adv_results['baseline_accuracy']:.4f}")
            for eps_result in adv_results["epsilons"]:
                logger.info(
                    f"  FGSM eps={eps_result['epsilon']}: "
                    f"acc={eps_result['adversarial_accuracy']:.4f}, "
                    f"drop={eps_result['accuracy_drop']:.4f}"
                )

    # ─── Step 4d: Temporal Drift Detection ───
    if args.drift:
        logger.info("\n" + "=" * 50)
        logger.info("Temporal Drift Detection")
        logger.info("=" * 50)

        from src.evaluation.drift import plot_temporal_drift, simulate_temporal_drift

        primary_model = trained_models.get("xgboost") or list(trained_models.values())[0]
        from sklearn.base import clone
        try:
            model_copy = clone(primary_model)
        except Exception:
            model_copy = primary_model

        drift_results = simulate_temporal_drift(
            model_copy, data["X_train"], data["y_train"], n_splits=5
        )
        plot_temporal_drift(
            drift_results,
            save_path=os.path.join(figures_dir, "temporal_drift_CICIDS2017.png"),
        )
        logger.info(f"\nTemporal Drift Results:")
        for split in drift_results["splits"]:
            logger.info(
                f"  Split: train_acc={split['train_accuracy']:.4f}, "
                f"test_acc={split['test_accuracy']:.4f}, "
                f"drift_rate={split['feature_drift_rate']:.1%}"
            )

    # ─── Step 5: Explainability ───
    if not args.skip_explain:
        logger.info("\n" + "=" * 50)
        logger.info("STEP 5: Explainability (SHAP & LIME)")
        logger.info("=" * 50)

        from src.explainability.explain import run_explainability

        run_explainability(
            trained_models,
            data["X_train"],
            data["X_test"],
            data["y_test"],
            data["feature_names"],
            label_names,
            figures_dir=figures_dir,
            reports_dir=reports_dir,
            shap_sample_size=args.shap_samples,
        )
    else:
        logger.info("\nSkipping explainability analysis (--skip-explain)")

    # ─── Step 5b: Counterfactual Explanations ───
    if args.counterfactuals:
        logger.info("\n" + "=" * 50)
        logger.info("Counterfactual Explanations")
        logger.info("=" * 50)

        from src.explainability.counterfactual import generate_counterfactuals_for_classes

        primary_model = trained_models.get("xgboost") or list(trained_models.values())[0]
        cf_results = generate_counterfactuals_for_classes(
            primary_model,
            data["X_test"],
            data["y_test"],
            data["feature_names"],
            label_names,
            n_per_class=3,
        )

        logger.info(f"\nCounterfactual Results:")
        for cls_name, cf_data in cf_results.items():
            n_cf = len(cf_data.get("counterfactuals", []))
            logger.info(f"  {cls_name}: {n_cf} counterfactuals generated ({cf_data['method']})")

    # ─── Step 5c: Learned XCS Weights Calibration ───
    if args.learned_xcs:
        logger.info("\n" + "=" * 50)
        logger.info("Learned XCS Weights Calibration")
        logger.info("=" * 50)

        from src.explainability.explain import compute_shap_explanations
        from sklearn.linear_model import LogisticRegression
        import shap
        import lime.lime_tabular

        primary_model = trained_models.get("xgboost") or list(trained_models.values())[0]
        n_cal = min(200, len(data["X_test"]))
        X_cal = data["X_test"][:n_cal]
        y_cal = data["y_test"][:n_cal]
        y_pred_cal = primary_model.predict(X_cal)
        y_proba_cal = primary_model.predict_proba(X_cal)
        correct = (y_pred_cal == y_cal).astype(float)

        # Compute XCS components for calibration set
        confidences = np.max(y_proba_cal, axis=1)

        # SHAP instability (simplified: single run, use variance across features as proxy)
        explainer = shap.TreeExplainer(primary_model)
        sv = explainer.shap_values(X_cal)
        if isinstance(sv, list):
            sv = np.array([sv[p][i] for i, p in enumerate(y_pred_cal)])
        elif sv.ndim == 3:
            sv = np.array([sv[i, :, p] for i, p in enumerate(y_pred_cal)])
        else:
            sv = sv

        shap_instability = np.std(np.abs(sv), axis=0).mean()
        shap_instability_norm = np.minimum(shap_instability / max(np.mean(np.abs(sv)), 1e-8), 0.5)
        instab_array = np.full(n_cal, shap_instability_norm)

        # LIME-SHAP Jaccard (sample of 20 for speed)
        n_lime = min(20, n_cal)
        jaccard_array = np.zeros(n_cal)
        try:
            lime_exp = lime.lime_tabular.LimeTabularExplainer(
                X_cal[:n_lime],
                feature_names=data["feature_names"],
                mode="classification",
            )
            for i in range(n_lime):
                shap_top = set(np.argsort(np.abs(sv[i]))[-5:])
                shap_top_names = {data["feature_names"][j] for j in shap_top}
                exp = lime_exp.explain_instance(
                    X_cal[i], primary_model.predict_proba,
                    num_features=5, num_samples=200,
                )
                as_map = exp.as_map()
                name_to_w = {}
                for cls_id in as_map:
                    for fidx, w in as_map[cls_id]:
                        name_to_w[data["feature_names"][int(fidx)]] = \
                            name_to_w.get(data["feature_names"][int(fidx)], 0) + abs(w)
                lime_top = set(sorted(name_to_w, key=lambda k: name_to_w[k], reverse=True)[:5])
                if shap_top_names or lime_top:
                    jaccard_array[i] = len(shap_top_names & lime_top) / len(shap_top_names | lime_top)
        except Exception as e:
            logger.warning(f"LIME Jaccard computation failed: {e}")

        # Fit logistic regression
        X_features = np.column_stack([confidences, 1 - instab_array, jaccard_array])
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_features, correct)

        learned_weights = lr.coef_[0]
        learned_weights_norm = learned_weights / learned_weights.sum()

        logger.info(f"\nLearned XCS Weights vs Hand-Tuned:")
        logger.info(f"  {'Component':<20} {'Learned':>10} {'Hand-Tuned':>10}")
        logger.info(f"  {'Confidence':<20} {learned_weights_norm[0]:>10.4f} {0.4:>10.4f}")
        logger.info(f"  {'1-Instability':<20} {learned_weights_norm[1]:>10.4f} {0.3:>10.4f}")
        logger.info(f"  {'Jaccard':<20} {learned_weights_norm[2]:>10.4f} {0.3:>10.4f}")
        logger.info(f"  LR Accuracy: {lr.score(X_features, correct):.4f}")

        # Save learned weights
        import json
        weights_path = os.path.join(reports_dir, "learned_xcs_weights.json")
        with open(weights_path, "w") as f:
            json.dump({
                "learned_weights": learned_weights.tolist(),
                "normalized_weights": learned_weights_norm.tolist(),
                "hand_tuned": [0.4, 0.3, 0.3],
                "lr_accuracy": float(lr.score(X_features, correct)),
            }, f, indent=2)
        logger.info(f"  Saved learned weights to {weights_path}")

    # ─── Summary ───
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 70)
    logger.info("  PIPELINE COMPLETE")
    logger.info(f"  Total time: {elapsed:.1f} seconds ({elapsed / 60:.1f} minutes)")
    logger.info("=" * 70)

    logger.info("\nOutputs saved:")
    logger.info(f"  Metrics:    {metrics_path}")
    logger.info(f"  Figures:    {figures_dir}/")
    logger.info(f"  Models:     {models_dir}/")
    logger.info(f"  Reports:    {reports_dir}/")
    logger.info(f"  Logs:       {logs_dir}/")

    # Print metrics table
    if all_metrics:
        logger.info("\n" + "-" * 60)
        logger.info("MODEL COMPARISON SUMMARY")
        logger.info("-" * 60)
        header = f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}"
        logger.info(header)
        logger.info("-" * 65)
        for m in all_metrics:
            row = (
                f"{m['model']:<25} {m['accuracy']:>10.4f} "
                f"{m['precision']:>10.4f} {m['recall']:>10.4f} "
                f"{m['f1_score']:>10.4f}"
            )
            logger.info(row)

    return 0


def _generate_synthetic(args):
    """Generate synthetic dataset when real data is unavailable."""
    from src.data.generate_sample import generate_sample_dataset

    logger.info(
        f"Generating synthetic CIC-IDS-2017 dataset ({args.sample_size} samples)..."
    )
    generate_sample_dataset(
        n_samples=args.sample_size,
        output_dir=args.data_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
