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
import os
import sys
import time
from pathlib import Path

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
