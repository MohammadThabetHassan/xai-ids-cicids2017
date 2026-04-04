"""
Smoke tests for XAI-IDS pipeline.

Verifies that core modules import correctly and that the pipeline
can execute on a small synthetic dataset without errors.
"""

import os
import sys

import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
class TestImports:
    """Verify all modules can be imported successfully."""

    def test_import_logger(self):
        from src.utils.logger import get_logger

        logger = get_logger("test")
        assert logger is not None

    def test_import_loader(self):
        from src.data.loader import load_dataset

        assert callable(load_dataset)

    def test_import_preprocessing(self):
        from src.data.preprocessing import clean_data

        assert callable(clean_data)

    def test_import_generate_sample(self):
        from src.data.generate_sample import generate_sample_dataset

        assert callable(generate_sample_dataset)

    def test_import_models(self):
        from src.models.train import MODEL_CONFIGS

        assert "logistic_regression" in MODEL_CONFIGS
        assert "random_forest" in MODEL_CONFIGS
        assert "xgboost" in MODEL_CONFIGS

    def test_import_evaluation(self):
        from src.evaluation.metrics import compute_metrics

        assert callable(compute_metrics)

    def test_import_explainability(self):
        try:
            from src.explainability.explain import compute_shap_explanations
            assert callable(compute_shap_explanations)
        except ImportError:
            # shap/lime may not be installed in CI environment
            pass

    def test_import_download(self):
        from src.data.download import discover_csv_links

        assert callable(discover_csv_links)
class TestDataGeneration:
    """Test synthetic data generation."""

    def test_generate_sample(self, tmp_path):
        from src.data.generate_sample import generate_sample_dataset

        output_path = generate_sample_dataset(
            n_samples=500, output_dir=str(tmp_path), seed=42
        )
        assert os.path.exists(output_path)
        import pandas as pd

        df = pd.read_csv(output_path)
        assert len(df) > 400  # Some classes get minimum 10 samples
        assert "Label" in df.columns

    def test_dataset_has_all_classes(self, tmp_path):
        from src.data.generate_sample import ATTACK_CLASSES, generate_sample_dataset

        output_path = generate_sample_dataset(
            n_samples=5000, output_dir=str(tmp_path), seed=42
        )
        import pandas as pd

        df = pd.read_csv(output_path)
        for cls in ATTACK_CLASSES:
            assert cls in df["Label"].values, f"Missing class: {cls}"
class TestPreprocessing:
    """Test data preprocessing pipeline."""

    def test_clean_data(self):
        import pandas as pd

        from src.data.preprocessing import clean_data

        # Create dirty data
        df = pd.DataFrame(
            {
                "A": [1.0, 2.0, np.inf, 4.0, 5.0, 5.0],
                "B": [1.0, np.nan, 3.0, 4.0, 5.0, 5.0],
                "Label": ["a", "b", "c", "d", "e", "e"],
            }
        )
        cleaned = clean_data(df)
        assert len(cleaned) <= len(df)
        assert (
            not cleaned.select_dtypes(include=[np.number])
            .isin([np.inf, -np.inf])
            .any()
            .any()
        )

    def test_encode_labels(self):
        import pandas as pd

        from src.data.preprocessing import encode_labels

        df = pd.DataFrame(
            {
                "A": [1, 2, 3],
                "Label": ["BENIGN", "DoS Hulk", "BENIGN"],
            }
        )
        df_enc, le, mapping = encode_labels(df, save_path=None)
        assert df_enc["Label"].dtype in [np.int32, np.int64, np.intp]
        assert "BENIGN" in mapping
class TestPipelineMini:
    """Run a minimal pipeline end-to-end."""

    def test_mini_pipeline(self, tmp_path):
        from src.data.generate_sample import generate_sample_dataset
        from src.data.preprocessing import (
            clean_data,
            encode_labels,
            scale_features,
            split_data,
        )
        from src.evaluation.metrics import compute_metrics
        from src.models.train import train_model

        # Generate small dataset
        data_dir = str(tmp_path / "raw")
        os.makedirs(data_dir)
        generate_sample_dataset(n_samples=1000, output_dir=data_dir, seed=42)

        # Load and preprocess
        import pandas as pd

        csv_path = os.path.join(data_dir, "sample_cicids2017.csv")
        df = pd.read_csv(csv_path)
        df = clean_data(df)
        df, le, mapping = encode_labels(df, save_path=None)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

        X_train_s, X_val_s, X_test_s, scaler = scale_features(
            X_train, X_val, X_test, save_path=None
        )

        # Train a fast model
        model = train_model(
            "logistic_regression",
            X_train_s,
            y_train.values,
            save_dir=str(tmp_path / "models"),
        )

        # Evaluate
        y_pred = model.predict(X_test_s)
        metrics = compute_metrics(y_test.values, y_pred, "logistic_regression")

        assert metrics["accuracy"] > 0.0
        assert metrics["f1_score"] > 0.0
