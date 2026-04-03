# Benchmark Execution Plan

**Purpose:** Document how to run benchmarks on CIC-IDS-2017 and other IDS datasets

---

## Current Status

- **Synthetic benchmark**: ✅ Completed (50K samples, class-weighted)
- **Real CIC-IDS-2017 benchmark**: ✅ Completed (100K sample via Zenodo)
- **Multi-dataset Kaggle benchmark**: ✅ Completed (CICIDS2017, UNSW-NB15, CICIDS2018)

---

## Completed Results

### Synthetic Data (50K samples)

| Model | Accuracy | Precision | Recall | F1 (weighted) | F1 (macro) |
|-------|----------|-----------|--------|---------------|------------|
| Logistic Regression | 0.7601 | 0.7866 | 0.7601 | 0.7651 | 0.48 |
| Random Forest | 0.8347 | 0.8096 | 0.8347 | 0.8162 | 0.48 |
| XGBoost | 0.8250 | 0.8158 | 0.8250 | 0.8193 | 0.51 |

### Real CIC-IDS-2017 V2 (100K sample)

| Model | Accuracy | Precision | Recall | F1 (weighted) |
|-------|----------|-----------|--------|---------------|
| Logistic Regression | 0.8569 | 0.9637 | 0.8569 | 0.8994 |
| Random Forest | 0.9969 | 0.9971 | 0.9969 | 0.9970 |

### Kaggle Multi-Dataset (XGBoost, RF, LightGBM, VotingEnsemble)

See `RESULTS.md` and `model_metadata.json` for full results across all 3 datasets.

---

## How to Re-run Benchmarks

### Option 1: Quick Real Data Benchmark (100K sample)

```bash
# Download real data (~369MB)
python run_pipeline.py --download --sample-size 100000

# Run with CV and PR curves
python run_pipeline.py --download --sample-size 100000 --cv-folds 3 --pr-curves --skip-explain

# Runtime: ~15-20 minutes
```

### Option 2: Full Real Data Benchmark (~500K sample)

```bash
# Download and process full available data
python run_pipeline.py --download --sample-size 500000

# Full evaluation
python run_pipeline.py --download --sample-size 500000 --cv-folds 5 --pr-curves --calibration --failure-analysis

# Runtime: ~45-60 minutes
```

### Option 3: Multi-Dataset (Kaggle Notebook)

The multi-dataset benchmark (CICIDS2017, UNSW-NB15, CICIDS2018) with LightGBM, VotingEnsemble, XCS, and SMOTE is available in `xai_ids_multidataset.ipynb`. This requires a Kaggle environment with GPU (Tesla T4).

---

## Expected Outputs

All outputs saved to `outputs/`:

| Output | Path |
|--------|------|
| Metrics CSV | `outputs/results_metrics.csv` |
| Classification Reports | `outputs/reports/classification_report_*.txt` |
| Confusion Matrices | `outputs/figures/confusion_matrix_*.png` |
| Model Comparison | `outputs/figures/model_comparison.png` |
| PR Curves (if enabled) | `outputs/figures/pr_curves_*.png` |
| Calibration (if enabled) | `outputs/figures/calibration_*.png` |
| Failure Analysis | `outputs/reports/failure_analysis.txt` |

Multi-dataset outputs are in `plots/`, `explanations/`, and `model_metadata.json`.

---

## Hardware Requirements

| Data Size | RAM | CPU | Time |
|-----------|-----|-----|------|
| 50K | 8GB | 4 cores | ~10 min |
| 100K | 16GB | 8 cores | ~20 min |
| 500K | 32GB+ | 16 cores | ~60 min |
| Full (2.8M) | 64GB+ | 32 cores | ~4+ hours |
| Kaggle (GPU) | Tesla T4 | GPU | ~30 min per dataset |

---

## If Zenodo Download Fails

1. Try manual download:
   ```bash
   curl -L -o data/raw/CIC-IDS-2017-V2.zip https://zenodo.org/records/10141593/files/CIC-IDS-2017-V2.zip
   unzip data/raw/CIC-IDS-2017-V2.zip -d data/raw/
   ```

2. Or try alternative sources:
   - Kaggle: https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset
   - IoTDataset.com: https://iotdataset.com/data/cicids2017-network-intrusion-detection-dataset

---

*Last updated: April 2026*
