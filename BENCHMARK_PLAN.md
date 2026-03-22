# Benchmark Execution Plan

**Purpose:** Document how to run the real CIC-IDS-2017 benchmark

---

## Current Status

- **Synthetic benchmark**: ✅ Completed (50K samples, class-weighted)
- **Real data benchmark**: ⏳ NOT YET RUN

---

## What Exists

### Code Support
- `python run_pipeline.py --download` - Downloads data from Zenodo fallback
- Automatic fallback from UNB server to Zenodo
- All evaluation features work with any dataset

### Data Sources
1. **Zenodo** (primary fallback): https://zenodo.org/records/10141593
   - CIC-IDS-2017 V2
   - 369MB ZIP file
   - Normalized data + new "Comb" class

2. **UNB Original** (may be offline): https://www.unb.ca/cic/datasets/ids-2017.html

---

## Execution Commands

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

### Option 3: Full Benchmark with Explainability

```bash
# Complete run (may take 2+ hours on large data)
python run_pipeline.py --download --sample-size 100000

# Runtime: ~30-45 minutes with SHAP
```

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

---

## Hardware Requirements

| Data Size | RAM | CPU | Time |
|-----------|-----|-----|------|
| 50K | 8GB | 4 cores | ~10 min |
| 100K | 16GB | 8 cores | ~20 min |
| 500K | 32GB+ | 16 cores | ~60 min |
| Full (2.8M) | 64GB+ | 32 cores | ~4+ hours |

---

## What To Do After Benchmark

1. **Update README.md**: Replace synthetic results with real benchmark
2. **Update docs/site/index.html**: Update model comparison table
3. **Update PROJECT_STATUS.md**: Mark "Real data benchmark" as completed
4. **Commit and push**: Document the change in CHANGELOG.md

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

## Current Benchmark Results (Synthetic Only)

| Model | Accuracy | Precision | Recall | F1 (weighted) | F1 (macro) |
|-------|----------|-----------|--------|---------------|------------|
| Logistic Regression | 0.7601 | 0.7866 | 0.7601 | 0.7651 | 0.48 |
| Random Forest | 0.8347 | 0.8096 | 0.8347 | 0.8162 | 0.48 |
| XGBoost | 0.8250 | 0.8158 | 0.8250 | 0.8193 | 0.51 |

**Note:** These are from synthetic data. Real data results will differ.

---

*Last updated: March 2026*
