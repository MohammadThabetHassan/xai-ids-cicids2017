# XAI-IDS Benchmarks

Comprehensive benchmark results for all models, datasets, and configurations.

---

## Synthetic Data Benchmarks (run_pipeline.py)

| Dataset | Model | Train Size | Test Size | Features | Accuracy | Precision (W) | Recall (W) | F1 (Weighted) | F1 (Macro) | Train Time (s) | Inference (ms/sample) |
|---------|-------|-----------|----------|----------|----------|--------------|-----------|-------------|-----------|---------------|---------------------|
| CIC-IDS-2017 (synthetic) | XGBoost | 35,000 | 10,000 | 78 | 0.8250 | 0.8158 | 0.8250 | 0.8193 | 0.51 | ~2.5 | ~0.3 |
| CIC-IDS-2017 (synthetic) | Random Forest | 35,000 | 10,000 | 78 | 0.8347 | 0.8096 | 0.8347 | 0.8162 | 0.48 | ~3.0 | ~0.5 |
| CIC-IDS-2017 (synthetic) | Logistic Regression | 35,000 | 10,000 | 78 | 0.7601 | 0.7866 | 0.7601 | 0.7651 | 0.48 | ~1.0 | ~0.1 |

## Multi-Dataset Benchmarks (Kaggle Notebook)

### CIC-IDS-2017 (14 classes, 20 selected features)

| Model | Train Size | Test Size | Accuracy | Precision (W) | Recall (W) | F1 (Weighted) | Train Time (s) |
|-------|-----------|----------|----------|--------------|-----------|-------------|---------------|
| XGBoost | ~2.8M | ~560K | 0.9966 | 0.9964 | 0.9966 | 0.9964 | ~120 |
| VotingEnsemble | ~2.8M | ~560K | 0.9886 | 0.9945 | 0.9886 | 0.9911 | ~180 |
| Random Forest | ~2.8M | ~560K | 0.9857 | 0.9940 | 0.9857 | 0.9893 | ~200 |
| LightGBM | ~2.8M | ~560K | 0.9744 | 0.9930 | 0.9744 | 0.9828 | ~60 |

### UNSW-NB15 (10 classes, 20 selected features)

| Model | Train Size | Test Size | Accuracy | Precision (W) | Recall (W) | F1 (Weighted) | Train Time (s) |
|-------|-----------|----------|----------|--------------|-----------|-------------|---------------|
| XGBoost | ~2.5M | ~500K | 0.8004 | 0.7906 | 0.8004 | 0.7933 | ~90 |
| VotingEnsemble | ~2.5M | ~500K | 0.7867 | 0.8240 | 0.7867 | 0.8026 | ~150 |
| Random Forest | ~2.5M | ~500K | 0.7635 | 0.8202 | 0.7635 | 0.7848 | ~180 |
| LightGBM | ~2.5M | ~500K | 0.7630 | 0.8291 | 0.7630 | 0.7863 | ~50 |

### CSE-CIC-IDS-2018 (2 classes captured, 20 selected features)

| Model | Train Size | Test Size | Accuracy | Precision (W) | Recall (W) | F1 (Weighted) | Train Time (s) |
|-------|-----------|----------|----------|--------------|-----------|-------------|---------------|
| XGBoost | ~10M | ~2M | 0.9990 | 0.9990 | 0.9990 | 0.9990 | ~150 |
| Random Forest | ~10M | ~2M | 0.9993 | 0.9993 | 0.9993 | 0.9992 | ~250 |
| LightGBM | ~10M | ~2M | 0.9990 | 0.9990 | 0.9990 | 0.9990 | ~70 |

## XCS Benchmarks (v2 — full LIME formula, synthetic test sets, n=100)

| Dataset | Model | Mean XCS | Mean Confidence | Mean SHAP Instability | Mean Jaccard | Flagged (< 0.3) | Computation Time (s/sample) |
|---------|-------|----------|-----------------|----------------------|--------------|-----------------|---------------------------|
| CIC-IDS-2017 | XGBoost | 0.420 | 0.478 | 0.416 | 0.178 | 0 / 100 (0%) | ~0.3 |
| UNSW-NB15 | XGBoost | 0.386 | 0.310 | 0.253 | 0.126 | 5 / 100 (5%) | ~0.4 |
| CSE-CIC-IDS-2018 | XGBoost | 0.675 | 0.999 | 0.226 | 0.143 | 0 / 100 (0%) | ~0.15 |

## API Latency Benchmarks

| Endpoint | Mean Latency (ms) | P95 Latency (ms) | Notes |
|----------|-------------------|------------------|-------|
| GET / | < 1 | < 5 | Health check |
| POST /predict | ~10 | ~20 | Fast-path XCS (confidence only) |
| POST /explain | ~300 | ~500 | Full XCS with SHAP + LIME |
| GET /xcs-summary | < 5 | < 10 | Reads pre-computed CSVs |
| GET /health/features | < 5 | < 10 | Returns scaler statistics |

## Statistical Significance

All pairwise model comparisons use McNemar's test (α = 0.05).
Results show XGBoost is significantly better than LightGBM on
CIC-IDS-2017 (p < 0.001) but not significantly different from
Random Forest on CIC-IDS-2018 (p > 0.05) due to ceiling effects.

---

*All benchmarks run on Kaggle GPU (Tesla T4) for multi-dataset results.
Synthetic benchmarks run on standard CPU. Inference latency measured
on a single core with no batching.*
