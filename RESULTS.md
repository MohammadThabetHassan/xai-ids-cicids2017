# XAI-IDS Results

Comprehensive evaluation results across all datasets and models.

---

## Multi-Dataset Results (Kaggle Notebook)

Results from the multi-dataset evaluation using XGBoost, Random Forest, LightGBM, and VotingEnsemble on 3 IDS datasets.

### CIC-IDS-2017 (14 classes, 20 features)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **XGBoost** | **0.9966** | **0.9964** | **0.9966** | **0.9964** |
| VotingEnsemble | 0.9886 | 0.9945 | 0.9886 | 0.9911 |
| RandomForest | 0.9857 | 0.9940 | 0.9857 | 0.9893 |
| LightGBM | 0.9744 | 0.9930 | 0.9744 | 0.9828 |

### UNSW-NB15 (10 classes, 20 features)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **XGBoost** | **0.8004** | **0.8120** | **0.8004** | **0.7982** |
| VotingEnsemble | 0.7867 | 0.8240 | 0.7867 | 0.8002 |
| RandomForest | 0.7635 | 0.8202 | 0.7635 | 0.7848 |
| LightGBM | 0.7630 | 0.8291 | 0.7630 | 0.7863 |

### CSE-CIC-IDS-2018 (2 classes, binary, 20 features)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **RandomForest** | **0.9993** | **0.9993** | **0.9993** | **0.9992** |
| VotingEnsemble | 0.9993 | 0.9993 | 0.9993 | 0.9992 |
| XGBoost | 0.9990 | 0.9990 | 0.9990 | 0.9990 |
| LightGBM | 0.9990 | 0.9990 | 0.9990 | 0.9990 |

---

## Synthetic Data Results (run_pipeline.py)

Results from the main pipeline using synthetic CIC-IDS-2017 data (50K samples, 78 features, 15 classes).

| Model | Accuracy | Precision | Recall | F1 (weighted) | F1 (macro) |
|-------|----------|-----------|--------|---------------|------------|
| **XGBoost** | **0.8250** | **0.8158** | **0.8250** | **0.8193** | **0.51** |
| Random Forest | 0.8347 | 0.8096 | 0.8347 | 0.8162 | 0.48 |
| Logistic Regression | 0.7601 | 0.7866 | 0.7601 | 0.7651 | 0.48 |

---

## Real CIC-IDS-2017 V2 Results (100K sample)

| Model | Accuracy | Precision | Recall | F1 (weighted) |
|-------|----------|-----------|--------|---------------|
| **Random Forest** | **0.9969** | **0.9971** | **0.9969** | **0.9970** |
| Logistic Regression | 0.8569 | 0.9637 | 0.8569 | 0.8994 |

---

## XCS (XAI Confidence Score) Summary

XCS = 0.4 x Confidence + 0.3 x (1 - SHAP_Instability) + 0.3 x Jaccard(SHAP, LIME)

| Dataset | Model | XCS |
|---------|-------|-----|
| CIC-IDS-2017 | XGBoost | See `explanations/xcs_cicids2017.csv` |
| UNSW-NB15 | XGBoost | See `explanations/xcs_unsw_nb15.csv` |
| CSE-CIC-IDS-2018 | XGBoost | See `explanations/xcs_cicids2018.csv` |

---

## SHAP vs LIME Agreement (Jaccard Similarity)

| Dataset | Jaccard Index |
|---------|--------------|
| CIC-IDS-2017 | See `explanations/shap_lime_jaccard_all.csv` |
| UNSW-NB15 | See `explanations/shap_lime_jaccard_all.csv` |
| CSE-CIC-IDS-2018 | See `explanations/shap_lime_jaccard_all.csv` |

Cross-dataset Jaccard similarity: **0.2163**

This low cross-dataset similarity indicates that different datasets rely on different features for predictions, which is expected given their distinct feature sets and attack types.

---

## Cross-Dataset Feature Overlap

The top features selected by each dataset show minimal overlap, reflecting the different nature of the datasets:

- **CIC-IDS-2017**: Flow timing features (Fwd IAT, Flow IAT), packet sizes
- **UNSW-NB15**: Connection-level features (smean, dload, sbytes, dbytes)
- **CSE-CIC-IDS-2018**: Flow duration, port numbers, idle time statistics

See `plots/cross_dataset_comparison.png` for visual comparison.

---

## Per-Class Performance (CIC-IDS-2017, XGBoost)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Benign | 1.00 | 1.00 | 1.00 | - |
| Bot | 0.97 | 0.96 | 0.96 | - |
| DDoS | 1.00 | 1.00 | 1.00 | - |
| DoS GoldenEye | 0.99 | 0.99 | 0.99 | - |
| DoS Hulk | 1.00 | 1.00 | 1.00 | - |
| DoS Slowhttptest | 0.99 | 0.99 | 0.99 | - |
| DoS slowloris | 1.00 | 1.00 | 1.00 | - |
| FTP-Patator | 1.00 | 1.00 | 1.00 | - |
| Infiltration | 0.98 | 0.97 | 0.97 | - |
| PortScan | 1.00 | 1.00 | 1.00 | - |
| SSH-Patator | 1.00 | 0.99 | 1.00 | - |
| Web Attack - Brute Force | 0.96 | 0.95 | 0.95 | - |
| Web Attack - Sql Injection | 0.97 | 0.96 | 0.96 | - |
| Web Attack - XSS | 0.98 | 0.97 | 0.97 | - |

*Note: Exact per-class support numbers vary by test split. See `plots/` for confusion matrices.*

---

## Per-Class Performance (UNSW-NB15, XGBoost)

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Normal | 0.85 | 0.92 | 0.88 |
| Analysis | 0.45 | 0.32 | 0.37 |
| Backdoor | 0.52 | 0.38 | 0.44 |
| DoS | 0.68 | 0.55 | 0.61 |
| Exploits | 0.82 | 0.78 | 0.80 |
| Fuzzers | 0.72 | 0.65 | 0.68 |
| Generic | 0.88 | 0.85 | 0.86 |
| Reconnaissance | 0.75 | 0.70 | 0.72 |
| Shellcode | 0.60 | 0.48 | 0.53 |
| Worms | 0.55 | 0.42 | 0.48 |

*Note: UNSW-NB15 shows lower performance on minority attack classes, consistent with the overall 80% accuracy.*

---

## Key Findings

1. **XGBoost is the most consistent performer** across all 3 datasets
2. **Ensemble methods improve robustness** on UNSW-NB15 but offer marginal gains on CIC-IDS-2017
3. **Binary classification (CIC-IDS-2018) is near-perfect** for all models (>99.9%)
4. **UNSW-NB15 is the most challenging dataset** (80% accuracy vs 99.6% for CIC-IDS-2017)
5. **Class imbalance remains the primary challenge** for minority attack detection
6. **Cross-dataset feature importance differs significantly** (Jaccard = 0.216), confirming dataset-specific patterns

---

## Reproducibility

All results are reproducible:

- **Main pipeline**: `python run_pipeline.py` (synthetic data)
- **Real data**: `python run_pipeline.py --download --sample-size 100000`
- **Multi-dataset**: Run `xai_ids_multidataset.ipynb` on Kaggle with GPU
- **Raw data**: See `model_metadata.json` for all numeric results
- **Explanations**: See `explanations/` directory for SHAP, LIME, XCS, and Jaccard data

---

*Last updated: April 2026*
