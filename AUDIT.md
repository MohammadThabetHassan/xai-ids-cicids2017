# XAI-IDS Project Audit Report

**Audit Date:** March 2026  
**Auditor:** AI Engineering Review  
**Version:** 1.0  

---

## 1. Executive Summary

The XAI-IDS project is a well-engineered demonstration of an ML pipeline with explainability capabilities. However, it currently functions as a **proof-of-concept** rather than a research-grade or production-grade system. The primary limitation is that all published results come from synthetic data, and significant class imbalance issues cause 5 out of 14 attack types to have 0% detection rates.

---

## 2. Repository Overview

| Aspect | Status |
|--------|--------|
| **Repository URL** | https://github.com/MohammadThabetHassan/Explainable-AI-Intrusion-Detection-System-XAI-IDS- |
| **Live Site** | https://mohammadthabethassan.github.io/Explainable-AI-Intrusion-Detection-System-XAI-IDS-/ |
| **Total Commits** | 41 |
| **Primary Author** | Mohammad Thabet Hassan |
| **Last Updated** | March 2026 |

---

## 3. Strengths

### 3.1 Engineering Quality
- **Clean Architecture**: Modular code organization with separate packages for data, models, evaluation, and explainability
- **Complete Pipeline**: End-to-end workflow from raw data to trained models to explainable outputs
- **Working CI/CD**: Functional GitHub Actions for testing and Pages deployment
- **Good Documentation**: Comprehensive README with honest limitations disclosure

### 3.2 Technical Implementation
- **Proper Preprocessing**: Handles infinite values, NaN, duplicates; uses stratified splitting
- **No Data Leakage**: Scaler fitted only on training data
- **Complete Evaluation**: Confusion matrices, precision/recall/F1 for all models
- **Explainability**: SHAP global + LIME local explanations working

### 3.3 Outputs Generated
- All model artifacts (pickle files)
- 21 visualization files (confusion matrices, SHAP, LIME)
- Metrics CSV with comparison table
- Classification reports for each model

---

## 4. Weaknesses

### 4.1 Scientific Validity (Critical)

**Issue 1: Complete Class Failure**

The classification reports reveal catastrophic performance gaps on minority attack classes:

| Attack Class | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| DoS GoldenEye | 0.00 | 0.00 | 0.00 | 400 |
| DoS Slowhttptest | 0.00 | 0.00 | 0.00 | 150 |
| DoS slowloris | 0.00 | 0.00 | 0.00 | 200 |
| Heartbleed | 0.00 | 0.00 | 0.00 | 10 |
| Web Attack - Sql Injection | 0.00 | 0.00 | 0.00 | 20 |
| DDoS | 0.31 | 0.02 | 0.04 | 700 |

**Impact**: The model cannot detect 5 of 14 attack types at all. DDoS detection has only 2% recall.

**Issue 2: Macro vs Weighted Disparity**

- Weighted Average F1: **0.80**
- Macro Average F1: **0.44**

The high weighted average is misleading because it reflects BENIGN class (60% of data) performance. The macro average shows the true model capability across all classes.

### 4.2 Data Quality (Critical)

| Aspect | Status | Notes |
|--------|--------|-------|
| Real Dataset Download | **BROKEN** | HTTP URL to UNB server appears offline |
| Real Dataset Evaluation | **NOT AVAILABLE** | Only synthetic data has been run |
| Synthetic Data Quality | Good | Realistic feature distributions, 78 features match CIC-IDS-2017 schema |
| Data Provenance | Not Documented | No checksums, version info, or source verification |

### 4.3 Class Imbalance (High)

```
Class Distribution (50K synthetic):
- BENIGN:          30,000 samples (60%)
- DoS Hulk:         5,000 samples (10%)
- PortScan:         4,000 samples (8%)
- DDoS:             3,500 samples (7%)
- DoS GoldenEye:    2,000 samples (4%)
- FTP-Patator:      1,500 samples (3%)
- SSH-Patator:      1,500 samples (3%)
- DoS slowloris:    1,000 samples (2%)
- DoS Slowhttptest:   750 samples (1.5%)
- Bot:                500 samples (1%)
- Web Attack Brute Force: 250 samples (0.5%)
- Web Attack XSS:     250 samples (0.5%)
- Infiltration:       100 samples (0.2%)
- Web Attack SQLi:    100 samples (0.2%)
- Heartbleed:          50 samples (0.1%)
```

The ratio between largest and smallest class is **600:1**, well beyond what standard ML can handle without specialized techniques.

### 4.4 Testing Gaps (Medium)

| Test Type | Coverage |
|-----------|----------|
| Import Tests | ✓ Complete (8 tests) |
| Data Generation | ✓ Basic (2 tests) |
| Preprocessing | ✓ Basic (2 tests) |
| Mini Pipeline | ✓ 1 test |
| Model Training | ✗ None |
| Evaluation Metrics | ✗ None |
| Explainability | ✗ None |
| Regression (output stability) | ✗ None |

---

## 5. Technical Debt

### 5.1 Code Quality
- **XGBoost deprecated parameter**: `use_label_encoder=False` is deprecated but still functional
- **No hyperparameter tuning**: Fixed hyperparameters, no grid search or cross-validation
- **No error handling** in explainability for edge cases

### 5.2 Reproducibility
- No centralized configuration file
- Random seeds not explicitly documented in all modules
- No dataset manifest or checksums

### 5.3 Infrastructure
- No Docker support
- No API layer for inference
- Demo uses heuristic simulation instead of real model inference

---

## 6. Deployment Status

| Component | Status | URL/Notes |
|-----------|--------|-----------|
| GitHub Repository | ✓ Active | https://github.com/MohammadThabetHassan/Explainable-AI-Intrusion-Detection-System-XAI-IDS- |
| GitHub Pages | ✓ Deployed | https://mohammadthabethassan.github.io/Explainable-AI-Intrusion-Detection-System-XAI-IDS-/ |
| CI Pipeline | ✓ Functional | Runs on Python 3.10, 3.11 |
| Tests | ⚠ Partial | 13 smoke tests, no regression tests |

---

## 7. Priority Recommendations

### Immediate Actions (Week 1)
1. **Add per-class metrics** to README - make class failures visible
2. **Add macro/weighted F1** prominently - the macro 0.44 is the real story
3. **Create this audit document** - transparency about limitations
4. **Document class imbalance** prominently in Limitations section

### Short-term (Month 1)
5. Implement class weighting in training
6. Add cross-validation
7. Generate per-class precision-recall curves
8. Add failure analysis report

### Medium-term (Quarter)
9. Find alternative CIC-IDS-2017 source
10. Run real dataset benchmark
11. Add comprehensive tests
12. Create Docker deployment

---

## 8. Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Academic rejection due to synthetic-only results | HIGH | HIGH | Run real dataset benchmark |
| Portraying demo as production system | HIGH | MEDIUM | Add clear disclaimers |
| Misleading accuracy claims (0.84 is misleading) | HIGH | HIGH | Show macro metrics |
| Code unmaintainable without tests | MEDIUM | MEDIUM | Add regression tests |
| Dataset provenance questions | MEDIUM | LOW | Document data sources |

---

## 9. Conclusion

The XAI-IDS project demonstrates excellent engineering fundamentals and would be suitable for **educational purposes** or **portfolio demonstration**. However, for **academic submission** or **production deployment**, it requires:

1. Real dataset evaluation
2. Class imbalance resolution
3. Comprehensive testing
4. Proper validation methodology

The project has strong foundations - with targeted improvements it can achieve research-grade quality.

---

*This audit was performed as part of project enhancement initiative.*

---

## v2.0.2 Audit Update (April 2026)

### Resolved Issues

| Issue | Status |
|-------|--------|
| XCS LIME component was 0 in all CSVs | ✅ Fixed — scripts/recompute_xcs.py |
| test_model_prediction assumed dict wrapper | ✅ Fixed — tests now pass |
| Model saves used .pkl, Kaggle used .joblib | ✅ Fixed — standardised to .joblib |
| API /xcs-summary endpoint missing | ✅ Added |

### Remaining Research Limitations

| Limitation | Severity | Status |
|-----------|----------|--------|
| CICIDS2018 has 2 classes (14+ available) | HIGH | Pending Kaggle re-run |
| UNSWNB15 Normal class FPR = 14.7% | MEDIUM | Documented |
| Models trained on 2017/2018 data (temporal drift) | MEDIUM | Documented |
| XCS computed on synthetic test set (not real data) | MEDIUM | Noted in scripts/

---

## Threats to Validity

### Construct Validity
- **Does XCS actually measure trustworthiness?** XCS is a composite of confidence, SHAP stability, and SHAP-LIME agreement. While each component has theoretical grounding, the weights (0.4, 0.3, 0.3) are hand-tuned. The learned-weight calibration (IMPROVE 2) partially addresses this by fitting logistic regression on prediction correctness.
- **Are SHAP and LIME appropriate for IDS?** Both methods assume feature independence, which is violated in network flow data (e.g., packet length statistics are correlated). This may inflate or deflate explanation quality.

### Internal Validity
- **No data leakage?** The scaler is fitted only on training data. Train/test splits use stratification. However, the synthetic data generator uses fixed distributions that may not capture real-world correlations.
- **Random seed effects?** All experiments use seed=42. Results may vary with different seeds. The cross-validation framework (5-fold) partially mitigates this.
- **SMOTE oversampling** introduces synthetic minority samples that may create artificial decision boundaries not present in real data.

### External Validity
- **Do results generalize beyond these 3 datasets?** Cross-dataset generalization experiments show significant performance drops (FIX 4), suggesting models are dataset-specific. The 0.216 Jaccard similarity between top features across datasets confirms this.
- **Temporal generalization?** Models trained on 2017/2018 data may not detect modern attack variants. The temporal drift analysis (IMPROVE 5) quantifies this degradation.
- **Synthetic vs real data gap?** The main pipeline uses synthetic data by default. Real data results come only from the Kaggle notebook.

### Reliability
- **Are results reproducible with different seeds?** The pipeline uses seed=42 consistently. Running with different seeds may produce different results, especially for minority classes.
- **Environment reproducibility?** Package versions are pinned in pyproject.toml. However, SHAP and LIME may produce slightly different results across versions due to algorithmic changes.
- **Hardware effects?** Kaggle notebook runs on Tesla T4 GPU. The main pipeline runs on CPU. Results should be consistent but training times differ.