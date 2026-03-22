# XAI-IDS: Explainable AI Intrusion Detection System

[![CI](https://github.com/MohammadThabetHassan/Explainable-AI-Intrusion-Detection-System-XAI-IDS-/actions/workflows/ci.yml/badge.svg)](https://github.com/MohammadThabetHassan/Explainable-AI-Intrusion-Detection-System-XAI-IDS-/actions/workflows/ci.yml)
[![Pages](https://github.com/MohammadThabetHassan/Explainable-AI-Intrusion-Detection-System-XAI-IDS-/actions/workflows/pages.yml/badge.svg)](https://github.com/MohammadThabetHassan/Explainable-AI-Intrusion-Detection-System-XAI-IDS-/actions/workflows/pages.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An Explainable AI-based Intrusion Detection System that combines machine learning classifiers with SHAP and LIME explainability techniques on the CIC-IDS-2017 dataset.

**Live Documentation:** [https://mohammadthabethassan.github.io/Explainable-AI-Intrusion-Detection-System-XAI-IDS-/](https://mohammadthabethassan.github.io/Explainable-AI-Intrusion-Detection-System-XAI-IDS-/)

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Installation & Setup](#installation--setup)
- [How to Run](#how-to-run)
- [Pipeline Overview](#pipeline-overview)
- [Models](#models)
- [Results](#results)
- [Explainability](#explainability)
- [Repository Structure](#repository-structure)
- [Testing](#testing)
- [Limitations & Future Work](#limitations--future-work)
- [Authors & Contributors](#authors--contributors)

---

## Problem Statement

Network intrusion detection is critical for cybersecurity. Traditional ML-based IDS systems often operate as "black boxes," making it difficult for security analysts to understand why specific traffic is flagged as malicious. This project addresses the need for **explainable** intrusion detection by combining effective ML classifiers with state-of-the-art interpretability techniques.

**Objectives:**
1. Build an ML pipeline for multi-class network intrusion detection
2. Compare multiple classifiers (Logistic Regression, Random Forest, XGBoost)
3. Provide transparent explanations using SHAP (global) and LIME (local)
4. Create a reproducible, production-quality research pipeline

---

## Dataset

### CIC-IDS-2017

The **CIC-IDS-2017** dataset was created by the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html) at the University of New Brunswick. It contains labeled network traffic captured over 5 days with both benign activity and common attack types.

**Key characteristics:**
- **78 network flow features** extracted using CICFlowMeter
- **15 traffic classes** (1 benign + 14 attack types)
- **2.8+ million flow records** across 8 CSV files

| Class | Description |
|-------|-------------|
| BENIGN | Normal network traffic |
| DoS Hulk | HTTP flood denial-of-service |
| PortScan | Network port scanning |
| DDoS | Distributed denial-of-service |
| DoS GoldenEye | HTTP DoS using GoldenEye tool |
| FTP-Patator | FTP brute force |
| SSH-Patator | SSH brute force |
| DoS slowloris | Slowloris DoS attack |
| DoS Slowhttptest | Slow HTTP DoS attack |
| Bot | Botnet traffic |
| Web Attack - Brute Force | Web application brute force |
| Web Attack - XSS | Cross-site scripting |
| Infiltration | Network infiltration |
| Web Attack - Sql Injection | SQL injection |
| Heartbleed | Heartbleed vulnerability exploit |

**Source:** 
- Primary: [UNB CIC](https://www.unb.ca/cic/datasets/ids-2017.html) (server may be offline)
- Fallback: [Zenodo](https://zenodo.org/records/10141593) (CIC-IDS-2017 V2, 369MB)

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- pip

### Install

```bash
# Clone the repository
git clone https://github.com/MohammadThabetHassan/Explainable-AI-Intrusion-Detection-System-XAI-IDS-.git
cd Explainable-AI-Intrusion-Detection-System-XAI-IDS-

# Install dependencies
pip install -r requirements.txt
```

---

## How to Run

### Full Pipeline (Synthetic Data)

```bash
python run_pipeline.py
```

### Download Real CIC-IDS-2017 Dataset

```bash
python run_pipeline.py --download
```

### Quick Test (Small Dataset)

```bash
python run_pipeline.py --sample-size 5000 --skip-explain
```

### CLI Options

| Flag | Description |
|------|-------------|
| `--download` | Download real CIC-IDS-2017 data (via Zenodo fallback) |
| `--sample-size N` | Synthetic dataset size (default: 50000) |
| `--skip-explain` | Skip SHAP and LIME analysis |
| `--skip-shap` | Skip SHAP only |
| `--skip-lime` | Skip LIME only |
| `--models lr rf xgb` | Select specific models |
| `--shap-samples N` | Samples for SHAP (default: 500) |
| `--cv-folds N` | Run N-fold cross-validation (0 to skip) |
| `--pr-curves` | Generate precision-recall curves |
| `--calibration` | Generate calibration curves |
| `--failure-analysis` | Generate failure analysis report |

### Using Make

```bash
make pipeline          # Full pipeline
make pipeline-small    # Quick test with small data
make pipeline-fast     # Skip explainability
make test              # Run tests
make clean             # Remove outputs
```

---

## Pipeline Overview

```
Data Acquisition → Preprocessing → Model Training → Evaluation → Explainability
```

1. **Data Acquisition**: Download CIC-IDS-2017 or generate synthetic data
2. **Preprocessing**: Clean (NaN, Inf, duplicates), encode labels, scale features, stratified split (70/10/20)
3. **Model Training**: Train Logistic Regression, Random Forest, and XGBoost
4. **Evaluation**: Compute metrics, confusion matrices, comparison charts
5. **Explainability**: SHAP global analysis + LIME local explanations

---

## Models

| Model | Type | Key Parameters |
|-------|------|---------------|
| **Logistic Regression** | Linear | solver=lbfgs, max_iter=1000, C=1.0 |
| **Random Forest** | Ensemble | n_estimators=100, max_depth=20 |
| **XGBoost** | Boosting | n_estimators=100, max_depth=8, lr=0.1 |

---

## Results

### Real Data Benchmark (CIC-IDS-2017 V2)

**Dataset:** 100K random sample from real CIC-IDS-2017 V2 (Zenodo)
**Classes:** 14 (includes new "Comb" class)

| Model | Accuracy | Precision | Recall | F1-Score (Weighted) |
|-------|----------|-----------|--------|---------------------|
| Logistic Regression | 0.8569 | 0.9637 | 0.8569 | 0.8994 |
| **Random Forest** | **0.9969** | **0.9971** | **0.9969** | **0.9970** |

### Per-Class Performance (Random Forest - Real Data, 100K)

| Class | Precision | Recall | F1-Score | Samples |
|-------|-----------|--------|----------|---------|
| BENIGN | 1.00 | 1.00 | 1.00 | 15542 |
| Bot | 0.44 | 0.69 | 0.54 | 16 |
| Comb | 1.00 | 1.00 | 1.00 | 1253 |
| DDoS | 1.00 | 1.00 | 1.00 | 951 |
| DoS GoldenEye | 1.00 | 1.00 | 1.00 | 71 |
| DoS Hulk | 1.00 | 0.99 | 0.99 | 1255 |
| DoS Slowhttptest | 0.89 | 0.95 | 0.92 | 43 |
| DoS slowloris | 1.00 | 1.00 | 1.00 | 33 |
| FTP-Patator | 1.00 | 1.00 | 1.00 | 42 |
| PortScan | 0.98 | 1.00 | 0.99 | 743 |
| SSH-Patator | 1.00 | 0.97 | 0.98 | 30 |
| Web Attack - Brute Force | 0.75 | 0.67 | 0.71 | 9 |
| Web Attack - XSS | 0.67 | 0.50 | 0.57 | 4 |

**Summary Metrics:**
- **Macro Avg F1:** 0.90 (treats all classes equally)
- **Weighted Avg F1:** 1.00 (favors majority classes)

---

### Synthetic Data Results (For Comparison)

| Model | Accuracy | Precision | Recall | F1-Score (Weighted) | Macro F1 |
|-------|----------|-----------|--------|---------------------|----------|
| Logistic Regression | 0.7601 | 0.7866 | 0.7601 | 0.7651 | 0.48 |
| Random Forest | 0.8347 | 0.8096 | 0.8347 | 0.8162 | 0.48 |
| **XGBoost** | **0.8250** | **0.8158** | **0.8250** | **0.8193** | **0.51** |

> Note: Real data (99.69% RF accuracy) significantly outperforms synthetic (83.47% RF accuracy). The synthetic data had 5 classes with 0% detection; real data has near-perfect detection for most classes.

### Model Comparison Chart

![Model Comparison](outputs/figures/model_comparison.png)

### Confusion Matrices

#### Logistic Regression
![LR Confusion Matrix](outputs/figures/confusion_matrix_logistic_regression_normalized.png)

#### Random Forest
![RF Confusion Matrix](outputs/figures/confusion_matrix_random_forest_normalized.png)

#### XGBoost
![XGB Confusion Matrix](outputs/figures/confusion_matrix_xgboost_normalized.png)

---

## Explainability

### SHAP - Global Feature Importance

SHAP (SHapley Additive exPlanations) provides global feature importance by computing the average contribution of each feature to predictions.

#### Random Forest - SHAP Feature Importance
![RF SHAP](outputs/figures/shap_feature_importance_random_forest.png)

#### XGBoost - SHAP Feature Importance
![XGB SHAP](outputs/figures/shap_feature_importance_xgboost.png)

### LIME - Local Explanations

LIME (Local Interpretable Model-agnostic Explanations) explains individual predictions by fitting surrogate models around specific data points.

#### Correctly Classified Sample (Random Forest)
![RF LIME Correct](outputs/figures/lime_correct_random_forest.png)

#### Misclassified Sample (XGBoost)
![XGB LIME Misclassified](outputs/figures/lime_misclassified_xgboost.png)

### Generating Explainability Outputs

```bash
# Full pipeline with explainability
python run_pipeline.py

# Explainability with custom SHAP sample size
python run_pipeline.py --shap-samples 200
```

---

## Repository Structure

```
XAI-IDS/
├── .github/
│   └── workflows/
│       ├── ci.yml                    # CI workflow (tests + imports)
│       └── pages.yml                 # GitHub Pages deployment
├── data/
│   ├── raw/                          # Raw CSV files
│   └── processed/                    # Processed data
├── notebooks/
│   └── exploration.ipynb             # Interactive demo notebook
├── src/
│   ├── data/
│   │   ├── download.py               # Dataset download utility
│   │   ├── generate_sample.py        # Synthetic data generator
│   │   ├── loader.py                 # Chunked CSV loader
│   │   └── preprocessing.py          # Cleaning, encoding, scaling
│   ├── features/
│   │   └── feature_engineering.py    # Feature selection utilities
│   ├── models/
│   │   └── train.py                  # Model training (LR, RF, XGB)
│   ├── evaluation/
│   │   └── metrics.py                # Metrics, confusion matrices, charts
│   ├── explainability/
│   │   └── explain.py                # SHAP and LIME implementations
│   └── utils/
│       └── logger.py                 # Centralized logging
├── outputs/
│   ├── figures/                      # Confusion matrices, SHAP, LIME plots
│   ├── models/                       # Trained models (.pkl)
│   ├── logs/                         # Pipeline logs
│   ├── reports/                      # Classification reports, LIME reports
│   └── results_metrics.csv           # Model comparison metrics
├── docs/
│   ├── index.md                      # Documentation home
│   └── methodology.md               # Detailed methodology
├── tests/
│   └── test_smoke.py                 # Smoke tests for pipeline
├── README.md                         # This file
├── CONTRIBUTORS.md                   # Team and acknowledgments
├── requirements.txt                  # Python dependencies
├── run_pipeline.py                   # Main pipeline entry point
└── Makefile                          # Build automation
```

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run smoke tests only
python -m pytest tests/test_smoke.py -v -x
```

Tests include:
- Module import verification
- Synthetic data generation
- Data preprocessing pipeline
- Mini end-to-end pipeline execution

---

## Limitations & Future Work

### Limitations

**Mitigated Issues:**

1. **Class Imbalance (Improved)**: Added balanced class weights. Macro F1 improved from 0.44 to 0.51. Only SQL Injection (20 samples) remains at 0% detection.

**Remaining Issues:**

2. **Real Data vs Synthetic**: Real CIC-IDS-2017 V2 benchmark (100K sample) shows 99.69% accuracy vs 83.47% for synthetic. Use `--random-sample` to run benchmarks.

3. **XGBoost Label Issue**: XGBoost fails with rare classes that get dropped during stratified split. Random Forest handles this better.

4. **Class Distribution**: Real data has severe imbalance (~600:1 ratio) but Random Forest handles it well with balanced class weights.

**Other Limitations:**
- Current benchmark uses 100K sample from 2.8M+ CIC-IDS-2017 V2 dataset
- Full dataset benchmark would provide more comprehensive evaluation (use --sample-size)
- SHAP KernelExplainer for Logistic Regression is computationally expensive on large datasets

---

## Deployment

### Docker

```bash
# Build the container
docker build -t xai-ids .

# Run the pipeline
docker run -v $(pwd)/outputs:/app/outputs xai-ids
```

### FastAPI Inference

```bash
# Start the API server
uvicorn api.app:app --host 0.0.0.0 --port 8000

# Test the API
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.0] * 78}'
```

### Configuration

The project includes `config.yaml` with default settings. This serves as a reference for pipeline configuration.

---

### Future Work
- Evaluate on the full CIC-IDS-2017 dataset
- Implement deep learning models (LSTM, Transformer-based)
- Add real-time inference pipeline
- Explore additional XAI techniques (Counterfactual explanations, Anchors)
- Apply oversampling (SMOTE) to handle class imbalance
- Deploy as a web application with interactive dashboards

---

## Authors & Contributors

### Authors

| Name | Role | GitHub |
|------|------|--------|
| **Mohammad Thabet Hassan** | Lead Developer & Primary Contributor | [@MohammadThabetHassan](https://github.com/MohammadThabetHassan) |
| **Fahad Sadek** | Contributor | [@fahad6789123](https://github.com/fahad6789123) |
| **Ahmed Sami** | Contributor | [@AhmedSamiAlameri](https://github.com/AhmedSamiAlameri) |

### Supervisor

**Dr. Mehak Khurana** - Mehak.Khurana@cud.ac.ae

### Acknowledgments

- [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/) - CIC-IDS-2017 dataset
- [SHAP](https://github.com/shap/shap) - Lundberg & Lee, NeurIPS 2017
- [LIME](https://github.com/marcotcr/lime) - Ribeiro, Singh & Guestrin, KDD 2016

---

## License

This project is developed for academic and research purposes.

## References

1. Sharafaldin, I., Lashkari, A.H. and Ghorbani, A.A., "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization," ICISSP 2018.
2. Lundberg, S.M. and Lee, S.-I., "A Unified Approach to Interpreting Model Predictions," NeurIPS 2017.
3. Ribeiro, M.T., Singh, S. and Guestrin, C., "Why Should I Trust You? Explaining the Predictions of Any Classifier," KDD 2016.
