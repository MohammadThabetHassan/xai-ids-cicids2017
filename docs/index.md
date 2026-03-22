# XAI-IDS: Explainable AI Intrusion Detection System

## Overview

XAI-IDS is an Explainable AI-based Intrusion Detection System built on the CIC-IDS-2017 dataset. It combines traditional machine learning classifiers with state-of-the-art explainability techniques (SHAP and LIME) to create a transparent and interpretable network intrusion detection pipeline.

## Key Features

- **Multi-model training**: Logistic Regression, Random Forest, and XGBoost
- **Class-balanced training**: Balanced class weights for fairer evaluation
- **Comprehensive evaluation**: Accuracy, Precision, Recall, F1-score (weighted + macro), confusion matrices
- **Cross-validation**: Stratified k-fold CV support
- **Advanced analysis**: Precision-recall curves, calibration curves, failure analysis
- **Global explainability**: SHAP feature importance and summary plots
- **Local explainability**: LIME instance-level explanations for correct and misclassified samples
- **Production-ready**: Docker support, FastAPI inference endpoint, config.yaml
- **CI/CD**: Automated testing and documentation deployment

## Dataset

The CIC-IDS-2017 dataset contains network traffic data with 78 features and 15 traffic classes:
- 1 benign class (normal traffic)
- 14 attack classes (DoS, DDoS, Brute Force, Port Scan, Web Attacks, etc.)

## Getting Started

```bash
# Clone the repository
git clone https://github.com/MohammadThabetHassan/Explainable-AI-Intrusion-Detection-System-XAI-IDS-.git

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python run_pipeline.py
```

## Documentation

- [Methodology](methodology.md) - Detailed technical methodology
- [README](../README.md) - Full project documentation
- [Contributors](../CONTRIBUTORS.md) - Team and acknowledgments

## Links

- **Repository**: [GitHub](https://github.com/MohammadThabetHassan/Explainable-AI-Intrusion-Detection-System-XAI-IDS-)
- **Dataset**: [CIC-IDS-2017](https://www.unb.ca/cic/datasets/ids-2017.html)
