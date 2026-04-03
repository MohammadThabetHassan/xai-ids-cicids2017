# Changelog

All notable changes to this project will be documented in this file.

## [2.0.0] - 2026-04-03

### Added
- **Multi-Dataset Support**: Evaluation on 3 datasets (CIC-IDS-2017, UNSW-NB15, CSE-CIC-IDS-2018)
- **New Models**: LightGBM and VotingEnsemble (from Kaggle notebook)
- **XCS Metric**: Novel XAI Confidence Score for explanation reliability
- **RESULTS.md**: Comprehensive results tables for all datasets and models
- **CONTRIBUTING.md**: Contribution guidelines
- **SECURITY.md**: Security policy and best practices
- **PR Template**: Pull request template for consistent contributions
- **Model Prediction Tests**: Tests for loading saved models and making predictions
- **Example Scripts**: `examples/predict_example.py` for model usage
- **API Improvements**: XCS score in predictions, lifespan context manager, input validation

### Changed
- **README.md**: Complete overhaul with architecture diagram, multi-dataset results, XCS formula
- **MODEL_CARD.md**: Updated with actual metrics from all 3 datasets
- **docs/methodology.md**: Added XCS formula, multi-dataset discussion, UNSW-NB15 and CICIDS2018 details
- **ROADMAP.md**: Updated to reflect completed phases
- **BENCHMARK_PLAN.md**: Updated with completed benchmark results
- **api/app.py**: Modernized with FastAPI lifespan, improved model loading, input validation
- **requirements.txt**: Added lightgbm, imbalanced-learn; pinned upper bounds
- **CITATION.cff**: Removed placeholder ORCID
- **.gitignore**: Added __MACOSX/, .DS_Store, .ipynb_checkpoints/, logs

### Fixed
- XGBoost deprecated `use_label_encoder` parameter removed
- File handle leak in `src/data/loader.py`
- HTTP URL changed to HTTPS in `src/data/download.py`
- Typo fix: `MACHINELEARNIGCVE_PATH` → `MACHINE_LEARNING_CVE_PATH`
- Unicode corruption in `model_metadata.json` class names
- Removed committed `__MACOSX/` macOS artifact directory

### Removed
- `pyyaml` from requirements.txt (config.yaml is reference-only, not loaded)

---

## [1.0.0] - 2026-03-22

### Added
- **Models**: Added balanced class weights to address class imbalance
  - Logistic Regression: class_weight='balanced'
  - Random Forest: class_weight='balanced'
  - XGBoost: sample_weight based on class frequencies
- **Evaluation**: Added cross-validation support (`--cv-folds`)
- **Evaluation**: Added precision-recall curves (`--pr-curves`)
- **Evaluation**: Added calibration curves (`--calibration`)
- **Evaluation**: Added failure analysis report (`--failure-analysis`)
- **Data**: Added Zenodo fallback for CIC-IDS-2017 download
- **API**: Added FastAPI inference endpoint (api/app.py)
- **Infrastructure**: Added config.yaml for pipeline configuration
- **Infrastructure**: Added Dockerfile for containerized deployment
- **Testing**: Added unit tests (tests/test_evaluation.py)
- **Documentation**: Added MODEL_CARD.md
- **Documentation**: Added AUDIT.md, ROADMAP.md, PROJECT_STATUS.md

### Changed
- README.md: Updated results with per-class metrics
- README.md: Expanded limitations section with class imbalance details
- ROADMAP.md: Updated with completed items

### Performance
- Macro F1 improved from 0.44 to 0.51
- 4 of 5 previously failing attack classes now have detection

### Known Limitations
- SQL Injection remains at 0% detection (20 samples)
- DoS attacks commonly confused with each other
- Default training on synthetic data

---

## [0.0.1] - 2026-01-01

### Added
- Initial release
- Logistic Regression, Random Forest, XGBoost models
- SHAP and LIME explainability
- Synthetic data generation
- GitHub Pages documentation
