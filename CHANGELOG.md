# Changelog

All notable changes to this project will be documented in this file.

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
- **Testing**: Added comprehensive unit tests (tests/test_evaluation.py)
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
