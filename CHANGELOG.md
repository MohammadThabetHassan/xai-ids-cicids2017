# Changelog

All notable changes to this project will be documented in this file.

## [3.0.0] - 2026-04-07

### Added
- **Real-time XCS in API**: `/explain` endpoint now computes full
  XCS = 0.4×Conf + 0.3×(1-SHAP_Instability) + 0.3×Jaccard(SHAP,LIME)
  with `xcs_components` breakdown and `xcs_reliable` flag.
- **Statistical significance testing**: `src/evaluation/stats.py` with
  McNemar's test, paired t-test, and cross-validation with pairwise
  model comparisons.
- **SMOTE oversampling**: `src/models/train.py` now supports
  `use_smote=True` flag for minority class boosting.
- **Adversarial robustness evaluation**: `src/evaluation/adversarial.py`
  with FGSM attacks via ART, accuracy degradation curves, and XCS
  drop analysis on adversarial inputs.
- **Counterfactual explanations**: `src/explainability/counterfactual.py`
  using DiCE library with gradient-free fallback.
- **Temporal drift detection**: `src/evaluation/drift.py` with KS-test
  per feature and sequential temporal split evaluation.
- **Cross-dataset generalization**: `src/evaluation/cross_dataset.py`
  with feature name mapping and domain shift metrics.
- **API /health/features endpoint**: Returns training feature statistics
  for client-side drift detection.
- **Test coverage expanded**: Added test_api.py (all endpoints),
  test_explainability.py (SHAP+LIME+counterfactuals), test_xcs.py
  (XCS formula properties), test_stats.py (McNemar+t-test),
  test_regression.py (performance threshold guards).
- **BENCHMARKS.md**: Comprehensive benchmark table with all models,
  datasets, train/test sizes, accuracy, F1, and latency.

### Changed
- **API version**: 2.0.0 → 3.0.0
- **generate_sample.py**: minimum 200 samples per rare class (was 10)
- **README.md**: Added paper-style abstract and expanded Key Findings
- **AUDIT.md**: Added Threats to Validity section
- **RESULTS.md**: Expanded Key Findings with 8 scientific findings

### Fixed
- **SMOTE type annotation**: Fixed sampling_strategy dict typing
- **API model loading**: Prioritizes .joblib files over .pkl fallbacks

---

## [2.0.2] - 2026-04-06

### Fixed
- **XCS recomputed with full formula**: LIME per-sample Jaccard
  component was 0 in v2.0.1 CSVs because LIME never ran during
  XCS computation. Added scripts/recompute_xcs.py which runs the
  complete XCS = 0.4×Conf + 0.3×(1-Instab) + 0.3×Jaccard formula
  and commits the corrected explanations/xcs_*_v2.csv files.
- **test_model_prediction.py**: fixed dict-vs-bare-model assumption;
  tests now pass on the committed bare-model joblib files.
- **Documentation**: all XCS tables updated with v2 recomputed values.

### Added
- **scripts/recompute_xcs.py** — full XCS recomputation with LIME
- **scripts/verify_results.py** — model artifact integrity checker
- **scripts/generate_xcs_plots.py** — regenerate XCS plots from v2 CSVs
- **api: /xcs-summary endpoint** returns offline XCS evaluation summary

### Changed
- Standardised model save format from .pkl to .joblib throughout
- Updated README.md, RESULTS.md, MODEL_CARD.md, AUDIT.md, PROJECT_STATUS.md

---

## [2.0.1] - 2026-04-05

### Fixed
- **api/app.py `/explain`**: replaced misleading global `feature_importances_`
  with real per-sample SHAP TreeExplainer values; global importances kept as
  fallback if SHAP fails
- **api/app.py XCS score**: added inline documentation clarifying that the
  API returns the confidence component only (fast-path); full
  XCS = 0.4×Conf + 0.3×(1−Instab) + 0.3×Jaccard is computed offline in the
  Kaggle evaluation notebook
- **RESULTS.md**: filled in XCS and SHAP–LIME Jaccard tables with actual
  numbers (previously "See CSV"); added per-class XCS breakdown
- **RESULTS.md**: honest framing for CICIDS2018 — 2 classes captured, not
  "binary classification"; stratified sampling fix applied to notebook
- **README.md**: synced CICIDS2018 class count and XCS results table

### Known limitation (pending re-run)
- CICIDS2018 full multi-class results (14+ classes) require re-running
  `xai_ids_multidataset.ipynb` on Kaggle with `CICIDS2018_ROWS_PER_FILE = 20_000`
  and `load_csv_files_stratified()` — the code fix is already in the notebook

---

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
