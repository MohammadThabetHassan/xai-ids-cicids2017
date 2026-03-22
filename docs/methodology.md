# Methodology

## 1. Problem Statement

Network intrusion detection is critical for cybersecurity, but traditional ML-based IDS systems often operate as "black boxes," making it difficult for security analysts to understand why specific traffic is flagged as malicious. This project addresses the need for **explainable** intrusion detection by combining effective ML classifiers with interpretability techniques.

## 2. Dataset: CIC-IDS-2017

The Canadian Institute for Cybersecurity Intrusion Detection System 2017 (CIC-IDS-2017) dataset contains labeled network traffic captured over 5 days. The dataset was generated using CICFlowMeter, which extracts 78 network flow features from pcap files.

### Traffic Classes

| Class | Description |
|-------|-------------|
| BENIGN | Normal network traffic |
| DoS Hulk | HTTP flood DoS attack |
| PortScan | Network port scanning |
| DDoS | Distributed Denial of Service |
| DoS GoldenEye | HTTP DoS attack using GoldenEye tool |
| FTP-Patator | FTP brute force attack |
| SSH-Patator | SSH brute force attack |
| DoS slowloris | Slowloris DoS attack |
| DoS Slowhttptest | Slow HTTP DoS attack |
| Bot | Botnet traffic |
| Web Attack - Brute Force | Web application brute force |
| Web Attack - XSS | Cross-site scripting attack |
| Infiltration | Network infiltration |
| Web Attack - Sql Injection | SQL injection attack |
| Heartbleed | Heartbleed vulnerability exploit |

### Features

The 78 features include:
- **Flow-level**: Duration, packet counts, byte counts
- **Packet-level**: Packet length statistics (min, max, mean, std)
- **Inter-arrival time**: Flow IAT, forward IAT, backward IAT
- **Flag-based**: TCP flag counts (SYN, FIN, RST, PSH, ACK, URG)
- **Subflow**: Subflow packet and byte counts
- **Window**: Initial TCP window sizes

## 3. Data Preprocessing

1. **Loading**: Memory-efficient chunked CSV reading with error handling
2. **Cleaning**: Remove infinite values, NaN values, and duplicates
3. **Encoding**: Label encoding for the 15 traffic classes
4. **Splitting**: Stratified 70/10/20 train/validation/test split
5. **Scaling**: StandardScaler fitted on training data only

## 4. Models

### Logistic Regression (Baseline)
- Linear model for multi-class classification
- Uses L-BFGS solver with multinomial loss
- Serves as interpretable baseline

### Random Forest
- Ensemble of 100 decision trees
- Max depth of 20 with minimum sample constraints
- Provides built-in feature importance

### XGBoost
- Gradient boosted trees with histogram-based splitting
- 100 estimators, max depth 8, learning rate 0.1
- Evaluated with multi-class log loss on validation set

## 5. Evaluation Metrics

For each model, we compute:
- **Accuracy**: Overall correct classification rate
- **Precision** (weighted): Proportion of true positives among predicted positives
- **Recall** (weighted): Proportion of true positives identified
- **F1-Score** (weighted): Harmonic mean of precision and recall
- **F1-Score** (macro): Treats all classes equally (important for imbalanced data)
- **Confusion Matrix**: Full class-level prediction breakdown
- **Normalized Confusion Matrix**: Row-normalized for class-level comparison

### Advanced Evaluation

The pipeline also supports:
- **Cross-validation**: Stratified k-fold CV (`--cv-folds`)
- **Precision-Recall curves**: Per-class PR analysis (`--pr-curves`)
- **Calibration curves**: Model confidence vs accuracy (`--calibration`)
- **Failure analysis**: Identifies failing classes and top confusions (`--failure-analysis`)

### Class Imbalance Handling

All models use balanced class weights to address the severe imbalance (600:1 ratio) in the dataset:
- Logistic Regression: `class_weight='balanced'`
- Random Forest: `class_weight='balanced'`
- XGBoost: Sample weights based on class frequencies

This improves macro F1 from 0.44 to 0.51, trading some weighted F1 for fairer per-class performance.

## 6. Explainability

### SHAP (SHapley Additive exPlanations)

SHAP provides **global** feature importance by computing the average contribution of each feature to the model's predictions across many samples.

- **TreeExplainer**: Used for Random Forest and XGBoost (exact, fast)
- **KernelExplainer**: Used for Logistic Regression (model-agnostic)
- **Outputs**: Summary plots and feature importance bar charts

### LIME (Local Interpretable Model-agnostic Explanations)

LIME provides **local** explanations for individual predictions by fitting interpretable surrogate models around specific data points.

- Explains both correctly classified and misclassified instances
- Shows which features pushed the prediction toward each class
- Outputs: Feature contribution plots and text reports

## 7. Reproducibility

- Random seed (42) used consistently throughout the pipeline
- Deterministic train/test splitting with stratification
- All preprocessing artifacts (scaler, encoder) saved for inference
- Full pipeline executable via single command: `python run_pipeline.py`

## References

1. Sharafaldin, I., Lashkari, A.H. and Ghorbani, A.A., "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization," ICISSP 2018.
2. Lundberg, S.M. and Lee, S.-I., "A Unified Approach to Interpreting Model Predictions," NeurIPS 2017.
3. Ribeiro, M.T., Singh, S. and Guestrin, C., "Why Should I Trust You? Explaining the Predictions of Any Classifier," KDD 2016.
