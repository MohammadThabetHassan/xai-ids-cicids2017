# Methodology

## 1. Problem Statement

Network intrusion detection is critical for cybersecurity, but traditional ML-based IDS systems often operate as "black boxes," making it difficult for security analysts to understand why specific traffic is flagged as malicious. This project addresses the need for **explainable** intrusion detection by combining effective ML classifiers with interpretability techniques (SHAP, LIME) and a novel XAI Confidence Score (XCS).

## 2. Datasets

### 2.1 CIC-IDS-2017

The Canadian Institute for Cybersecurity Intrusion Detection System 2017 (CIC-IDS-2017) dataset contains labeled network traffic captured over 5 days. The dataset was generated using CICFlowMeter, which extracts 78 network flow features from pcap files.

#### Traffic Classes

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

#### Features

The 78 features include:
- **Flow-level**: Duration, packet counts, byte counts
- **Packet-level**: Packet length statistics (min, max, mean, std)
- **Inter-arrival time**: Flow IAT, forward IAT, backward IAT
- **Flag-based**: TCP flag counts (SYN, FIN, RST, PSH, ACK, URG)
- **Subflow**: Subflow packet and byte counts
- **Window**: Initial TCP window sizes

### 2.2 UNSW-NB15

The UNSW-NB15 dataset was created by the Australian Centre for Cyber Security. It contains a mix of real normal network activity and synthetic attack traffic.

- **10 classes**: Normal + 9 attack types (Analysis, Backdoor, DoS, Exploits, Fuzzers, Generic, Reconnaissance, Shellcode, Worms)
- **49 features**, reduced to 20 most important for this study
- **2.5M+ records**

### 2.3 CSE-CIC-IDS-2018

An extension of CIC-IDS-2017 with additional attack scenarios.

- **2 classes**: Benign + DDoS attacks-LOIC-HTTP (binary classification)
- **80 features**, reduced to 20 most important for this study
- **10M+ records**

## 3. Data Preprocessing

1. **Loading**: Memory-efficient chunked CSV reading with error handling
2. **Cleaning**: Remove infinite values, NaN values, and duplicates
3. **Encoding**: Label encoding for traffic classes
4. **Splitting**: Stratified 70/10/20 train/validation/test split
5. **Scaling**: StandardScaler fitted on training data only (no data leakage)
6. **Feature Selection**: For Kaggle multi-dataset evaluation, top 20 features selected by importance per dataset

## 4. Models

### 4.1 Logistic Regression (Baseline)
- Linear model for multi-class classification
- Uses L-BFGS solver with multinomial loss
- Serves as interpretable baseline
- Parameters: `max_iter=1000`, `C=1.0`, `class_weight='balanced'`

### 4.2 Random Forest
- Ensemble of 100 decision trees
- Max depth of 20 with minimum sample constraints
- Provides built-in feature importance
- Parameters: `n_estimators=100`, `max_depth=20`, `class_weight='balanced'`

### 4.3 XGBoost
- Gradient boosted trees with histogram-based splitting
- Primary model for SHAP explanations (TreeExplainer)
- Parameters: `n_estimators=100`, `max_depth=8`, `learning_rate=0.1`, `tree_method='hist'`

### 4.4 LightGBM (Kaggle Notebook)
- Gradient boosting framework optimized for speed and memory
- Trained on Kaggle with GPU (Tesla T4)
- Parameters tuned per dataset

### 4.5 VotingEnsemble (Kaggle Notebook)
- Soft voting ensemble of XGBoost, Random Forest, and LightGBM
- Combines predictions using weighted average of class probabilities

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

### 6.1 SHAP (SHapley Additive exPlanations)

SHAP provides **global** feature importance by computing the average contribution of each feature to the model's predictions across many samples.

- **TreeExplainer**: Used for Random Forest and XGBoost (exact, fast)
- **KernelExplainer**: Used for Logistic Regression (model-agnostic)
- **Outputs**: Summary plots, beeswarm plots, waterfall plots, and feature importance bar charts

### 6.2 LIME (Local Interpretable Model-agnostic Explanations)

LIME provides **local** explanations for individual predictions by fitting interpretable surrogate models around specific data points.

- Explains both correctly classified and misclassified instances
- Shows which features pushed the prediction toward each class
- Outputs: Feature contribution plots and text reports

### 6.3 XAI Confidence Score (XCS)

The **XCS** is a novel metric that quantifies the reliability of explanations for individual predictions:

```
XCS = w1 × Confidence + w2 × (1 - SHAP_Instability) + w3 × Jaccard(SHAP, LIME)
```

Where:
- **w1 = 0.4**: Weight for model prediction confidence (P(y|x))
- **w2 = 0.3**: Weight for SHAP explanation stability (1 - feature ranking variance across perturbations)
- **w3 = 0.3**: Weight for agreement between SHAP and LIME (Jaccard similarity of top-k features)

**Component Details:**

1. **Confidence**: The model's predicted probability for the predicted class. Higher confidence suggests the model is more certain about its prediction.

2. **SHAP Instability**: Measures how much the SHAP feature rankings change when the input is slightly perturbed. Computed as:
   ```
   SHAP_Instability = 1 - (number of stable top-k features / k)
   ```
   Lower instability means more reliable explanations.

3. **Jaccard(SHAP, LIME)**: Measures agreement between SHAP and LIME on the top-k most important features:
   ```
   Jaccard = |SHAP_top_k ∩ LIME_top_k| / |SHAP_top_k ∪ LIME_top_k|
   ```
   Higher agreement means both methods identify the same important features.

**Interpretation:**
- **XCS > 0.7**: High-confidence, reliable explanation — trust the prediction and its explanation
- **XCS 0.3–0.7**: Moderate confidence — use with caution, consider human review
- **XCS < 0.3**: Low-confidence explanation — do not trust the prediction without additional verification

**Threshold**: XCS < 0.3 flags predictions that should be reviewed by a security analyst.

### 6.4 Cross-Dataset Analysis

We compare feature importance across all 3 datasets using:
- **Jaccard similarity** of top-k features between datasets
- **Visual comparison** of SHAP feature importance rankings
- **Feature overlap analysis** to identify universal vs dataset-specific indicators

Cross-dataset Jaccard similarity: **0.216**, indicating that different datasets rely on different features for predictions.

## 7. Reproducibility

- Random seed (42) used consistently throughout the pipeline
- Deterministic train/test splitting with stratification
- All preprocessing artifacts (scaler, encoder) saved for inference
- Full pipeline executable via single command: `python run_pipeline.py`
- Multi-dataset results reproducible via Kaggle notebook with GPU (Tesla T4)

## 8. Pipeline Architecture

```
Data Acquisition → Preprocessing → Model Training → Evaluation → Explainability → XCS Scoring
     ↓                  ↓                ↓              ↓            ↓              ↓
  3 IDS Datasets    Clean/Encode    4 Models       Metrics     SHAP/LIME     Confidence
  (CICIDS2017,      Scale/Split     (XGB, RF,      Confusion   Global/Local  Score
   UNSW-NB15,                        LGBM, Ens)     Matrices    Explanations
   CICIDS2018)
```

## References

1. Sharafaldin, I., Lashkari, A.H. and Ghorbani, A.A., "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization," ICISSP 2018.
2. Lundberg, S.M. and Lee, S.-I., "A Unified Approach to Interpreting Model Predictions," NeurIPS 2017.
3. Ribeiro, M.T., Singh, S. and Guestrin, C., "Why Should I Trust You? Explaining the Predictions of Any Classifier," KDD 2016.
4. Moustafa, N. and Slay, J., "UNSW-NB15: a comprehensive data set for network intrusion detection systems," MilCIS 2015.
