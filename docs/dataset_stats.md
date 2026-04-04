# XAI-IDS Dataset Statistics

## CIC-IDS-2017 Dataset

### Overview
- **Total Records**: 2,830,743
- **Features**: 78 original, 20 selected for Kaggle evaluation
- **Classes**: 14 (1 benign + 13 attack types)
- **Collection Period**: 5 days (Monday-Friday)
- **Source**: Canadian Institute for Cybersecurity (CIC)

### Selected Features (20)
| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | Fwd Seg Size Min | Numeric | Minimum segment size in forward direction |
| 2 | Fwd IAT Total | Numeric | Total inter-arrival time (forward) |
| 3 | Fwd IAT Std | Numeric | Standard deviation of forward IAT |
| 4 | Flow IAT Mean | Numeric | Mean inter-arrival time (flow) |
| 5 | Fwd IAT Mean | Numeric | Mean inter-arrival time (forward) |
| 6 | Avg Packet Size | Numeric | Average packet size |
| 7 | Packet Length Mean | Numeric | Mean packet length |
| 8 | Packet Length Max | Numeric | Maximum packet length |
| 9 | Flow IAT Std | Numeric | Standard deviation of flow IAT |
| 10 | Bwd IAT Std | Numeric | Standard deviation of backward IAT |
| 11 | Bwd Packet Length Max | Numeric | Maximum backward packet length |
| 12 | Flow IAT Max | Numeric | Maximum flow IAT |
| 13 | Bwd Packet Length Min | Numeric | Minimum backward packet length |
| 14 | Flow Duration | Numeric | Total duration of the flow |
| 15 | Fwd IAT Max | Numeric | Maximum forward IAT |
| 16 | Bwd Packet Length Mean | Numeric | Mean backward packet length |
| 17 | Bwd IAT Max | Numeric | Maximum backward IAT |
| 18 | Packet Length Std | Numeric | Standard deviation of packet length |
| 19 | Bwd Packet Length Std | Numeric | Standard deviation of backward packet length |
| 20 | PSH Flag Count | Numeric | Count of PSH flags |

### Class Distribution
| Class | Count | Percentage |
|-------|-------|------------|
| BENIGN | 2,273,097 | 80.3% |
| DoS Hulk | 231,073 | 8.2% |
| PortScan | 158,930 | 5.6% |
| DDoS | 128,027 | 4.5% |
| DoS GoldenEye | 10,293 | 0.4% |
| FTP-Patator | 7,938 | 0.3% |
| SSH-Patator | 5,897 | 0.2% |
| DoS slowloris | 5,796 | 0.2% |
| DoS Slowhttptest | 5,499 | 0.2% |
| Bot | 1,966 | 0.1% |
| Web Attack - Brute Force | 1,507 | 0.05% |
| Web Attack - XSS | 652 | 0.02% |
| Infiltration | 36 | <0.01% |
| Web Attack - Sql Injection | 21 | <0.01% |

### Preprocessing
- **Scaler**: RobustScaler (fitted on training data only)
- **Split**: 70% train, 10% validation, 20% test
- **SMOTE**: Applied for class balance in Kaggle notebook
- **Feature Selection**: Top 20 by XGBoost importance

### Model Performance
| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| XGBoost | 0.9966 | 0.9964 | 0.9966 | 0.9964 |
| VotingEnsemble | 0.9886 | 0.9945 | 0.9886 | 0.9911 |
| RandomForest | 0.9857 | 0.9940 | 0.9857 | 0.9893 |
| LightGBM | 0.9744 | 0.9930 | 0.9744 | 0.9828 |
