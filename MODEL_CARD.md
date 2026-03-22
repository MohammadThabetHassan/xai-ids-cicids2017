# Model Card: XAI-IDS

## Model Details

- **Model Name**: XAI-IDS (Explainable AI Intrusion Detection System)
- **Model Type**: Multi-class classification (15 classes)
- **Algorithm**: Random Forest, XGBoost, Logistic Regression
- **Version**: 1.0.0
- **Release Date**: March 2026
- **Authors**: Mohammad Thabet Hassan, Fahad Sadek, Ahmed Sami
- **Supervisor**: Dr. Mehak Khurana

## Dataset

- **Training Data**: CIC-IDS-2017 (synthetic simulation)
- **Samples**: 50,000 (synthetic) / 2.8M+ (real, optional)
- **Features**: 78 network flow features
- **Classes**: 15 (1 benign + 14 attack types)

### Classes
| Class | Description |
|-------|-------------|
| BENIGN | Normal network traffic |
| DoS Hulk | HTTP flood DoS |
| PortScan | Network port scanning |
| DDoS | Distributed DoS |
| DoS GoldenEye | HTTP DoS |
| FTP-Patator | FTP brute force |
| SSH-Patator | SSH brute force |
| DoS slowloris | Slowloris DoS |
| DoS Slowhttptest | Slow HTTP DoS |
| Bot | Botnet traffic |
| Web Attack - Brute Force | Web brute force |
| Web Attack - XSS | Cross-site scripting |
| Infiltration | Network infiltration |
| Web Attack - Sql Injection | SQL injection |
| Heartbleed | Heartbleed exploit |

## Performance

### Metrics (XGBoost with Balanced Weights)
| Metric | Score |
|--------|-------|
| Accuracy | 0.8250 |
| Precision (weighted) | 0.8158 |
| Recall (weighted) | 0.8250 |
| F1-Score (weighted) | 0.8193 |
| **F1-Score (macro)** | **0.51** |

### Per-Class Performance
- Strong (F1 > 0.9): BENIGN, Bot, PortScan, FTP-Patator, SSH-Patator
- Moderate (F1 0.3-0.9): DDoS, DoS Hulk, Infiltration, Web Attack - XSS
- Weak (F1 < 0.3): DoS GoldenEye, DoS Slowhttptest, DoS slowloris, Heartbleed, SQL Injection

## Limitations

1. **Class Imbalance**: Severe imbalance (600:1 ratio) causes poor detection of minority classes
2. **Synthetic Data**: Default training on synthetic data; real CIC-IDS-2017 requires download
3. **Undetectable Classes**: SQL Injection (0% recall) remains undetectable with current approach

## Usage

### Training
```bash
python run_pipeline.py --sample-size 50000
```

### API
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

### Docker
```bash
docker build -t xai-ids .
docker run -p 8000:8000 xai-ids
```

## Ethical Considerations

- This model is for research/educational purposes
- Does not constitute a production security solution
- False positives may impact legitimate network traffic
- Should be used as part of a defense-in-depth strategy

## References

- Sharafaldin, I., Lashkari, A.H. and Ghorbani, A.A., "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization," ICISSP 2018.
- Lundberg, S.M. and Lee, S.-I., "A Unified Approach to Interpreting Model Predictions," NeurIPS 2017.
- Ribeiro, M.T., Singh, S. and Guestrin, C., "Why Should I Trust You?" KDD 2016.
