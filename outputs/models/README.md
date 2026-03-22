# XAI-IDS: Explainable AI Intrusion Detection System

## Model Metadata

### Preprocessing
- Scaler: StandardScaler (fitted on training data)
- Label Encoder: LabelEncoder (15 classes)
- Random State: 42

### Training Configuration
- Train/Val/Test Split: 70/10/20
- Stratified splitting

### Saved Artifacts
- `label_encoder.pkl`: Fitted LabelEncoder
- `scaler.pkl`: Fitted StandardScaler
- `logistic_regression.pkl`: Trained Logistic Regression model
- `random_forest.pkl`: Trained Random Forest model
- `xgboost.pkl`: Trained XGBoost model

### Reproducing Inference
```python
import joblib
model = joblib.load('outputs/models/random_forest.pkl')
scaler = joblib.load('outputs/models/scaler.pkl')
le = joblib.load('outputs/models/label_encoder.pkl')

X_scaled = scaler.transform(X_new)
y_pred = model.predict(X_scaled)
labels = le.inverse_transform(y_pred)
```

