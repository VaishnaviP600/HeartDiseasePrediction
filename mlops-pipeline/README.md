# 🫀 MLOps Pipeline — Heart Disease Prediction

> **Wrapping published research into a production MLOps system**
> Based on: *"An Effective Prediction Of Heart Diseases Using Machine Learning Techniques"*
> — Pujala Vaishnavi et al., G. Narayanamma Institute of Technology & Science, Hyderabad

[![CI/CD](https://github.com/YOUR_USERNAME/mlops-heart-disease/actions/workflows/mlops_pipeline.yml/badge.svg)](https://github.com/YOUR_USERNAME/mlops-heart-disease/actions)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://python.org)
[![MLflow](https://img.shields.io/badge/MLflow-tracked-orange.svg)](https://mlflow.org)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MLOPS PIPELINE                               │
│                                                                 │
│  Raw Data ──► DVC Versioning ──► Preprocessing                 │
│                                        │                        │
│                               Evidently Drift Detection         │
│                                        │                        │
│                            [Drift?] ──► Retrain                 │
│                                        │                        │
│                               MLflow Experiment Tracking        │
│                                   ├── Parameters                │
│                                   ├── Metrics                   │
│                                   └── Model Registry            │
│                                        │                        │
│                               FastAPI Serving                   │
│                                   ├── /predict                  │
│                                   ├── /predict/batch            │
│                                   ├── /health                   │
│                                   └── /metrics (Prometheus)     │
│                                        │                        │
│                          Prometheus ──► Grafana Dashboard        │
│                                                                 │
│  GitHub Actions CI/CD ──► Docker ──► Production Deploy          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/mlops-heart-disease.git
cd mlops-heart-disease
pip install -r requirements.txt
```

### 2. Run Full Pipeline (Local)

```bash
# Step 1: Generate & preprocess data
python src/preprocess.py --generate --samples 1025

# Step 2: Train all models (Random Forest, LR, SVM, KNN, GBM)
python src/train.py --model all --register

# Step 3: View MLflow experiment results
mlflow ui
# Open: http://localhost:5000

# Step 4: Run drift detection
python src/drift_detection.py

# Step 5: Serve the API
uvicorn src.serve:app --reload --port 8000
# Open: http://localhost:8000/docs
```

### 3. Run with Docker Compose (Full Stack)

```bash
docker-compose up --build
```

| Service | URL |
|---------|-----|
| **Prediction API** | http://localhost:8000/docs |
| **MLflow UI** | http://localhost:5000 |
| **Prometheus** | http://localhost:9090 |
| **Grafana** | http://localhost:3000 (admin/admin123) |

### 4. Run Tests

```bash
pytest tests/ -v --cov=src
```

### 5. DVC Pipeline (Data Versioning)

```bash
pip install dvc
dvc init
dvc repro          # runs full pipeline in order
dvc dag            # visualize pipeline DAG
dvc params diff    # compare parameter changes
```

---

## 📊 Research Paper Results vs This Pipeline

| Metric | Paper (Random Forest) | This Pipeline |
|--------|----------------------|---------------|
| Sensitivity (Recall) | 90.6% | ~90%+ |
| Specificity | 82.7% | ~83%+ |
| AUC-ROC | 93.3% | ~93%+ |
| Dataset | Framingham Heart Study | Synthetic Framingham-style |

---

## 🔑 Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Experiment Tracking** | MLflow | Log params, metrics, models |
| **Data Versioning** | DVC | Version control for datasets |
| **Drift Detection** | Evidently AI + PSI/KS tests | Auto-detect distribution shift |
| **Model Serving** | FastAPI | REST API with Pydantic validation |
| **Monitoring** | Prometheus + Grafana | Real-time metrics dashboards |
| **Containerization** | Docker + Docker Compose | Reproducible deployments |
| **CI/CD** | GitHub Actions | Automated train/test/deploy |
| **ML Framework** | Scikit-learn | Random Forest + 4 other models |

---

## 🌐 API Usage

### Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 52, "sex": 1, "cp": 0, "restbp": 125, "chol": 212,
    "fbs": 0, "restecg": 1, "thalach": 168, "exang": 0,
    "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3
  }'
```

### Python Client

```python
import requests

patient = {
    "age": 63, "sex": 1, "cp": 3, "restbp": 145, "chol": 233,
    "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
}

r = requests.post("http://localhost:8000/predict", json=patient)
result = r.json()
print(f"Prediction: {result['prediction_label']}")
print(f"Risk Level: {result['risk_level']}")
print(f"Probability: {result['probability_heart_disease']:.1%}")
print(f"Advice: {result['advice']}")
```

---

## 📁 Project Structure

```
mlops-heart-disease/
├── src/
│   ├── preprocess.py        # Data ingestion & preprocessing
│   ├── train.py             # MLflow-tracked multi-model training
│   ├── drift_detection.py   # Evidently AI + PSI/KS drift detection
│   └── serve.py             # FastAPI with Prometheus metrics
├── tests/
│   └── test_pipeline.py     # Unit & integration tests
├── monitoring/
│   └── prometheus.yml       # Prometheus scrape config
├── data/
│   ├── raw/                 # Raw data (DVC tracked)
│   ├── processed/           # Preprocessed data (DVC tracked)
│   └── drift/               # Drift detection batches
├── models/                  # Trained model artifacts
├── .github/
│   └── workflows/
│       └── mlops_pipeline.yml  # CI/CD pipeline
├── dvc.yaml                 # DVC pipeline stages
├── params.yaml              # Hyperparameters (DVC tracked)
├── docker-compose.yml       # Full stack deployment
├── Dockerfile               # API container
└── requirements.txt
```

---

## 🔄 Automated Retraining Flow

```
Daily Cron (GitHub Actions)
        │
        ▼
Drift Detection runs
        │
   PSI > 0.2 OR KS p-value < 0.05?
        │
   YES ─► Trigger training job ─► Register new model ─► Deploy
   NO  ─► Log "stable" ─► Continue monitoring
```

---

