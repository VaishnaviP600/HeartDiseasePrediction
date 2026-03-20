"""
serve.py — FastAPI production serving endpoint for Heart Disease Prediction
Includes Prometheus metrics, health checks, model versioning, and prediction logging.
"""

import os
import time
import json
import logging
from datetime import datetime
from typing import Optional
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import mlflow
import mlflow.sklearn


# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ─── Prometheus Metrics ───────────────────────────────────────────────────────
PREDICTION_COUNTER = Counter(
    "heart_disease_predictions_total",
    "Total predictions made",
    ["prediction", "model_version"]
)
PREDICTION_LATENCY = Histogram(
    "heart_disease_prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
)
MODEL_ACCURACY_GAUGE = Gauge(
    "heart_disease_model_accuracy",
    "Current model accuracy"
)
REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)
HIGH_RISK_COUNTER = Counter(
    "heart_disease_high_risk_predictions_total",
    "Total high-risk predictions (probability > 0.7)"
)


# ─── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Heart Disease Prediction API",
    description="""
    MLOps-powered prediction API based on the research paper:
    'An Effective Prediction Of Heart Diseases Using Machine Learning Techniques'
    — G. Narayanamma Institute of Technology & Science, Hyderabad

    Uses Random Forest with 93.3% AUC-ROC as reported in the paper.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Model Loading ────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "models/random_forest.pkl")
MLFLOW_MODEL_URI = os.getenv("MLFLOW_MODEL_URI", None)

model = None
model_version = "1.0.0"
model_loaded_at = None

def load_model():
    global model, model_version, model_loaded_at

    if MLFLOW_MODEL_URI:
        try:
            logger.info(f"Loading model from MLflow: {MLFLOW_MODEL_URI}")
            model = mlflow.sklearn.load_model(MLFLOW_MODEL_URI)
            model_version = MLFLOW_MODEL_URI.split("/")[-1]
        except Exception as e:
            logger.warning(f"MLflow load failed: {e}. Falling back to local model.")

    if model is None and os.path.exists(MODEL_PATH):
        logger.info(f"Loading local model: {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        model_version = "local-1.0.0"
        MODEL_ACCURACY_GAUGE.set(0.933)  # from research paper

    if model is None:
        logger.warning("No model found. API will return 503 until model is loaded.")

    model_loaded_at = datetime.now().isoformat()
    logger.info(f"Model ready: version={model_version}")


# ─── Schemas ─────────────────────────────────────────────────────────────────
class PatientFeatures(BaseModel):
    """
    Input features as described in the research paper (Framingham Heart Study dataset).
    """
    age:     float = Field(..., ge=1, le=120, description="Patient age in years")
    sex:     int   = Field(..., ge=0, le=1,   description="0=female, 1=male")
    cp:      int   = Field(..., ge=0, le=3,   description="Chest pain type: 0=typical angina, 1=atypical, 2=non-anginal, 3=asymptomatic")
    restbp:  float = Field(..., ge=60, le=300, description="Resting blood pressure (mmHg)")
    chol:    float = Field(..., ge=50, le=700, description="Serum cholesterol (mg/dl)")
    fbs:     int   = Field(..., ge=0, le=1,   description="Fasting blood sugar: 0=>=120mg/dl, 1=<120mg/dl")
    restecg: int   = Field(..., ge=0, le=2,   description="Resting ECG: 0=normal, 1=ST-T wave abnormality, 2=LV hypertrophy")
    thalach: float = Field(..., ge=50, le=250, description="Maximum heart rate achieved")
    exang:   int   = Field(..., ge=0, le=1,   description="Exercise induced angina: 0=no, 1=yes")
    oldpeak: float = Field(..., ge=0, le=10,  description="ST depression induced by exercise relative to rest")
    slope:   int   = Field(..., ge=0, le=2,   description="Slope of peak exercise ST segment: 0=positive, 1=flat, 2=negative")
    ca:      float = Field(..., ge=0, le=3,   description="Number of major vessels colored by fluoroscopy (0-3)")
    thal:    int   = Field(..., ge=1, le=3,   description="Thalium heart scan: 1=normal, 2=fixed defect, 3=reversible defect")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 52, "sex": 1, "cp": 0, "restbp": 125, "chol": 212,
                "fbs": 0, "restecg": 1, "thalach": 168, "exang": 0,
                "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3
            }
        }


class PredictionResponse(BaseModel):
    prediction:       int
    prediction_label: str
    probability_heart_disease: float
    probability_healthy:       float
    risk_level:   str
    model_version: str
    timestamp:    str
    advice:       str


class BatchPredictionRequest(BaseModel):
    patients: list[PatientFeatures]


# ─── Middleware ───────────────────────────────────────────────────────────────
@app.middleware("http")
async def track_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    REQUESTS_TOTAL.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    return response


# ─── Routes ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    load_model()


@app.get("/", tags=["Info"])
def root():
    return {
        "service": "Heart Disease Prediction API",
        "version": "1.0.0",
        "paper": "An Effective Prediction Of Heart Diseases Using Machine Learning Techniques",
        "institution": "G. Narayanamma Institute of Technology & Science, Hyderabad",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


@app.get("/health", tags=["Health"])
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model_version": model_version,
        "model_loaded_at": model_loaded_at,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/metrics", tags=["Monitoring"])
def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model/info", tags=["Model"])
def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_version": model_version,
        "model_type": type(model.named_steps["model"]).__name__ if hasattr(model, "named_steps") else str(type(model)),
        "loaded_at": model_loaded_at,
        "features": [
            "age", "sex", "cp", "restbp", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ],
        "paper_metrics": {
            "sensitivity": "90.6%",
            "specificity": "82.7%",
            "auc_roc": "93.3%",
        }
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(patient: PatientFeatures):
    """
    Predict heart disease risk for a single patient.
    Returns prediction, probability, risk level, and health advice.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    start = time.time()

    features = pd.DataFrame([patient.dict()])
    feature_order = ["age","sex","cp","restbp","chol","fbs","restecg",
                     "thalach","exang","oldpeak","slope","ca","thal"]
    features = features[feature_order]

    try:
        prediction = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]
        prob_disease = round(float(probabilities[1]), 4)
        prob_healthy = round(float(probabilities[0]), 4)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    latency = time.time() - start
    PREDICTION_LATENCY.observe(latency)

    # Risk level
    if prob_disease >= 0.7:
        risk_level = "HIGH"
        HIGH_RISK_COUNTER.inc()
        advice = ("⚠️ High risk of heart disease detected. "
                  "Please consult a cardiologist immediately. "
                  "Regular monitoring of blood pressure, cholesterol, and heart rate is essential.")
    elif prob_disease >= 0.4:
        risk_level = "MODERATE"
        advice = ("⚡ Moderate risk. Consider lifestyle changes: "
                  "balanced diet, regular exercise, stress management. "
                  "Schedule a check-up with your healthcare provider.")
    else:
        risk_level = "LOW"
        advice = ("✅ Low risk. Maintain your healthy lifestyle! "
                  "Continue regular exercise, balanced diet, and routine health check-ups.")

    label = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
    PREDICTION_COUNTER.labels(prediction=str(prediction), model_version=model_version).inc()

    logger.info(f"Prediction: {label} | prob={prob_disease:.3f} | latency={latency:.3f}s")

    return PredictionResponse(
        prediction=prediction,
        prediction_label=label,
        probability_heart_disease=prob_disease,
        probability_healthy=prob_healthy,
        risk_level=risk_level,
        model_version=model_version,
        timestamp=datetime.now().isoformat(),
        advice=advice,
    )


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(request: BatchPredictionRequest):
    """Batch prediction for multiple patients."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(request.patients) > 100:
        raise HTTPException(status_code=400, detail="Max 100 patients per batch")

    results = []
    for patient in request.patients:
        result = predict(patient)
        results.append(result)

    summary = {
        "total": len(results),
        "high_risk": sum(1 for r in results if r.risk_level == "HIGH"),
        "moderate_risk": sum(1 for r in results if r.risk_level == "MODERATE"),
        "low_risk": sum(1 for r in results if r.risk_level == "LOW"),
    }

    return {"predictions": results, "summary": summary}


@app.post("/model/reload", tags=["Model"])
def reload_model():
    """Hot-reload the model without restarting the server."""
    load_model()
    return {"status": "reloaded", "model_version": model_version, "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=True)
