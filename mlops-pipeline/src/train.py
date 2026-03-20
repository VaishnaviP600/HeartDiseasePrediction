"""
train.py — MLflow-tracked training pipeline for Heart Disease Prediction
Uses Random Forest (as in the published research paper) + other classifiers
with full experiment tracking, model registry, and artifact logging.
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings("ignore")


# ─── Config ──────────────────────────────────────────────────────────────────
DATA_PATH = "data/processed/heart_disease.csv"
MODEL_DIR = "models"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
EXPERIMENT_NAME = "heart-disease-prediction"

MODELS = {
    "random_forest": RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=5,
        random_state=42, n_jobs=-1
    ),
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=7),
    "svm": SVC(probability=True, random_state=42),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, random_state=42
    ),
}


def load_data(path: str):
    """Load and split dataset."""
    df = pd.read_csv(path)
    X = df.drop("target", axis=1)
    y = df["target"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def evaluate(model, X_test, y_test) -> dict:
    """Compute all evaluation metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy":    round(accuracy_score(y_test, y_pred), 4),
        "precision":   round(precision_score(y_test, y_pred), 4),
        "recall":      round(recall_score(y_test, y_pred), 4),        # sensitivity
        "f1_score":    round(f1_score(y_test, y_pred), 4),
        "auc_roc":     round(roc_auc_score(y_test, y_prob), 4),
        "specificity": round(tn / (tn + fp), 4),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def train_and_log(model_name: str, model, X_train, X_test, y_train, y_test,
                  feature_names: list, register: bool = False):
    """Train a model and log everything to MLflow."""

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=model_name):
        # Build pipeline with scaler
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model),
        ])

        # Train
        pipeline.fit(X_train, y_train)

        # Evaluate
        metrics = evaluate(pipeline, X_test, y_test)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")

        # Log parameters
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("features", len(feature_names))
        if hasattr(model, "n_estimators"):
            mlflow.log_param("n_estimators", model.n_estimators)
        if hasattr(model, "max_depth"):
            mlflow.log_param("max_depth", model.max_depth)

        # Log metrics
        for k, v in metrics.items():
            if isinstance(v, float):
                mlflow.log_metric(k, v)
        mlflow.log_metric("cv_mean_accuracy", round(cv_scores.mean(), 4))
        mlflow.log_metric("cv_std_accuracy", round(cv_scores.std(), 4))

        # Log model
        signature = infer_signature(X_train, pipeline.predict(X_train))
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            signature=signature,
            input_example=X_train.iloc[:3],
        )

        # Feature importance (Random Forest only)
        if model_name == "random_forest":
            importances = model.feature_importances_
            feat_imp = dict(zip(feature_names, importances.round(4).tolist()))
            mlflow.log_dict(feat_imp, "feature_importance.json")

        # Save confusion matrix
        cm_data = {
            "true_positive": metrics["tp"], "true_negative": metrics["tn"],
            "false_positive": metrics["fp"], "false_negative": metrics["fn"],
        }
        mlflow.log_dict(cm_data, "confusion_matrix.json")

        # Save model locally
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = f"{MODEL_DIR}/{model_name}.pkl"
        joblib.dump(pipeline, model_path)
        mlflow.log_artifact(model_path)

        # Register best model
        if register:
            run_id = mlflow.active_run().info.run_id
            model_uri = f"runs:/{run_id}/model"
            mv = mlflow.register_model(model_uri, f"heart-disease-{model_name}")
            print(f"  → Registered: {mv.name} v{mv.version}")

        print(f"  ✓ {model_name:25s} | acc={metrics['accuracy']:.4f} | "
              f"auc={metrics['auc_roc']:.4f} | sensitivity={metrics['recall']:.4f} | "
              f"specificity={metrics['specificity']:.4f}")

        return metrics


def main(model_name: str = "all", register: bool = False):
    print("\n═══ Heart Disease MLOps Pipeline — Training ═══\n")

    # Load data
    X_train, X_test, y_train, y_test = load_data(DATA_PATH)
    feature_names = X_train.columns.tolist()
    print(f"Data loaded: {len(X_train)} train / {len(X_test)} test samples")
    print(f"Features: {feature_names}\n")

    models_to_run = MODELS if model_name == "all" else {model_name: MODELS[model_name]}
    results = {}

    for name, model in models_to_run.items():
        results[name] = train_and_log(
            name, model, X_train, X_test, y_train, y_test,
            feature_names, register=(register or name == "random_forest")
        )

    # Best model summary
    best = max(results, key=lambda k: results[k]["auc_roc"])
    print(f"\n🏆 Best model: {best} (AUC-ROC: {results[best]['auc_roc']:.4f})")
    print(f"   Research paper accuracy: 93.3% | This run: {results.get('random_forest', {}).get('accuracy', 'N/A')}")
    print(f"\nView MLflow UI: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all", choices=list(MODELS.keys()) + ["all"])
    parser.add_argument("--register", action="store_true", help="Register model in MLflow registry")
    args = parser.parse_args()
    main(args.model, args.register)
