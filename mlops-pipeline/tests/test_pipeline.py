"""
tests/test_pipeline.py — Unit and integration tests for the MLOps pipeline
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─── Preprocess Tests ─────────────────────────────────────────────────────────
class TestPreprocess:
    def test_generate_sample_data(self):
        from src.preprocess import generate_sample_data
        df = generate_sample_data(n_samples=100)
        assert len(df) == 100
        assert "target" in df.columns
        assert df["target"].isin([0, 1]).all()

    def test_required_features(self):
        from src.preprocess import generate_sample_data
        df = generate_sample_data(100)
        required = ["age","sex","cp","restbp","chol","fbs","restecg",
                    "thalach","exang","oldpeak","slope","ca","thal","target"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_preprocess_handles_missing(self):
        from src.preprocess import generate_sample_data, preprocess
        df = generate_sample_data(100)
        df.loc[0, "age"] = np.nan
        df.loc[1, "chol"] = np.nan
        df_processed = preprocess(df)
        assert df_processed.isnull().sum().sum() == 0

    def test_preprocess_no_duplicates(self):
        from src.preprocess import generate_sample_data, preprocess
        df = generate_sample_data(100)
        df = pd.concat([df, df.iloc[:10]])  # add duplicates
        df_processed = preprocess(df)
        assert not df_processed.duplicated().any()

    def test_feature_ranges(self):
        from src.preprocess import generate_sample_data
        df = generate_sample_data(200)
        assert df["age"].between(1, 120).all()
        assert df["sex"].isin([0, 1]).all()
        assert df["cp"].isin([0, 1, 2, 3]).all()


# ─── Drift Detection Tests ────────────────────────────────────────────────────
class TestDriftDetection:
    def test_psi_no_drift(self):
        from src.drift_detection import compute_psi
        data = pd.Series(np.random.normal(0, 1, 500))
        psi = compute_psi(data, data)
        assert psi < 0.1  # same distribution — no drift

    def test_psi_high_drift(self):
        from src.drift_detection import compute_psi
        ref = pd.Series(np.random.normal(0, 1, 500))
        cur = pd.Series(np.random.normal(5, 1, 500))  # very different
        psi = compute_psi(ref, cur)
        assert psi > 0.2  # should detect drift

    def test_ks_statistic(self):
        from src.drift_detection import compute_ks_statistic
        ref = pd.Series(np.random.normal(0, 1, 300))
        cur = pd.Series(np.random.normal(0, 1, 300))
        result = compute_ks_statistic(ref, cur)
        assert "statistic" in result
        assert "pvalue" in result
        assert 0 <= result["statistic"] <= 1

    def test_detect_drift_structure(self):
        from src.preprocess import generate_sample_data
        from src.drift_detection import detect_drift
        ref = generate_sample_data(200)
        cur = generate_sample_data(100, seed=99)
        report = detect_drift(ref, cur)
        assert "overall_drift_score" in report
        assert "should_retrain" in report
        assert "feature_reports" in report
        assert isinstance(report["should_retrain"], bool)


# ─── Training Tests ───────────────────────────────────────────────────────────
class TestTraining:
    @pytest.fixture
    def sample_data(self):
        from src.preprocess import generate_sample_data
        df = generate_sample_data(300)
        X = df.drop("target", axis=1)
        y = df["target"]
        from sklearn.model_selection import train_test_split
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def test_evaluate_metrics(self, sample_data):
        from src.train import evaluate
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X_train, X_test, y_train, y_test = sample_data
        pipeline = Pipeline([("scaler", StandardScaler()),
                              ("model", RandomForestClassifier(n_estimators=10, random_state=42))])
        pipeline.fit(X_train, y_train)

        metrics = evaluate(pipeline, X_test, y_test)
        assert "accuracy" in metrics
        assert "auc_roc" in metrics
        assert "recall" in metrics
        assert "specificity" in metrics
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["auc_roc"] <= 1

    def test_model_predicts(self, sample_data):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X_train, X_test, y_train, y_test = sample_data
        pipeline = Pipeline([("scaler", StandardScaler()),
                              ("model", RandomForestClassifier(n_estimators=10, random_state=42))])
        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)
        assert len(preds) == len(X_test)
        assert set(preds).issubset({0, 1})


# ─── API Tests ────────────────────────────────────────────────────────────────
class TestAPI:
    SAMPLE_PATIENT = {
        "age": 52, "sex": 1, "cp": 0, "restbp": 125, "chol": 212,
        "fbs": 0, "restecg": 1, "thalach": 168, "exang": 0,
        "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3
    }

    def test_patient_features_validation(self):
        from src.serve import PatientFeatures
        patient = PatientFeatures(**self.SAMPLE_PATIENT)
        assert patient.age == 52
        assert patient.sex == 1

    def test_invalid_age_rejected(self):
        from src.serve import PatientFeatures
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            PatientFeatures(**{**self.SAMPLE_PATIENT, "age": 200})

    def test_invalid_sex_rejected(self):
        from src.serve import PatientFeatures
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            PatientFeatures(**{**self.SAMPLE_PATIENT, "sex": 5})
