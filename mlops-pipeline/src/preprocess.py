"""
preprocess.py — Data ingestion and preprocessing pipeline
Handles the Framingham Heart Study / UCI Heart Disease dataset
as described in the published research paper.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import argparse


RAW_DATA_PATH = "data/raw/heart_disease.csv"
PROCESSED_DATA_PATH = "data/processed/heart_disease.csv"


def generate_sample_data(n_samples: int = 1025, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic data matching the Framingham/UCI Heart Disease dataset
    features described in the research paper. Used when real data is not available.
    """
    np.random.seed(seed)
    n = n_samples

    df = pd.DataFrame({
        "age":     np.random.randint(29, 78, n).astype(float),
        "sex":     np.random.randint(0, 2, n),           # 0=female, 1=male
        "cp":      np.random.randint(0, 4, n),            # chest pain type
        "restbp":  np.random.randint(94, 200, n).astype(float),  # resting BP
        "chol":    np.random.randint(126, 564, n).astype(float),  # cholesterol
        "fbs":     np.random.randint(0, 2, n),            # fasting blood sugar
        "restecg": np.random.randint(0, 3, n),            # resting ECG
        "thalach": np.random.randint(71, 202, n).astype(float),  # max heart rate
        "exang":   np.random.randint(0, 2, n),            # exercise induced angina
        "oldpeak": np.round(np.random.uniform(0, 6.2, n), 1),   # ST depression
        "slope":   np.random.randint(0, 3, n),            # slope of peak exercise
        "ca":      np.random.randint(0, 4, n).astype(float),     # major vessels
        "thal":    np.random.randint(1, 4, n),            # thalium heart scan
    })

    # Generate target with realistic correlation to features
    risk_score = (
        (df["age"] > 55).astype(int) * 0.3 +
        (df["sex"] == 1).astype(int) * 0.2 +
        (df["cp"] == 3).astype(int) * 0.4 +
        (df["thalach"] < 130).astype(int) * 0.25 +
        (df["exang"] == 1).astype(int) * 0.3 +
        (df["oldpeak"] > 2).astype(int) * 0.25 +
        (df["ca"] > 0).astype(int) * 0.3 +
        np.random.normal(0, 0.1, n)
    )
    df["target"] = (risk_score > 0.7).astype(int)

    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
    - Handle missing values
    - Encode categorical variables
    - Remove duplicates
    - Validate ranges
    """
    print(f"  Raw shape: {df.shape}")
    print(f"  Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    # Drop duplicates
    df = df.drop_duplicates()
    print(f"  After dedup: {df.shape}")

    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Encode any string categoricals
    le = LabelEncoder()
    for col in categorical_cols:
        if col != "target":
            df[col] = le.fit_transform(df[col])

    # Clip outliers to valid ranges
    range_map = {
        "age": (1, 120), "restbp": (60, 300),
        "chol": (50, 700), "thalach": (50, 250),
        "oldpeak": (0, 10),
    }
    for col, (lo, hi) in range_map.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)

    print(f"  Final shape: {df.shape}")
    print(f"  Target distribution:\n{df['target'].value_counts()}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="Generate synthetic data")
    parser.add_argument("--samples", type=int, default=1025)
    args = parser.parse_args()

    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    if args.generate or not os.path.exists(RAW_DATA_PATH):
        print("Generating synthetic Framingham-style dataset...")
        df = generate_sample_data(args.samples)
        df.to_csv(RAW_DATA_PATH, index=False)
        print(f"  Saved raw data: {RAW_DATA_PATH}")
    else:
        print(f"Loading raw data from {RAW_DATA_PATH}...")
        df = pd.read_csv(RAW_DATA_PATH)

    print("\nPreprocessing...")
    df_processed = preprocess(df)
    df_processed.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"\n✓ Processed data saved: {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    main()
