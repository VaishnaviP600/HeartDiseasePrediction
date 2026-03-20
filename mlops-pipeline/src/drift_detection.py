"""
drift_detection.py — Automated data drift detection using Evidently AI
Triggers retraining when significant drift is detected in incoming data.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import argparse


DRIFT_REPORT_DIR = "monitoring/drift_reports"
DRIFT_THRESHOLD = 0.15   # retrain if drift score > 15%
REFERENCE_DATA_PATH = "data/processed/heart_disease.csv"
CURRENT_DATA_PATH = "data/drift/current_batch.csv"
RETRAIN_FLAG_PATH = "data/drift/retrain_flag.json"


def generate_drifted_batch(reference_df: pd.DataFrame, drift_factor: float = 0.3,
                            n_samples: int = 200) -> pd.DataFrame:
    """
    Simulate a new data batch with artificial drift for demonstration.
    In production, this would be real incoming data.
    """
    np.random.seed(int(datetime.now().timestamp()) % 1000)
    batch = reference_df.sample(n=n_samples, replace=True).copy()

    # Introduce distribution shift in key features
    batch["age"] = batch["age"] + np.random.normal(drift_factor * 10, 3, n_samples)
    batch["chol"] = batch["chol"] + np.random.normal(drift_factor * 30, 10, n_samples)
    batch["thalach"] = batch["thalach"] - np.random.normal(drift_factor * 15, 5, n_samples)

    # Clip to valid ranges
    batch["age"] = batch["age"].clip(29, 78)
    batch["chol"] = batch["chol"].clip(126, 564)
    batch["thalach"] = batch["thalach"].clip(71, 202)

    return batch


def compute_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """
    Population Stability Index (PSI) — measures feature distribution shift.
    PSI < 0.1: no change | 0.1-0.2: moderate | > 0.2: significant drift
    """
    ref_counts, edges = np.histogram(reference.dropna(), bins=bins)
    cur_counts, _ = np.histogram(current.dropna(), bins=edges)

    ref_pct = (ref_counts + 1e-6) / len(reference)
    cur_pct = (cur_counts + 1e-6) / len(current)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return round(float(psi), 4)


def compute_ks_statistic(reference: pd.Series, current: pd.Series) -> dict:
    """Kolmogorov-Smirnov test for distribution similarity."""
    from scipy import stats
    stat, pvalue = stats.ks_2samp(reference.dropna(), current.dropna())
    return {"statistic": round(float(stat), 4), "pvalue": round(float(pvalue), 4)}


def detect_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
    """
    Run drift detection on all numeric features.
    Returns a report with per-feature drift scores and overall verdict.
    """
    numeric_features = reference_df.select_dtypes(include=[np.number]).columns.tolist()
    if "target" in numeric_features:
        numeric_features.remove("target")

    feature_reports = {}
    drift_detected_count = 0

    for feat in numeric_features:
        if feat not in current_df.columns:
            continue

        psi = compute_psi(reference_df[feat], current_df[feat])
        ks = compute_ks_statistic(reference_df[feat], current_df[feat])

        ref_stats = {
            "mean": round(float(reference_df[feat].mean()), 3),
            "std":  round(float(reference_df[feat].std()), 3),
            "min":  round(float(reference_df[feat].min()), 3),
            "max":  round(float(reference_df[feat].max()), 3),
        }
        cur_stats = {
            "mean": round(float(current_df[feat].mean()), 3),
            "std":  round(float(current_df[feat].std()), 3),
            "min":  round(float(current_df[feat].min()), 3),
            "max":  round(float(current_df[feat].max()), 3),
        }

        mean_shift = abs(cur_stats["mean"] - ref_stats["mean"]) / (ref_stats["std"] + 1e-6)
        drifted = psi > 0.2 or ks["pvalue"] < 0.05

        if drifted:
            drift_detected_count += 1

        feature_reports[feat] = {
            "psi": psi,
            "ks_statistic": ks["statistic"],
            "ks_pvalue": ks["pvalue"],
            "mean_shift_zscore": round(mean_shift, 3),
            "drift_detected": drifted,
            "reference_stats": ref_stats,
            "current_stats": cur_stats,
        }

    overall_drift_score = drift_detected_count / len(numeric_features) if numeric_features else 0
    should_retrain = overall_drift_score > DRIFT_THRESHOLD

    report = {
        "timestamp": datetime.now().isoformat(),
        "reference_samples": len(reference_df),
        "current_samples": len(current_df),
        "features_analyzed": len(numeric_features),
        "features_with_drift": drift_detected_count,
        "overall_drift_score": round(overall_drift_score, 4),
        "drift_threshold": DRIFT_THRESHOLD,
        "should_retrain": should_retrain,
        "verdict": "🔴 DRIFT DETECTED — Retraining triggered" if should_retrain else "🟢 No significant drift",
        "feature_reports": feature_reports,
    }
    return report


def try_evidently_report(reference_df: pd.DataFrame, current_df: pd.DataFrame,
                          output_path: str):
    """Generate HTML report using Evidently AI if installed."""
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset

        report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
        report.run(reference_data=reference_df.drop("target", axis=1, errors="ignore"),
                   current_data=current_df.drop("target", axis=1, errors="ignore"))
        report.save_html(output_path)
        print(f"  Evidently HTML report saved: {output_path}")
    except ImportError:
        print("  Evidently not installed — skipping HTML report (pip install evidently)")
    except Exception as e:
        print(f"  Evidently report failed: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drift-factor", type=float, default=0.3,
                        help="Artificial drift magnitude (0=none, 1=heavy)")
    args = parser.parse_args()

    os.makedirs(DRIFT_REPORT_DIR, exist_ok=True)
    os.makedirs("data/drift", exist_ok=True)

    # Load reference data
    print("Loading reference data...")
    reference_df = pd.read_csv(REFERENCE_DATA_PATH)

    # Load or generate current batch
    if not os.path.exists(CURRENT_DATA_PATH):
        print(f"Generating drifted batch (factor={args.drift_factor})...")
        current_df = generate_drifted_batch(reference_df, args.drift_factor)
        current_df.to_csv(CURRENT_DATA_PATH, index=False)
    else:
        current_df = pd.read_csv(CURRENT_DATA_PATH)

    print(f"Reference: {len(reference_df)} samples | Current: {len(current_df)} samples\n")

    # Run drift detection
    print("Running drift detection...")
    report = detect_drift(reference_df, current_df)

    # Save JSON report
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = f"{DRIFT_REPORT_DIR}/drift_report_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    # Try Evidently HTML report
    html_path = f"{DRIFT_REPORT_DIR}/evidently_report_{ts}.html"
    try_evidently_report(reference_df, current_df, html_path)

    # Print summary
    print(f"\n{'═'*50}")
    print(f"  Drift Score: {report['overall_drift_score']:.1%}")
    print(f"  Features drifted: {report['features_with_drift']}/{report['features_analyzed']}")
    print(f"  Verdict: {report['verdict']}")
    print(f"{'═'*50}")

    # Write retrain flag
    retrain_flag = {
        "should_retrain": report["should_retrain"],
        "drift_score": report["overall_drift_score"],
        "timestamp": report["timestamp"],
        "report_path": json_path,
    }
    with open(RETRAIN_FLAG_PATH, "w") as f:
        json.dump(retrain_flag, f, indent=2)

    print(f"\n✓ Drift report: {json_path}")
    print(f"✓ Retrain flag: {RETRAIN_FLAG_PATH}")

    if report["should_retrain"]:
        print("\n⚠️  Retraining recommended! Run: python src/train.py --register")
        exit(1)  # Non-zero exit triggers CI/CD retraining pipeline
    else:
        print("\n✅ Model is stable — no retraining needed")
        exit(0)


if __name__ == "__main__":
    main()
