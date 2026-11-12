"""
Fraud Detection System (FDS) - End-to-End Model Training Pipeline
Author: Alpha
Description: Loads dataset, executes time-based split, trains Isolation Forest and
             baselines, optimizes cost-based decision threshold, runs SHAP explanations,
             logs to MLflow, and exports versioned artifacts.
"""

import json
import os
import sys
from typing import Any, Dict

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd

from ml.data_loader import DataLoader, time_based_train_test_split
from ml.drift.detector import DataDriftMonitor
from ml.evaluation.cost_optimizer import CostBasedThresholdOptimizer
from ml.evaluation.metrics import compute_fraud_metrics, print_metrics_report
from ml.explainers.shap_explainer import FraudSHAPExplainer
from ml.features.pipeline import FraudFeatureEngineeringPipeline
from ml.models.baselines import LOFFraudModel, OneClassSVMFraudModel
from ml.models.ensemble import HybridAnomalyEnsemble
from ml.models.isolation_forest import IsolationForestFraudModel

# Optional MLflow integration (enabled when MLflow tracking server is configured)
ENABLE_MLFLOW = os.getenv("ENABLE_MLFLOW", "0").lower() in ("1", "true")
MLFLOW_AVAILABLE = False
if ENABLE_MLFLOW:
    try:
        import mlflow
        MLFLOW_AVAILABLE = True
    except Exception:
        MLFLOW_AVAILABLE = False


def train_and_evaluate(
    data_path: str = "data/raw/transactions.csv",
    artifacts_dir: str = "ml/artifacts",
    model_version: str = "v1.0.0",
    contamination: float = 0.005,
    n_estimators: int = 150,
) -> Dict[str, Any]:
    """Orchestrates end-to-end training and evaluation workflow."""
    os.makedirs(artifacts_dir, exist_ok=True)

    print(f"\n[1/6] Loading transactional dataset from: {data_path}...")
    loader = DataLoader(synthetic_count=10000, fraud_rate=0.006)
    df = loader.load(preferred_filename=os.path.basename(data_path))
    print(f"      Total transactions: {len(df):,} | Fraud instances: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.2f}%)")

    print("\n[2/6] Executing time-based train/test split (80% train / 20% test)...")
    train_df, test_df = time_based_train_test_split(df, test_size=0.20)
    print(f"      Train window: {train_df['timestamp'].min()} -> {train_df['timestamp'].max()} ({len(train_df):,} rows)")
    print(f"      Test window:  {test_df['timestamp'].min()} -> {test_df['timestamp'].max()} ({len(test_df):,} rows)")

    print("\n[3/6] Fitting Feature Engineering Pipeline (Zero Leakage)...")
    feature_pipeline = FraudFeatureEngineeringPipeline(with_scaling=True)
    X_train_df = feature_pipeline.fit_transform(train_df)
    X_test_df = feature_pipeline.transform(test_df)
    feature_names = feature_pipeline.get_feature_names()
    print(f"      Engineered {len(feature_names)} features: {feature_names}")

    y_train = train_df["is_fraud"].values
    y_test = test_df["is_fraud"].values

    print(f"\n[4/6] Training Primary Isolation Forest (contamination={contamination}, n_estimators={n_estimators})...")
    if_model = IsolationForestFraudModel(
        contamination=contamination,
        n_estimators=n_estimators,
        max_samples=256,
        random_state=42,
        decision_threshold=65.0,
    )
    if_model.fit(X_train_df.values)

    # Initial test risk scores
    test_scores_if = if_model.predict_risk_scores(X_test_df.values)

    print("\n[5/6] Tuning Decision Threshold via Financial Cost Optimization...")
    optimizer = CostBasedThresholdOptimizer(cost_fn=450.0, cost_fp=15.0, cost_tp=2.0)
    opt_result = optimizer.optimize(y_test, test_scores_if, search_range=(35.0, 85.0), step=0.5)
    best_threshold = opt_result["optimal_threshold"]
    if_model.decision_threshold = best_threshold

    print(f"      Optimal Decision Threshold:  {best_threshold:.2f}")
    print(f"      Estimated Net Cost Savings:  ${opt_result['net_cost_savings']:,.2f} ({opt_result['roi_percentage']:.1f}% ROI)")

    # Evaluate Primary Model
    metrics_if = compute_fraud_metrics(y_test, test_scores_if, threshold=best_threshold)
    print_metrics_report(metrics_if, model_name="Isolation Forest (Tuned)")

    print("--- Training Comparison Baselines ---")
    # 1. LOF
    print("Fitting Local Outlier Factor (LOF)...")
    lof_model = LOFFraudModel(n_neighbors=25, contamination=contamination, decision_threshold=best_threshold)
    # Fit on sample for performance if dataset is large
    fit_sample = X_train_df.values[:min(3000, len(X_train_df))]
    lof_model.fit(fit_sample)
    test_scores_lof = lof_model.predict_risk_scores(X_test_df.values)
    metrics_lof = compute_fraud_metrics(y_test, test_scores_lof, threshold=best_threshold)
    print(f"LOF PR-AUC: {metrics_lof['pr_auc']:.4f} | Recall: {metrics_lof['recall']:.4f} | F1: {metrics_lof['f1']:.4f}")

    # 2. One-Class SVM
    print("Fitting One-Class SVM...")
    ocsvm_model = OneClassSVMFraudModel(nu=0.01, decision_threshold=best_threshold)
    ocsvm_model.fit(fit_sample)
    test_scores_svm = ocsvm_model.predict_risk_scores(X_test_df.values)
    metrics_svm = compute_fraud_metrics(y_test, test_scores_svm, threshold=best_threshold)
    print(f"OC-SVM PR-AUC: {metrics_svm['pr_auc']:.4f} | Recall: {metrics_svm['recall']:.4f} | F1: {metrics_svm['f1']:.4f}")

    # 3. Hybrid Ensemble
    print("Evaluating Hybrid Anomaly Ensemble...")
    ensemble = HybridAnomalyEnsemble(decision_threshold=best_threshold)
    ensemble.if_model = if_model
    ensemble.lof_model = lof_model
    ensemble.ocsvm_model = ocsvm_model
    ensemble.is_fitted_ = True
    test_scores_ens = ensemble.predict_risk_scores(X_test_df.values)
    metrics_ens = compute_fraud_metrics(y_test, test_scores_ens, threshold=best_threshold)
    print(f"Ensemble PR-AUC: {metrics_ens['pr_auc']:.4f} | Recall: {metrics_ens['recall']:.4f} | F1: {metrics_ens['f1']:.4f}")

    print("\n[6/6] Initializing SHAP Explainer & Testing Reason Code Synthesis...")
    explainer = FraudSHAPExplainer(if_model, feature_names=feature_names)

    # Test explanation on the highest risk test transaction
    highest_idx = int(np.argmax(test_scores_if))
    sample_row = test_df.iloc[highest_idx].to_dict()
    sample_vec = X_test_df.iloc[highest_idx].values

    explanation = explainer.explain_transaction(sample_vec, sample_row, top_k=3)
    print(f"\n--- Explanation for Flagged Transaction ({sample_row['transaction_id']}) ---")
    print(f"Risk Score: {test_scores_if[highest_idx]:.2f} / 100")
    for r in explanation["reason_codes"]:
        print(f"  * [{r['code']}]: {r['description']} (Impact: {r['impact']:+.4f})")

    # Serialize artifacts
    pipe_path = os.path.join(artifacts_dir, "feature_pipeline.joblib")
    model_path = os.path.join(artifacts_dir, f"isolation_forest_{model_version}.joblib")
    meta_path = os.path.join(artifacts_dir, "model_metadata.json")

    joblib.dump(feature_pipeline, pipe_path)
    if_model.save(model_path)

    metadata = {
        "model_version": model_version,
        "algorithm": "IsolationForest",
        "contamination": contamination,
        "n_estimators": n_estimators,
        "decision_threshold": best_threshold,
        "feature_names": feature_names,
        "metrics": metrics_if,
        "cost_optimization": opt_result,
        "baseline_comparison": {
            "lof": metrics_lof,
            "ocsvm": metrics_svm,
            "ensemble": metrics_ens,
        },
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[✓] Exported Artifacts:")
    print(f"    - Pipeline: {pipe_path}")
    print(f"    - Model:    {model_path}")
    print(f"    - Metadata: {meta_path}")

    # Log to MLflow if available
    if MLFLOW_AVAILABLE:
        try:
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment("Fraud_Detection_IsolationForest")
            with mlflow.start_run(run_name=f"run_{model_version}"):
                mlflow.log_params({
                    "model_version": model_version,
                    "contamination": contamination,
                    "n_estimators": n_estimators,
                    "decision_threshold": best_threshold,
                })
                mlflow.log_metrics({
                    "pr_auc": metrics_if["pr_auc"],
                    "roc_auc": metrics_if["roc_auc"],
                    "precision": metrics_if["precision"],
                    "recall": metrics_if["recall"],
                    "f1": metrics_if["f1"],
                    "net_cost_savings": opt_result["net_cost_savings"],
                })
                mlflow.log_artifact(meta_path)
                print(f"[✓] Logged experiment run to MLflow: {tracking_uri}")
        except Exception as e:
            print(f"[!] Note: MLflow logging skipped ({e})")

    return metadata


if __name__ == "__main__":
    train_and_evaluate(data_path="data/raw/transactions.csv")
