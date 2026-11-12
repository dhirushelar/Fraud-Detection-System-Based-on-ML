"""
Fraud Detection System (FDS) - Retraining Service
Author: Alpha
Description: Automates model retraining triggered by admin or drift alert.
"""

import datetime as dt
import json
import os
from typing import Any, Dict

import joblib
import pandas as pd

from app.extensions import db
from app.models.alert import AnalystFeedback
from app.models.prediction import ModelVersion
from app.models.transaction import Transaction
from ml.data_loader import DataLoader
from ml.evaluation.cost_optimizer import CostBasedThresholdOptimizer
from ml.evaluation.metrics import compute_fraud_metrics
from ml.features.pipeline import FraudFeatureEngineeringPipeline
from ml.models.isolation_forest import IsolationForestFraudModel


class RetrainService:
    """Retrains the model incorporating latest transactions and analyst ground truth."""

    @staticmethod
    def retrain_model(artifacts_dir: str = "ml/artifacts") -> Dict[str, Any]:
        # 1. Fetch historical transactions from DB
        transactions = Transaction.query.order_by(Transaction.timestamp.asc()).all()

        if len(transactions) < 200:
            # Fall back to file/generator if DB has few rows
            loader = DataLoader(synthetic_count=5000, fraud_rate=0.01)
            df = loader.load()
        else:
            records = []
            for t in transactions:
                records.append({
                    "transaction_id": t.transaction_uid,
                    "user_id": str(t.user_id),
                    "merchant_id": str(t.merchant_id),
                    "merchant_category": "general_retail",
                    "amount": float(t.amount),
                    "currency": t.currency,
                    "timestamp": t.timestamp.isoformat(),
                    "location_lat": t.location_lat or 40.7128,
                    "location_lon": t.location_lon or -74.0060,
                    "device_id": t.device_id,
                    "ip_address": t.ip_address or "127.0.0.1",
                    "channel": t.channel,
                    "is_fraud": 1 if t.is_ground_truth_fraud else 0,
                    "fraud_type": "analyst_label" if t.is_ground_truth_fraud else "none",
                })
            df = pd.DataFrame(records)

        # 2. Retrain Pipeline & Isolation Forest
        pipeline = FraudFeatureEngineeringPipeline(with_scaling=True)
        X = pipeline.fit_transform(df)
        y = df["is_fraud"].values

        # Increment version tag
        new_tag = f"v{dt.datetime.utcnow().strftime('%Y%m%d%H%M')}"
        model = IsolationForestFraudModel(contamination=0.01, n_estimators=100, random_state=42)
        model.fit(X.values)

        # Optimize Threshold
        scores = model.predict_risk_scores(X.values)
        optimizer = CostBasedThresholdOptimizer(cost_fn=450.0, cost_fp=15.0)
        opt_res = optimizer.optimize(y, scores)
        model.decision_threshold = opt_res["optimal_threshold"]

        metrics = compute_fraud_metrics(y, scores, threshold=model.decision_threshold)

        # 3. Export Artifacts
        os.makedirs(artifacts_dir, exist_ok=True)
        model_path = os.path.join(artifacts_dir, f"isolation_forest_{new_tag}.joblib")
        pipe_path = os.path.join(artifacts_dir, "feature_pipeline.joblib")
        model.save(model_path)
        joblib.dump(pipeline, pipe_path)

        # 4. Update Database Model Versions (deactivate old, activate new)
        ModelVersion.query.update({ModelVersion.is_active: False})

        new_mv = ModelVersion(
            version_tag=new_tag,
            algorithm="IsolationForest",
            parameters={"contamination": 0.01, "n_estimators": 100},
            metrics=metrics,
            decision_threshold=model.decision_threshold,
            artifact_path=model_path,
            is_active=True,
            trained_at=dt.datetime.utcnow(),
        )
        db.session.add(new_mv)
        db.session.commit()

        return {
            "status": "SUCCESS",
            "new_model_version": new_tag,
            "decision_threshold": model.decision_threshold,
            "metrics": metrics,
            "cost_optimization": opt_res,
        }
