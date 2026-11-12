"""
Fraud Detection System (FDS) - Scoring Service
Author: Alpha
Description: End-to-end real-time and batch transaction scoring service with
             dynamic feature enrichment, SHAP explanation, and atomic persistence.
"""

import datetime as dt
import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from app.extensions import db
from app.models.alert import Alert
from app.models.merchant import Merchant
from app.models.prediction import ModelVersion, Prediction
from app.models.transaction import Transaction
from app.models.user import User
from app.schemas.prediction import PredictionResponseSchema, ReasonCodeSchema
from app.schemas.transaction import TransactionIngestSchema
from app.services.history_service import HistoryService
from ml.explainers.shap_explainer import FraudSHAPExplainer
from ml.features.pipeline import FraudFeatureEngineeringPipeline
from ml.models.isolation_forest import IsolationForestFraudModel


class ScoringService:
    """Manages model loading, feature transformation, inference, and persistence."""

    _instance: Optional["ScoringService"] = None

    def __init__(self, artifacts_dir: str = "ml/artifacts") -> None:
        self.artifacts_dir = artifacts_dir
        self.pipeline: Optional[FraudFeatureEngineeringPipeline] = None
        self.model: Optional[IsolationForestFraudModel] = None
        self.explainer: Optional[FraudSHAPExplainer] = None
        self.active_version_tag: str = "v1.0.0"
        self.decision_threshold: float = 80.0

        self.load_artifacts()

    @classmethod
    def get_instance(cls, artifacts_dir: str = "ml/artifacts") -> "ScoringService":
        if cls._instance is None:
            cls._instance = cls(artifacts_dir=artifacts_dir)
        return cls._instance

    def load_artifacts(self) -> None:
        """Loads serialized pipeline and trained Isolation Forest model."""
        pipe_path = os.path.join(self.artifacts_dir, "feature_pipeline.joblib")
        model_path = os.path.join(self.artifacts_dir, f"isolation_forest_{self.active_version_tag}.joblib")
        meta_path = os.path.join(self.artifacts_dir, "model_metadata.json")

        if os.path.exists(pipe_path) and os.path.exists(model_path):
            self.pipeline = joblib.load(pipe_path)
            self.model = IsolationForestFraudModel.load(model_path)
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                    self.decision_threshold = float(meta.get("decision_threshold", 80.0))
                    self.model.decision_threshold = self.decision_threshold
            self.explainer = FraudSHAPExplainer(self.model, feature_names=self.pipeline.get_feature_names())
        else:
            # Fallback initialization if artifacts have not been generated yet
            self._initialize_fallback_model()

    def _initialize_fallback_model(self) -> None:
        """Initializes a lightweight model if training artifacts are not yet saved."""
        from ml.data_generator import generate_dataset

        sample_df = generate_dataset(n_transactions=500, fraud_rate=0.02, random_state=42)
        self.pipeline = FraudFeatureEngineeringPipeline(with_scaling=True)
        X = self.pipeline.fit_transform(sample_df)

        self.model = IsolationForestFraudModel(
            contamination=0.02,
            n_estimators=50,
            decision_threshold=75.0,
            random_state=42,
        )
        self.model.fit(X.values)
        self.decision_threshold = 75.0
        self.explainer = FraudSHAPExplainer(self.model, feature_names=self.pipeline.get_feature_names())

    def score_transaction(self, tx_in: TransactionIngestSchema) -> PredictionResponseSchema:
        """Scores a single transaction with sub-50ms latency budget and stores in DB."""
        t_start = time.perf_counter()

        # 1. Parse timestamp
        ts = dt.datetime.fromisoformat(tx_in.timestamp.replace("Z", "+00:00"))
        if ts.tzinfo is not None:
            ts = ts.astimezone(dt.timezone.utc).replace(tzinfo=None)

        # 2. Ensure Customer & Merchant exist
        user = User.query.filter_by(external_id=tx_in.user_external_id).first()
        if not user:
            user = User(
                external_id=tx_in.user_external_id,
                first_name=f"Customer",
                last_name=tx_in.user_external_id,
                home_lat=tx_in.location_lat or 40.7128,
                home_lon=tx_in.location_lon or -74.0060,
            )
            db.session.add(user)
            db.session.flush()

        merchant = Merchant.query.filter_by(external_id=tx_in.merchant_external_id).first()
        if not merchant:
            merchant = Merchant(
                external_id=tx_in.merchant_external_id,
                name=f"Merchant {tx_in.merchant_external_id}",
                category="general_retail",
                risk_rating=1.0,
            )
            db.session.add(merchant)
            db.session.flush()

        # 3. Retrieve Historical Context
        user_history = HistoryService.get_user_history_snapshot(user.id, ts)

        # 4. Feature Extraction & Scaling
        tx_dict = {
            "amount": tx_in.amount,
            "timestamp": tx_in.timestamp,
            "location_lat": tx_in.location_lat,
            "location_lon": tx_in.location_lon,
            "device_id": tx_in.device_id,
            "merchant_id": str(merchant.id),
        }
        scaled_vec, raw_features = self.pipeline.transform_single(tx_dict, user_history)

        # 5. Model Inference & Calibration
        raw_anomaly_score, risk_score, is_flagged = self.model.score_single(scaled_vec)

        # 6. Determine Severity
        if risk_score >= 85.0:
            severity = "CRITICAL"
        elif risk_score >= 70.0:
            severity = "HIGH"
        elif risk_score >= 45.0:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        # 7. Explainability & Reason Codes (TreeSHAP)
        reason_codes: List[ReasonCodeSchema] = []
        if is_flagged or risk_score >= 50.0:
            explanation = self.explainer.explain_transaction(scaled_vec, raw_features, top_k=3)
            for r in explanation["reason_codes"]:
                reason_codes.append(
                    ReasonCodeSchema(
                        code=r["code"],
                        description=r["description"],
                        impact=r["impact"],
                    )
                )

        latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

        # 8. Persist Transaction, Prediction, and Alert
        # Check active ModelVersion record in DB
        mv = ModelVersion.query.filter_by(version_tag=self.active_version_tag).first()
        if not mv:
            mv = ModelVersion(
                version_tag=self.active_version_tag,
                algorithm="IsolationForest",
                decision_threshold=self.decision_threshold,
                artifact_path=os.path.join(self.artifacts_dir, f"isolation_forest_{self.active_version_tag}.joblib"),
                is_active=True,
            )
            db.session.add(mv)
            db.session.flush()

        tx_record = Transaction(
            transaction_uid=tx_in.transaction_uid,
            user_id=user.id,
            merchant_id=merchant.id,
            amount=tx_in.amount,
            currency=tx_in.currency,
            timestamp=ts,
            location_lat=tx_in.location_lat,
            location_lon=tx_in.location_lon,
            device_id=tx_in.device_id,
            ip_address=tx_in.ip_address,
            channel=tx_in.channel,
            status="FLAGGED" if is_flagged else "PROCESSED",
        )
        db.session.add(tx_record)
        db.session.flush()

        pred_record = Prediction(
            transaction_id=tx_record.id,
            model_version_id=mv.id,
            raw_anomaly_score=raw_anomaly_score,
            calibrated_risk_score=risk_score,
            is_flagged=is_flagged,
            latency_ms=latency_ms,
            feature_snapshot=raw_features,
        )
        db.session.add(pred_record)
        db.session.flush()

        alert_uid = None
        if is_flagged:
            alert_uid = f"alt_{uuid.uuid4().hex[:12]}"
            alert_record = Alert(
                alert_uid=alert_uid,
                transaction_id=tx_record.id,
                prediction_id=pred_record.id,
                severity=severity,
                status="OPEN",
                trigger_reasons=[r.model_dump() for r in reason_codes],
            )
            db.session.add(alert_record)

        db.session.commit()

        return PredictionResponseSchema(
            transaction_uid=tx_in.transaction_uid,
            risk_score=risk_score,
            raw_anomaly_score=round(raw_anomaly_score, 4),
            is_flagged=is_flagged,
            severity=severity,
            decision_threshold=self.decision_threshold,
            model_version=self.active_version_tag,
            reason_codes=reason_codes,
            latency_ms=latency_ms,
            alert_uid=alert_uid,
        )

    def score_batch(self, transactions: List[TransactionIngestSchema]) -> Dict[str, Any]:
        """Processes a batch of transactions and returns summary statistics."""
        t_start = time.perf_counter()
        results: List[Dict[str, Any]] = []
        flagged_count = 0

        for tx in transactions:
            pred = self.score_transaction(tx)
            if pred.is_flagged:
                flagged_count += 1
            results.append(pred.model_dump())

        total_latency = round((time.perf_counter() - t_start) * 1000.0, 2)
        return {
            "total_processed": len(transactions),
            "total_flagged": flagged_count,
            "anomaly_rate_percentage": round((flagged_count / max(1, len(transactions))) * 100.0, 2),
            "batch_latency_ms": total_latency,
            "predictions": results,
        }
