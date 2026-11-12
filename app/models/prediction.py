"""
Fraud Detection System (FDS) - Prediction & Model Version Models
Author: Alpha
"""

import datetime as dt
from app.extensions import db


class ModelVersion(db.Model):
    """Metadata registry of trained and deployed anomaly detection models."""
    __tablename__ = "model_versions"

    id = db.Column(db.Integer, primary_key=True)
    version_tag = db.Column(db.String(50), unique=True, nullable=False, index=True)
    algorithm = db.Column(db.String(50), nullable=False)
    parameters = db.Column(db.JSON, nullable=False, default={})
    metrics = db.Column(db.JSON, nullable=False, default={})
    decision_threshold = db.Column(db.Float, nullable=False, default=65.0)
    artifact_path = db.Column(db.String(255), nullable=False)
    is_active = db.Column(db.Boolean, nullable=False, default=False, index=True)
    trained_at = db.Column(db.DateTime, nullable=False, default=dt.datetime.utcnow)
    created_at = db.Column(db.DateTime, nullable=False, default=dt.datetime.utcnow)

    predictions = db.relationship("Prediction", backref="model_version", lazy="dynamic")
    drift_metrics = db.relationship("DriftMetric", backref="model_version", lazy="dynamic")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "version_tag": self.version_tag,
            "algorithm": self.algorithm,
            "decision_threshold": self.decision_threshold,
            "is_active": self.is_active,
            "metrics": self.metrics,
            "parameters": self.parameters,
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
        }


class Prediction(db.Model):
    """Real-time and batch anomaly scoring inference records."""
    __tablename__ = "predictions"

    id = db.Column(
        db.BigInteger().with_variant(db.Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    transaction_id = db.Column(
        db.BigInteger,
        db.ForeignKey("transactions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    model_version_id = db.Column(
        db.Integer,
        db.ForeignKey("model_versions.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    raw_anomaly_score = db.Column(db.Float, nullable=False)
    calibrated_risk_score = db.Column(db.Float, nullable=False, index=True)
    is_flagged = db.Column(db.Boolean, nullable=False, default=False, index=True)
    latency_ms = db.Column(db.Float, nullable=False)
    feature_snapshot = db.Column(db.JSON, nullable=False, default={})
    created_at = db.Column(db.DateTime, nullable=False, default=dt.datetime.utcnow)

    alert = db.relationship("Alert", backref="prediction", uselist=False, cascade="all, delete-orphan")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "transaction_id": self.transaction_id,
            "model_version_id": self.model_version_id,
            "raw_anomaly_score": self.raw_anomaly_score,
            "calibrated_risk_score": self.calibrated_risk_score,
            "is_flagged": self.is_flagged,
            "latency_ms": self.latency_ms,
            "feature_snapshot": self.feature_snapshot,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
