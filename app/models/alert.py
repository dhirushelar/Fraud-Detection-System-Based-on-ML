"""
Fraud Detection System (FDS) - Alert & Feedback Models
Author: Alpha
"""

import datetime as dt
from app.extensions import db


class Alert(db.Model):
    """Analyst queue items for suspicious, high-risk transactions."""
    __tablename__ = "alerts"

    id = db.Column(
        db.BigInteger().with_variant(db.Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    alert_uid = db.Column(db.String(64), unique=True, nullable=False, index=True)
    transaction_id = db.Column(
        db.BigInteger,
        db.ForeignKey("transactions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    prediction_id = db.Column(
        db.BigInteger,
        db.ForeignKey("predictions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    severity = db.Column(db.String(20), nullable=False, default="HIGH", index=True)  # LOW, MEDIUM, HIGH, CRITICAL
    status = db.Column(db.String(30), nullable=False, default="OPEN", index=True)    # OPEN, INVESTIGATING, RESOLVED_FRAUD, RESOLVED_FALSE_POSITIVE
    trigger_reasons = db.Column(db.JSON, nullable=False, default=[])
    assigned_to_user_id = db.Column(
        db.Integer,
        db.ForeignKey("auth_users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    created_at = db.Column(db.DateTime, nullable=False, default=dt.datetime.utcnow, index=True)
    updated_at = db.Column(
        db.DateTime,
        nullable=False,
        default=dt.datetime.utcnow,
        onupdate=dt.datetime.utcnow,
    )

    feedbacks = db.relationship("AnalystFeedback", backref="alert", cascade="all, delete-orphan", lazy="dynamic")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "alert_uid": self.alert_uid,
            "transaction_id": self.transaction_id,
            "prediction_id": self.prediction_id,
            "severity": self.severity,
            "status": self.status,
            "trigger_reasons": self.trigger_reasons,
            "assigned_to_user_id": self.assigned_to_user_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class AnalystFeedback(db.Model):
    """Ground truth reviews provided by fraud analysts for retraining."""
    __tablename__ = "analyst_feedback"

    id = db.Column(
        db.BigInteger().with_variant(db.Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    alert_id = db.Column(
        db.BigInteger,
        db.ForeignKey("alerts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    transaction_id = db.Column(
        db.BigInteger,
        db.ForeignKey("transactions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    analyst_id = db.Column(
        db.Integer,
        db.ForeignKey("auth_users.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    feedback_verdict = db.Column(
        db.String(30),
        nullable=False,
        index=True,
    )  # TRUE_FRAUD, FALSE_POSITIVE, SUSPICIOUS_UNCONFIRMED
    reason_tags = db.Column(db.JSON, nullable=False, default=[])
    notes = db.Column(db.Text, nullable=True)
    reviewed_at = db.Column(db.DateTime, nullable=False, default=dt.datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "alert_id": self.alert_id,
            "transaction_id": self.transaction_id,
            "analyst_id": self.analyst_id,
            "feedback_verdict": self.feedback_verdict,
            "reason_tags": self.reason_tags,
            "notes": self.notes,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
        }


class DriftMetric(db.Model):
    """Covariate drift records."""
    __tablename__ = "drift_metrics"

    id = db.Column(db.Integer, primary_key=True)
    model_version_id = db.Column(
        db.Integer,
        db.ForeignKey("model_versions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    feature_name = db.Column(db.String(100), nullable=False)
    metric_type = db.Column(db.String(50), nullable=False)  # PSI, KS_STATISTIC
    metric_value = db.Column(db.Float, nullable=False)
    p_value = db.Column(db.Float, nullable=True)
    threshold = db.Column(db.Float, nullable=False)
    is_drift_detected = db.Column(db.Boolean, nullable=False, default=False, index=True)
    sample_size = db.Column(db.Integer, nullable=False)
    evaluated_at = db.Column(db.DateTime, nullable=False, default=dt.datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "model_version_id": self.model_version_id,
            "feature_name": self.feature_name,
            "metric_type": self.metric_type,
            "metric_value": self.metric_value,
            "p_value": self.p_value,
            "threshold": self.threshold,
            "is_drift_detected": self.is_drift_detected,
            "sample_size": self.sample_size,
            "evaluated_at": self.evaluated_at.isoformat() if self.evaluated_at else None,
        }
