"""
Fraud Detection System (FDS) - Analyst Feedback Service
Author: Alpha
Description: Manages fraud analyst review submissions and alert resolution states.
"""

import datetime as dt
from typing import Any, Dict, List, Optional

from app.extensions import db
from app.models.alert import Alert, AnalystFeedback
from app.models.transaction import Transaction


class FeedbackService:
    """Handles operational workflow updates from human fraud analysts."""

    @staticmethod
    def process_feedback(
        alert_id: int,
        analyst_id: int,
        new_status: str,
        feedback_verdict: Optional[str] = None,
        reason_tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        alert = db.session.get(Alert, alert_id)
        if not alert:
            raise ValueError(f"Alert with id {alert_id} not found.")

        # Update alert status
        alert.status = new_status
        alert.assigned_to_user_id = analyst_id
        alert.updated_at = dt.datetime.utcnow()

        # Update underlying transaction ground truth
        tx = db.session.get(Transaction, alert.transaction_id)
        if tx:
            if feedback_verdict == "TRUE_FRAUD" or new_status == "RESOLVED_FRAUD":
                tx.is_fraud_ground_truth = True
            elif feedback_verdict == "FALSE_POSITIVE" or new_status == "RESOLVED_FALSE_POSITIVE":
                tx.is_fraud_ground_truth = False

        # If a verdict is explicitly provided, record or update analyst_feedback
        if feedback_verdict:
            fb = AnalystFeedback.query.filter_by(alert_id=alert.id).first()
            if not fb:
                fb = AnalystFeedback(
                    alert_id=alert.id,
                    transaction_id=alert.transaction_id,
                    analyst_id=analyst_id,
                    feedback_verdict=feedback_verdict,
                    reason_tags=reason_tags or [],
                    notes=notes,
                    reviewed_at=dt.datetime.utcnow(),
                )
                db.session.add(fb)
            else:
                fb.feedback_verdict = feedback_verdict
                fb.reason_tags = reason_tags or []
                fb.notes = notes
                fb.reviewed_at = dt.datetime.utcnow()

        db.session.commit()
        return alert.to_dict()

    @classmethod
    def update_alert_review(
        cls,
        alert_id: int,
        analyst_id: int,
        status: str,
        verdict: Optional[str] = None,
        reason_tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Convenience wrapper for dashboard review submissions."""
        return cls.process_feedback(
            alert_id=alert_id,
            analyst_id=analyst_id,
            new_status=status,
            feedback_verdict=verdict,
            reason_tags=reason_tags,
            notes=notes,
        )
