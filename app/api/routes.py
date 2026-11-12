"""
Fraud Detection System (FDS) - REST API Routes (v1)
Author: Alpha
"""

import datetime as dt
import io
from flask import Blueprint, current_app, jsonify, request
from flask_jwt_extended import get_jwt_identity, jwt_required
import pandas as pd
from pydantic import ValidationError

from app.auth.utils import role_required
from app.extensions import db, limiter
from app.models.alert import Alert, AnalystFeedback
from app.models.merchant import Merchant
from app.models.prediction import ModelVersion, Prediction
from app.models.transaction import Transaction
from app.models.user import User
from app.schemas.alert import AlertUpdateSchema
from app.schemas.transaction import (
    BatchTransactionIngestSchema,
    TransactionIngestSchema,
)
from app.services.feedback_service import FeedbackService
from app.services.retrain_service import RetrainService
from app.services.scoring_service import ScoringService

api_bp = Blueprint("api", __name__, url_prefix="/api/v1")


# -----------------------------------------------------------------------------
# 1. HEALTH CHECK
# -----------------------------------------------------------------------------
@api_bp.route("/health", methods=["GET"])
def health():
    """System health check and operational status."""
    svc = ScoringService.get_instance(
        artifacts_dir=current_app.config.get("MODEL_ARTIFACTS_DIR", "ml/artifacts")
    )
    db_status = "healthy"
    try:
        db.session.execute(db.select(1))
    except Exception:
        db_status = "disconnected"

    return jsonify({
        "status": "UP",
        "service": "Fraud Detection System (FDS)",
        "database": db_status,
        "active_model_version": svc.active_version_tag,
        "decision_threshold": svc.decision_threshold,
        "timestamp": dt.datetime.utcnow().isoformat(),
    }), 200


# -----------------------------------------------------------------------------
# 2. REAL-TIME PREDICT (SINGLE TRANSACTION)
# -----------------------------------------------------------------------------
@api_bp.route("/predict", methods=["POST"])
@jwt_required()
@limiter.limit("30 per second")
def predict_single():
    """
    Scores a single financial transaction.
    Returns: Calibrated risk score (0-100), binary flag, and TreeSHAP reason codes.
    """
    payload = request.get_json(silent=True) or {}
    try:
        tx_data = TransactionIngestSchema(**payload)
    except ValidationError as err:
        return jsonify({
            "error": "Validation Error",
            "details": err.errors(),
            "status_code": 422,
        }), 422

    svc = ScoringService.get_instance(
        artifacts_dir=current_app.config.get("MODEL_ARTIFACTS_DIR", "ml/artifacts")
    )
    result = svc.score_transaction(tx_data)

    return jsonify(result.model_dump()), 200


# -----------------------------------------------------------------------------
# 3. BATCH PREDICT (JSON LIST OR CSV UPLOAD)
# -----------------------------------------------------------------------------
@api_bp.route("/predict/batch", methods=["POST"])
@jwt_required()
def predict_batch():
    """
    Scores a batch of transactions via JSON list or multipart CSV file upload.
    """
    svc = ScoringService.get_instance(
        artifacts_dir=current_app.config.get("MODEL_ARTIFACTS_DIR", "ml/artifacts")
    )

    transactions: list[TransactionIngestSchema] = []

    # Handle file upload (CSV)
    if "file" in request.files:
        uploaded_file = request.files["file"]
        if not uploaded_file.filename.endswith(".csv"):
            return jsonify({"error": "Bad Request", "message": "Only CSV files are supported.", "status_code": 400}), 400

        try:
            df = pd.read_csv(io.StringIO(uploaded_file.stream.read().decode("utf-8")))
            # Standardize columns to expected schema
            for _, row in df.iterrows():
                transactions.append(
                    TransactionIngestSchema(
                        transaction_uid=str(row["transaction_id" if "transaction_id" in row else "transaction_uid"]),
                        user_external_id=str(row["user_id" if "user_id" in row else "user_external_id"]),
                        merchant_external_id=str(row["merchant_id" if "merchant_id" in row else "merchant_external_id"]),
                        amount=float(row["amount"]),
                        currency=str(row.get("currency", "USD")),
                        timestamp=str(row["timestamp"]),
                        location_lat=float(row["location_lat"]) if pd.notna(row.get("location_lat")) else None,
                        location_lon=float(row["location_lon"]) if pd.notna(row.get("location_lon")) else None,
                        device_id=str(row["device_id"]),
                        ip_address=str(row.get("ip_address", "127.0.0.1")),
                        channel=str(row.get("channel", "ONLINE")),
                    )
                )
        except Exception as e:
            return jsonify({"error": "Bad Request", "message": f"Malformed CSV file: {str(e)}", "status_code": 400}), 400

    else:
        # Handle JSON body
        payload = request.get_json(silent=True) or {}
        try:
            batch_data = BatchTransactionIngestSchema(**payload)
            transactions = batch_data.transactions
        except ValidationError as err:
            return jsonify({"error": "Validation Error", "details": err.errors(), "status_code": 422}), 422

    result = svc.score_batch(transactions)
    return jsonify(result), 200


# -----------------------------------------------------------------------------
# 4. ALERTS QUEUE (PAGINATED & FILTERABLE)
# -----------------------------------------------------------------------------
@api_bp.route("/alerts", methods=["GET"])
@jwt_required()
def get_alerts():
    """
    Retrieves fraud alert queue with optional status, severity, risk, and date filters.
    """
    status_filter = request.args.get("status")
    severity_filter = request.args.get("severity")
    min_risk = request.args.get("min_risk", type=float)
    page = request.args.get("page", 1, type=int)
    per_page = min(100, request.args.get("per_page", 20, type=int))

    query = (
        db.session.query(Alert, Transaction, Prediction)
        .join(Transaction, Alert.transaction_id == Transaction.id)
        .join(Prediction, Alert.prediction_id == Prediction.id)
    )

    if status_filter:
        query = query.filter(Alert.status == status_filter.upper())
    if severity_filter:
        query = query.filter(Alert.severity == severity_filter.upper())
    if min_risk is not None:
        query = query.filter(Prediction.calibrated_risk_score >= min_risk)

    query = query.order_by(Alert.created_at.desc())
    paginated = query.paginate(page=page, per_page=per_page, error_out=False)

    results = []
    for alert, tx, pred in paginated.items:
        results.append({
            "alert_id": alert.id,
            "alert_uid": alert.alert_uid,
            "severity": alert.severity,
            "status": alert.status,
            "created_at": alert.created_at.isoformat(),
            "trigger_reasons": alert.trigger_reasons,
            "transaction": {
                "transaction_uid": tx.transaction_uid,
                "amount": tx.amount,
                "currency": tx.currency,
                "timestamp": tx.timestamp.isoformat(),
                "device_id": tx.device_id,
                "channel": tx.channel,
            },
            "prediction": {
                "risk_score": pred.calibrated_risk_score,
                "raw_anomaly_score": pred.raw_anomaly_score,
                "latency_ms": pred.latency_ms,
            },
        })

    return jsonify({
        "total": paginated.total,
        "page": paginated.page,
        "pages": paginated.pages,
        "per_page": paginated.per_page,
        "alerts": results,
    }), 200


# -----------------------------------------------------------------------------
# 5. UPDATE ALERT (ANALYST FEEDBACK VERDICT)
# -----------------------------------------------------------------------------
@api_bp.route("/alerts/<int:alert_id>", methods=["PATCH"])
@jwt_required()
def update_alert(alert_id: int):
    """
    Analyst resolution endpoint: updates alert status and logs true fraud / false positive feedback.
    """
    payload = request.get_json(silent=True) or {}
    try:
        data = AlertUpdateSchema(**payload)
    except ValidationError as err:
        return jsonify({"error": "Validation Error", "details": err.errors(), "status_code": 422}), 422

    analyst_id = int(get_jwt_identity())

    try:
        updated = FeedbackService.process_feedback(
            alert_id=alert_id,
            analyst_id=analyst_id,
            new_status=data.status,
            feedback_verdict=data.feedback_verdict,
            reason_tags=data.reason_tags,
            notes=data.notes,
        )
    except ValueError as e:
        return jsonify({"error": "Not Found", "message": str(e), "status_code": 404}), 404

    return jsonify({"message": "Alert updated successfully.", "alert": updated}), 200


# -----------------------------------------------------------------------------
# 6. SYSTEM METRICS & PERFORMANCE TELEMETRY
# -----------------------------------------------------------------------------
@api_bp.route("/metrics", methods=["GET"])
@jwt_required()
def get_metrics():
    """
    Returns high-level model performance metrics, alert statistics, and volume breakdown.
    """
    total_tx = Transaction.query.count()
    flagged_tx = Prediction.query.filter_by(is_flagged=True).count()
    open_alerts = Alert.query.filter_by(status="OPEN").count()
    resolved_fraud = Alert.query.filter_by(status="RESOLVED_FRAUD").count()
    resolved_fp = Alert.query.filter_by(status="RESOLVED_FALSE_POSITIVE").count()

    active_model = ModelVersion.query.filter_by(is_active=True).first()

    return jsonify({
        "total_transactions_scored": total_tx,
        "total_anomalies_flagged": flagged_tx,
        "anomaly_rate_percentage": round((flagged_tx / max(1, total_tx)) * 100.0, 2),
        "alerts": {
            "open": open_alerts,
            "resolved_fraud": resolved_fraud,
            "resolved_false_positive": resolved_fp,
        },
        "active_model": active_model.to_dict() if active_model else {
            "version_tag": "v1.0.0",
            "algorithm": "IsolationForest",
            "decision_threshold": 80.0,
        },
    }), 200


# -----------------------------------------------------------------------------
# 7. MODEL RETRAINING TRIGGER (ADMIN ONLY)
# -----------------------------------------------------------------------------
@api_bp.route("/retrain", methods=["POST"])
@role_required("admin")
def trigger_retrain():
    """
    Triggers automated model retraining incorporating analyst feedback ground truth.
    Restricted strictly to administrators.
    """
    artifacts_dir = current_app.config.get("MODEL_ARTIFACTS_DIR", "ml/artifacts")
    retrain_summary = RetrainService.retrain_model(artifacts_dir=artifacts_dir)

    # Reload in-memory singleton
    ScoringService.get_instance(artifacts_dir=artifacts_dir).load_artifacts()

    return jsonify({
        "message": "Model retrained and deployed successfully.",
        "summary": retrain_summary,
    }), 200
