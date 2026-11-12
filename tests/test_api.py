"""
Fraud Detection System (FDS) - REST API & Authentication Test Suite
Author: Alpha
"""

import json
import pytest

from app import create_app
from app.extensions import db
from app.models.merchant import Merchant
from app.models.prediction import ModelVersion
from app.models.user import AuthUser, User


@pytest.fixture
def client():
    """Configures a clean test client with in-memory SQLite and seeded test users."""
    app = create_app("testing")
    with app.app_context():
        db.create_all()

        # Seed Admin and Analyst accounts
        admin = AuthUser(
            username="admin_test",
            email="admin@test.local",
            role="admin",
            is_active=True,
        )
        admin.set_password("AdminSecretPass123!")
        db.session.add(admin)

        analyst = AuthUser(
            username="analyst_test",
            email="analyst@test.local",
            role="analyst",
            is_active=True,
        )
        analyst.set_password("AnalystSecretPass123!")
        db.session.add(analyst)

        # Seed sample Customer and Merchant
        cust = User(
            external_id="usr_test_001",
            first_name="Alice",
            last_name="Tester",
            home_lat=40.7128,
            home_lon=-74.0060,
        )
        db.session.add(cust)

        mcht = Merchant(
            external_id="mcht_test_001",
            name="Test Electronics",
            category="electronics",
            risk_rating=2.0,
        )
        db.session.add(mcht)

        # Seed active model version
        mv = ModelVersion(
            version_tag="v1.0.0",
            algorithm="IsolationForest",
            decision_threshold=80.0,
            artifact_path="ml/artifacts/isolation_forest_v1.0.0.joblib",
            is_active=True,
        )
        db.session.add(mv)

        db.session.commit()

        with app.test_client() as test_client:
            yield test_client

        db.session.remove()
        db.drop_all()


def get_jwt_token(client, username, password):
    """Helper to authenticate and retrieve JWT access token."""
    res = client.post(
        "/auth/login",
        json={"username": username, "password": password},
    )
    assert res.status_code == 200
    return res.get_json()["access_token"]


def test_health_endpoints(client):
    """Verify system health check endpoints return status UP."""
    res1 = client.get("/health")
    assert res1.status_code == 200
    data1 = res1.get_json()
    assert data1["status"] == "UP"

    res2 = client.get("/api/v1/health")
    assert res2.status_code == 200
    data2 = res2.get_json()
    assert data2["database"] == "healthy"
    assert "active_model_version" in data2


def test_auth_login_and_me(client):
    """Verify authentication succeeds with valid credentials and fails on invalid."""
    # Valid login
    token = get_jwt_token(client, "analyst_test", "AnalystSecretPass123!")
    assert token is not None

    # Invalid password
    res_bad = client.post(
        "/auth/login",
        json={"username": "analyst_test", "password": "WrongPassword!"},
    )
    assert res_bad.status_code == 401

    # Protected me endpoint
    res_me = client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res_me.status_code == 200
    assert res_me.get_json()["user"]["username"] == "analyst_test"


def test_role_enforcement_on_admin_endpoint(client):
    """Verify analysts are forbidden from admin endpoints and admins are allowed."""
    analyst_token = get_jwt_token(client, "analyst_test", "AnalystSecretPass123!")
    admin_token = get_jwt_token(client, "admin_test", "AdminSecretPass123!")

    # Analyst attempt -> 403 Forbidden
    res_analyst = client.post(
        "/api/v1/retrain",
        headers={"Authorization": f"Bearer {analyst_token}"},
    )
    assert res_analyst.status_code == 403

    # Admin attempt -> 200 OK
    res_admin = client.post(
        "/api/v1/retrain",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert res_admin.status_code == 200
    assert "summary" in res_admin.get_json()


def test_predict_single_valid_and_invalid(client):
    """Verify single transaction scoring endpoint, validation, and explainability."""
    token = get_jwt_token(client, "analyst_test", "AnalystSecretPass123!")

    # 1. Valid transaction
    valid_tx = {
        "transaction_uid": "tx_api_test_001",
        "user_external_id": "usr_test_001",
        "merchant_external_id": "mcht_test_001",
        "amount": 89.50,
        "currency": "USD",
        "timestamp": "2026-03-10T14:30:00Z",
        "location_lat": 40.7128,
        "location_lon": -74.0060,
        "device_id": "dev_test_phone_01",
        "ip_address": "192.168.1.50",
        "channel": "MOBILE",
    }
    res = client.post(
        "/api/v1/predict",
        headers={"Authorization": f"Bearer {token}"},
        json=valid_tx,
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["transaction_uid"] == "tx_api_test_001"
    assert 0.0 <= data["risk_score"] <= 100.0
    assert data["severity"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    assert "reason_codes" in data
    assert data["latency_ms"] >= 0.0

    # 2. Invalid transaction (Negative amount -> 422)
    invalid_tx = dict(valid_tx)
    invalid_tx["amount"] = -50.0
    res_inv = client.post(
        "/api/v1/predict",
        headers={"Authorization": f"Bearer {token}"},
        json=invalid_tx,
    )
    assert res_inv.status_code == 422

    # 3. Missing authorization header -> 401
    res_no_auth = client.post("/api/v1/predict", json=valid_tx)
    assert res_no_auth.status_code == 401


def test_predict_batch_json(client):
    """Verify batch prediction scoring on a list of transactions."""
    token = get_jwt_token(client, "analyst_test", "AnalystSecretPass123!")

    batch_payload = {
        "transactions": [
            {
                "transaction_uid": f"tx_batch_{i:03d}",
                "user_external_id": "usr_test_001",
                "merchant_external_id": "mcht_test_001",
                "amount": float(50.0 + i * 10),
                "currency": "USD",
                "timestamp": f"2026-03-10T15:{i:02d}:00Z",
                "location_lat": 40.7128,
                "location_lon": -74.0060,
                "device_id": "dev_batch_1",
                "ip_address": "127.0.0.1",
                "channel": "ONLINE",
            }
            for i in range(5)
        ]
    }
    res = client.post(
        "/api/v1/predict/batch",
        headers={"Authorization": f"Bearer {token}"},
        json=batch_payload,
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["total_processed"] == 5
    assert len(data["predictions"]) == 5


def test_alerts_queue_and_analyst_feedback(client):
    """Verify alerts querying, filtering, and analyst resolution workflow."""
    token = get_jwt_token(client, "analyst_test", "AnalystSecretPass123!")

    # Ingest a deliberately anomalous transaction to trigger an alert
    anomalous_tx = {
        "transaction_uid": "tx_flagged_for_alert",
        "user_external_id": "usr_test_001",
        "merchant_external_id": "mcht_test_001",
        "amount": 2500.0,  # Extreme amount
        "currency": "USD",
        "timestamp": "2026-03-11T03:15:00Z",  # Odd hour
        "location_lat": 34.0522,  # Divergent location
        "location_lon": -118.2437,
        "device_id": "dev_unseen_attacker_device",
        "ip_address": "10.0.0.99",
        "channel": "ONLINE",
    }
    client.post(
        "/api/v1/predict",
        headers={"Authorization": f"Bearer {token}"},
        json=anomalous_tx,
    )

    # Query alerts queue
    res_alerts = client.get(
        "/api/v1/alerts",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res_alerts.status_code == 200
    alerts_data = res_alerts.get_json()
    assert "alerts" in alerts_data

    if alerts_data["alerts"]:
        target_alert = alerts_data["alerts"][0]
        alert_id = target_alert["alert_id"]

        # Analyst reviews and resolves alert
        update_payload = {
            "status": "RESOLVED_FRAUD",
            "feedback_verdict": "TRUE_FRAUD",
            "reason_tags": ["unusual_device", "high_amount"],
            "notes": "Verified fraudulent credential stuffing incident.",
        }
        res_patch = client.patch(
            f"/api/v1/alerts/{alert_id}",
            headers={"Authorization": f"Bearer {token}"},
            json=update_payload,
        )
        assert res_patch.status_code == 200
        assert res_patch.get_json()["alert"]["status"] == "RESOLVED_FRAUD"


def test_system_metrics_endpoint(client):
    """Verify metrics endpoint reports aggregate counters and active model."""
    token = get_jwt_token(client, "analyst_test", "AnalystSecretPass123!")
    res = client.get(
        "/api/v1/metrics",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 200
    data = res.get_json()
    assert "total_transactions_scored" in data
    assert "total_anomalies_flagged" in data
    assert "alerts" in data
    assert "active_model" in data
