"""
Fraud Detection System (FDS) - Dashboard UI Tests
Author: Alpha
"""

import pytest
from app import create_app
from app.extensions import db
from app.models.alert import Alert, AnalystFeedback
from app.models.merchant import Merchant
from app.models.prediction import ModelVersion, Prediction
from app.models.transaction import Transaction
from app.models.user import AuthUser, User
from app.schemas.transaction import TransactionIngestSchema
from app.services.scoring_service import ScoringService


@pytest.fixture
def app():
    """Create testing application context with seeded test data."""
    app_instance = create_app(config_name="testing")
    with app_instance.app_context():
        db.create_all()

        # Seed Model Version
        mv = ModelVersion(
            version_tag="v1.0.0",
            algorithm="IsolationForest",
            parameters={"n_estimators": 50, "contamination": 0.02},
            metrics={"pr_auc": 0.85},
            decision_threshold=75.0,
            artifact_path="ml/artifacts/isolation_forest_v1.0.0.joblib",
            is_active=True,
        )
        db.session.add(mv)

        # Seed Analyst
        analyst = AuthUser(
            username="analyst",
            email="analyst@internal.local",
            role="analyst",
        )
        analyst.set_password("AnalystSecretPass123!")
        db.session.add(analyst)

        # Seed customer and merchant
        u = User(external_id="usr_dash_001", home_lat=40.7128, home_lon=-74.0060)
        m = Merchant(external_id="mcht_dash_001", name="Apex Electronics", category="electronics")
        db.session.add(u)
        db.session.add(m)
        db.session.commit()

        # Score one normal and one high-risk transaction through ScoringService
        scoring = ScoringService.get_instance()

        # Normal
        scoring.score_transaction(
            TransactionIngestSchema(
                transaction_uid="tx_dash_norm_001",
                user_external_id="usr_dash_001",
                merchant_external_id="mcht_dash_001",
                amount=45.0,
                currency="USD",
                timestamp="2026-03-10T12:00:00Z",
                location_lat=40.7128,
                location_lon=-74.0060,
                device_id="dev_phone_normal",
                channel="POS",
            )
        )

        # Anomaly / Alert
        scoring.score_transaction(
            TransactionIngestSchema(
                transaction_uid="tx_dash_anom_001",
                user_external_id="usr_dash_001",
                merchant_external_id="mcht_dash_001",
                amount=9800.0,
                currency="USD",
                timestamp="2026-03-10T03:30:00Z",
                location_lat=19.0760,
                location_lon=72.8777,
                device_id="dev_foreign_rogue",
                channel="ONLINE",
            )
        )

        yield app_instance

        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


def test_dashboard_redirect_and_overview(client):
    """Verify root redirect and executive overview dashboard page."""
    # 1. Root redirect
    res_root = client.get("/")
    assert res_root.status_code == 302
    assert "/dashboard" in res_root.headers["Location"]

    # 2. Main Dashboard
    res_dash = client.get("/dashboard")
    assert res_dash.status_code == 200
    html = res_dash.get_data(as_text=True)
    assert "System Telemetry &amp; Fraud Operations" in html or "System Telemetry & Fraud Operations" in html
    assert "Transactions Ingested" in html
    assert "Flagged Anomalies" in html
    assert "trendChart" in html
    assert "riskDistChart" in html
    assert "merchantChart" in html


def test_alerts_queue_view_and_filtering(client):
    """Verify live alerts queue rendering and filter parameters."""
    res = client.get("/dashboard/alerts")
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    assert "Analyst Alert Queue" in html
    assert "tx_dash_anom_001" in html

    # Filter by OPEN
    res_open = client.get("/dashboard/alerts?status=OPEN")
    assert res_open.status_code == 200
    assert "tx_dash_anom_001" in res_open.get_data(as_text=True)


def test_transaction_detail_and_shap_chart(client):
    """Verify detailed investigation page with SHAP explanation chart."""
    tx = Transaction.query.filter_by(transaction_uid="tx_dash_anom_001").first()
    assert tx is not None

    res = client.get(f"/dashboard/transaction/{tx.id}")
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    assert "Investigation #tx_dash_anom_001" in html
    assert "TreeSHAP Waterfall Analysis" in html
    assert "shapChart" in html
    assert "Primary Reason Codes" in html


def test_analyst_review_submission(client):
    """Verify analyst decision submission via dashboard form."""
    alert = Alert.query.first()
    assert alert is not None

    form_data = {
        "status": "RESOLVED_FRAUD",
        "verdict": "TRUE_FRAUD",
        "reason_tags": ["AMOUNT_ANOMALY", "FOREIGN_LOCATION"],
        "notes": "Verified fraudulent transaction with customer.",
    }

    res = client.post(
        f"/dashboard/alerts/{alert.id}/review",
        data=form_data,
        follow_redirects=True,
    )
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    assert "Analyst Audit Recorded" in html
    assert "TRUE_FRAUD" in html

    # Verify in DB
    refreshed_alert = db.session.get(Alert, alert.id)
    assert refreshed_alert.status == "RESOLVED_FRAUD"

    feedback = AnalystFeedback.query.filter_by(alert_id=alert.id).first()
    assert feedback is not None
    assert feedback.feedback_verdict == "TRUE_FRAUD"


def test_live_scoring_sandbox_simulation(client):
    """Verify interactive simulator GET and POST submission."""
    # 1. GET Simulator form
    res_get = client.get("/dashboard/simulate")
    assert res_get.status_code == 200
    assert "Live Scoring Sandbox" in res_get.get_data(as_text=True)

    # 2. POST Simulator transaction
    post_data = {
        "amount": "8500.00",
        "hour": "3",
        "user_id": "usr_dash_001",
        "merchant_id": "mcht_dash_001",
        "lat": "51.5074",
        "lon": "-0.1278",
        "device_id": "dev_london_sim",
        "channel": "ONLINE",
    }
    res_post = client.post("/dashboard/simulate", data=post_data)
    assert res_post.status_code == 200
    html = res_post.get_data(as_text=True)
    assert "Pipeline Decision Result" in html
    assert "simShapChart" in html
    assert "Risk Score" in html
