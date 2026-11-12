"""
Fraud Detection System (FDS) - End-to-End Live System Demo Walkthrough
Author: Alpha
Description: Automated script demonstrating the complete lifecycle:
             1. JWT Authentication (Analyst & Admin)
             2. Health & Model Status Check
             3. Real-time Transaction Scoring & SHAP Reason Codes
             4. Batch Scoring Ingestion
             5. Alert Queue Inspection & Filtering
             6. Analyst Feedback Loop Resolution (Ground Truth)
             7. Automated Model Retraining Trigger (Admin Only)
             8. Performance Telemetry & Cost ROI Aggregates
"""

import json
import os
import sys
import time

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import create_app
from app.extensions import db


def run_demo():
    print("=" * 75)
    print("      FRAUD DETECTION SYSTEM (FDS) - LIVE END-TO-END DEMO")
    print("      Author: Alpha | Model: Isolation Forest + TreeSHAP")
    print("=" * 75)

    app = create_app("development")
    client = app.test_client()

    with app.app_context():
        # Ensure schema exists
        db.create_all()

    # Step 1: Health Probe
    print("\n[Step 1] Checking System Health & Model Telemetry...")
    res = client.get("/api/v1/health")
    print(f"Status Code: {res.status_code}")
    print(json.dumps(res.get_json(), indent=2))

    # Step 2: Analyst Authentication
    print("\n[Step 2] Authenticating as Fraud Analyst ('analyst')...")
    login_res = client.post(
        "/api/v1/auth/login",
        json={"username": "analyst", "password": "AnalystSecretPass123!"},
    )
    if login_res.status_code != 200:
        print("Login failed with status:", login_res.status_code, login_res.get_json())
        # Try seeding first
        from scripts.seed_db import seed_database
        seed_database()
        login_res = client.post(
            "/api/v1/auth/login",
            json={"username": "analyst", "password": "AnalystSecretPass123!"},
        )
    assert login_res.status_code == 200, f"Analyst login failed: {login_res.get_json()}"
    analyst_token = login_res.get_json()["access_token"]
    analyst_headers = {"Authorization": f"Bearer {analyst_token}"}
    print("[✓] JWT Access Token Acquired.")

    # Step 3: Real-Time Scoring - Normal Transaction
    print("\n[Step 3] Scoring Normal Ingest Transaction ($42.50 at Grocery Store)...")
    tx_normal = {
        "transaction_uid": f"demo_norm_{int(time.time())}",
        "user_external_id": "usr_00001",
        "merchant_external_id": "mcht_0001",
        "amount": 42.50,
        "currency": "USD",
        "timestamp": "2026-03-15T14:30:00Z",
        "location_lat": 40.7128,
        "location_lon": -74.0060,
        "device_id": "dev_phone_safari",
        "channel": "POS",
    }
    res_norm = client.post("/api/v1/predict", headers=analyst_headers, json=tx_normal)
    norm_body = res_norm.get_json()
    decision_norm = "BLOCK/FLAG" if norm_body["is_flagged"] else "APPROVE"
    print(f"Decision:    {decision_norm}")
    print(f"Risk Score:  {norm_body['risk_score']} / 100 (Severity: {norm_body['severity']})")
    print(f"Flagged:     {norm_body['is_flagged']}")
    print(f"Latency:     {norm_body['latency_ms']} ms")

    # Step 4: Real-Time Scoring - High-Risk Fraud Anomaly
    print("\n[Step 4] Scoring High-Risk Anomaly ($9,450.00 at 3:15 AM from Foreign Country)...")
    tx_fraud = {
        "transaction_uid": f"demo_fraud_{int(time.time())}",
        "user_external_id": "usr_00001",
        "merchant_external_id": "mcht_0003",
        "amount": 9450.00,
        "currency": "USD",
        "timestamp": "2026-03-15T03:15:00Z",
        "location_lat": 48.8566,
        "location_lon": 2.3522,
        "device_id": "dev_unrecognized_foreign",
        "channel": "ONLINE",
    }
    res_fraud = client.post("/api/v1/predict", headers=analyst_headers, json=tx_fraud)
    fraud_body = res_fraud.get_json()
    decision_fraud = "BLOCK/FLAG" if fraud_body["is_flagged"] else "APPROVE"
    print(f"Decision:    {decision_fraud}")
    print(f"Risk Score:  {fraud_body['risk_score']} / 100 (Severity: {fraud_body['severity']})")
    print(f"Flagged:     {fraud_body['is_flagged']}")
    print(f"Latency:     {fraud_body['latency_ms']} ms")
    print("TreeSHAP Reason Codes:")
    for r in fraud_body.get("reason_codes", []):
        print(f"  • [Impact: {r['impact']:+.4f}] {r['code']}: {r['description']}")

    # Step 5: Query Live Alert Queue
    print("\n[Step 5] Querying Live Analyst Alert Queue...")
    alerts_res = client.get("/api/v1/alerts?status=OPEN", headers=analyst_headers)
    alerts_data = alerts_res.get_json()
    print(f"Total Open Alerts in Queue: {alerts_data['total']}")
    if alerts_data["alerts"]:
        target_alert = alerts_data["alerts"][0]
        alert_id = target_alert.get("alert_id") or target_alert.get("id")
        amount_val = target_alert.get("transaction", {}).get("amount", target_alert.get("amount", 0.0))
        print(f"Target Alert ID: {alert_id} (UID: {target_alert['alert_uid']})")
        print(f"Amount: ${amount_val}, Severity: {target_alert['severity']}")

        # Step 6: Analyst Reviews & Submits Ground Truth Verdict
        print("\n[Step 6] Submitting Analyst Ground-Truth Resolution (RESOLVED_FRAUD)...")
        feedback_payload = {
            "status": "RESOLVED_FRAUD",
            "feedback_verdict": "TRUE_FRAUD",
            "reason_tags": ["AMOUNT_ANOMALY", "FOREIGN_LOCATION"],
            "notes": "Verified unauthorized charge with cardholder over SMS verification.",
        }
        patch_res = client.patch(f"/api/v1/alerts/{alert_id}", headers=analyst_headers, json=feedback_payload)
        print(f"Status Code: {patch_res.status_code}")
        print(f"Updated Alert Status: {patch_res.get_json()['alert']['status']}")
        print("[✓] Feedback recorded in active retraining buffer.")

    # Step 7: Admin RBAC & Automated Retraining
    print("\n[Step 7] Authenticating as Administrator ('admin') & Triggering Retraining...")
    admin_login = client.post(
        "/auth/login",
        json={"username": "admin", "password": "AdminSecretPass123!"},
    )
    admin_token = admin_login.get_json()["access_token"]
    admin_headers = {"Authorization": f"Bearer {admin_token}"}

    retrain_res = client.post("/api/v1/retrain", headers=admin_headers)
    print(f"Retrain Status Code: {retrain_res.status_code}")
    retrain_data = retrain_res.get_json()
    print(f"New Model Version:     {retrain_data.get('new_version')}")
    print(f"Updated Cutoff:        {retrain_data.get('decision_threshold')}")
    print(f"Training Dataset Size: {retrain_data.get('dataset_size')} samples")
    print(f"Cost Savings vs Base:  ${retrain_data.get('cost_metrics', {}).get('savings_vs_baseline', 0):,.2f}")

    # Step 8: Telemetry KPI Aggregation
    print("\n[Step 8] Fetching Final Telemetry Metrics & Operations Aggregates...")
    metrics_res = client.get("/api/v1/metrics", headers=analyst_headers)
    print(json.dumps(metrics_res.get_json(), indent=2))

    print("\n" + "=" * 75)
    print("      [✓] DEMO WALKTHROUGH COMPLETED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    run_demo()
