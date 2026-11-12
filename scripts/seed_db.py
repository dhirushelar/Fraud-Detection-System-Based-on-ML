"""
Fraud Detection System (FDS) - Database Seeding Script
Author: Alpha
Description: Seeds initial RBAC accounts (analyst, admin), test customers,
             merchants, and active model version record.
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import create_app
from app.extensions import db
from app.models.merchant import Merchant
from app.models.prediction import ModelVersion
from app.models.user import AuthUser, User


def seed_database():
    app = create_app("development")
    with app.app_context():
        print("[+] Creating database schema tables...")
        db.create_all()

        # 1. Seed RBAC Users
        print("[+] Seeding RBAC users...")
        if not AuthUser.query.filter_by(username="admin").first():
            admin = AuthUser(
                username="admin",
                email="admin@alpha-fds.internal",
                role="admin",
                is_active=True,
            )
            admin.set_password("AdminSecretPass123!")
            db.session.add(admin)

        if not AuthUser.query.filter_by(username="analyst").first():
            analyst = AuthUser(
                username="analyst",
                email="analyst@alpha-fds.internal",
                role="analyst",
                is_active=True,
            )
            analyst.set_password("AnalystSecretPass123!")
            db.session.add(analyst)

        # 2. Seed Sample Customers
        print("[+] Seeding sample customers...")
        for i in range(1, 21):
            ext_id = f"usr_{i:05d}"
            if not User.query.filter_by(external_id=ext_id).first():
                u = User(
                    external_id=ext_id,
                    first_name=f"Customer_{i}",
                    last_name="Test",
                    email=f"user_{i}@example.com",
                    home_lat=40.7128 + (i * 0.005),
                    home_lon=-74.0060 + (i * 0.005),
                    risk_status="NORMAL",
                )
                db.session.add(u)

        # 3. Seed Sample Merchants
        print("[+] Seeding sample merchants...")
        categories = ["grocery", "electronics", "travel", "luxury_goods", "restaurant"]
        for j in range(1, 11):
            m_id = f"mcht_{j:04d}"
            if not Merchant.query.filter_by(external_id=m_id).first():
                m = Merchant(
                    external_id=m_id,
                    name=f"Merchant Brand {j}",
                    category=categories[j % len(categories)],
                    risk_rating=1.2,
                )
                db.session.add(m)

        # 4. Seed Active Model Version
        if not ModelVersion.query.filter_by(version_tag="v1.0.0").first():
            mv = ModelVersion(
                version_tag="v1.0.0",
                algorithm="IsolationForest",
                decision_threshold=81.50,
                artifact_path="ml/artifacts/isolation_forest_v1.0.0.joblib",
                is_active=True,
            )
            db.session.add(mv)

        db.session.commit()

        # 5. Seed Sample Scored Transactions & Alerts
        from app.models.transaction import Transaction
        if Transaction.query.count() == 0:
            print("[+] Scoring & seeding realistic transaction records...")
            import datetime as dt
            from app.schemas.transaction import TransactionIngestSchema
            from app.services.scoring_service import ScoringService

            scoring = ScoringService.get_instance()
            now = dt.datetime.utcnow()

            # Normal transactions across previous 7 days
            for i in range(1, 45):
                t_day = now - dt.timedelta(days=(i % 7), hours=(i % 12) + 8)
                cust_idx = (i % 20) + 1
                mcht_idx = (i % 10) + 1
                scoring.score_transaction(
                    TransactionIngestSchema(
                        transaction_uid=f"tx_seed_norm_{i:04d}",
                        user_external_id=f"usr_{cust_idx:05d}",
                        merchant_external_id=f"mcht_{mcht_idx:04d}",
                        amount=float(25.0 + (i * 3.5)),
                        currency="USD",
                        timestamp=t_day.isoformat() + "Z",
                        location_lat=40.7128 + (cust_idx * 0.005),
                        location_lon=-74.0060 + (cust_idx * 0.005),
                        device_id=f"dev_phone_{cust_idx:03d}",
                        channel="POS" if i % 2 == 0 else "ONLINE",
                    )
                )

            # High-risk anomalies
            anomaly_configs = [
                (1, 1, 9400.0, 3, 48.8566, 2.3522, "dev_foreign_x1", "ONLINE"),
                (2, 3, 7850.0, 2, 19.0760, 72.8777, "dev_foreign_x2", "ONLINE"),
                (3, 4, 8200.0, 4, 35.6762, 139.6503, "dev_foreign_x3", "ONLINE"),
                (4, 5, 9999.0, 1, -33.8688, 151.2093, "dev_foreign_x4", "ONLINE"),
                (5, 2, 6500.0, 23, 51.5074, -0.1278, "dev_foreign_x5", "ONLINE"),
            ]

            for idx, (c_id, m_id, amt, hr, lat, lon, dev, ch) in enumerate(anomaly_configs, start=1):
                t_anom = now - dt.timedelta(days=(idx % 3), hours=hr)
                scoring.score_transaction(
                    TransactionIngestSchema(
                        transaction_uid=f"tx_seed_fraud_{idx:04d}",
                        user_external_id=f"usr_{c_id:05d}",
                        merchant_external_id=f"mcht_{m_id:04d}",
                        amount=amt,
                        currency="USD",
                        timestamp=t_anom.isoformat() + "Z",
                        location_lat=lat,
                        location_lon=lon,
                        device_id=dev,
                        channel=ch,
                    )
                )

            # Add sample analyst feedback on 2 alerts
            from app.models.alert import Alert
            from app.services.feedback_service import FeedbackService
            analyst_user = AuthUser.query.filter_by(username="analyst").first()
            if analyst_user:
                alerts = Alert.query.limit(2).all()
                if len(alerts) >= 1:
                    FeedbackService.process_feedback(
                        alert_id=alerts[0].id,
                        analyst_id=analyst_user.id,
                        new_status="RESOLVED_FRAUD",
                        feedback_verdict="TRUE_FRAUD",
                        reason_tags=["AMOUNT_ANOMALY", "FOREIGN_LOCATION"],
                        notes="Cardholder confirmed unauthorized access from abroad.",
                    )
                if len(alerts) >= 2:
                    FeedbackService.process_feedback(
                        alert_id=alerts[1].id,
                        analyst_id=analyst_user.id,
                        new_status="INVESTIGATING",
                        feedback_verdict="SUSPICIOUS_UNCONFIRMED",
                        reason_tags=["VELOCITY_BURST"],
                        notes="Awaiting response from cardholder verification SMS.",
                    )

        print("\n[✓] Database successfully seeded!")
        print("    - Admin:   username='admin',   password='AdminSecretPass123!'")
        print("    - Analyst: username='analyst', password='AnalystSecretPass123!'")


if __name__ == "__main__":
    seed_database()
