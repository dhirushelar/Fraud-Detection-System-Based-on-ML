"""
Fraud Detection System (FDS) - Transaction Model
Author: Alpha
"""

import datetime as dt
from app.extensions import db


class Transaction(db.Model):
    """Financial transaction ledger record."""
    __tablename__ = "transactions"

    id = db.Column(
        db.BigInteger().with_variant(db.Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    transaction_uid = db.Column(db.String(64), unique=True, nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    merchant_id = db.Column(db.Integer, db.ForeignKey("merchants.id", ondelete="RESTRICT"), nullable=False, index=True)
    amount = db.Column(db.Float, nullable=False)
    currency = db.Column(db.String(3), nullable=False, default="USD")
    timestamp = db.Column(db.DateTime, nullable=False, index=True)
    location_lat = db.Column(db.Float, nullable=True)
    location_lon = db.Column(db.Float, nullable=True)
    device_id = db.Column(db.String(128), nullable=False, index=True)
    ip_address = db.Column(db.String(45), nullable=True)
    channel = db.Column(db.String(30), nullable=False, default="ONLINE")
    is_ground_truth_fraud = db.Column(db.Boolean, nullable=True)
    status = db.Column(db.String(30), nullable=False, default="PROCESSED")
    created_at = db.Column(db.DateTime, nullable=False, default=dt.datetime.utcnow)

    # Relationships
    prediction = db.relationship("Prediction", backref="transaction", uselist=False, cascade="all, delete-orphan")
    alert = db.relationship("Alert", backref="transaction", uselist=False, cascade="all, delete-orphan")

    __table_args__ = (
        db.Index("idx_tx_user_timestamp", "user_id", "timestamp"),
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "transaction_uid": self.transaction_uid,
            "user_id": self.user_id,
            "merchant_id": self.merchant_id,
            "amount": self.amount,
            "currency": self.currency,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "location_lat": self.location_lat,
            "location_lon": self.location_lon,
            "device_id": self.device_id,
            "ip_address": self.ip_address,
            "channel": self.channel,
            "status": self.status,
            "is_ground_truth_fraud": self.is_ground_truth_fraud,
        }
