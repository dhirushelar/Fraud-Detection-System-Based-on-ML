"""
Fraud Detection System (FDS) - Merchant Model
Author: Alpha
"""

import datetime as dt
from app.extensions import db


class Merchant(db.Model):
    """Merchant entity receiving transaction funds."""
    __tablename__ = "merchants"

    id = db.Column(db.Integer, primary_key=True)
    external_id = db.Column(db.String(64), unique=True, nullable=False, index=True)
    name = db.Column(db.String(255), nullable=False)
    category = db.Column(db.String(100), nullable=False, index=True)
    risk_rating = db.Column(db.Float, nullable=False, default=1.0)
    created_at = db.Column(db.DateTime, nullable=False, default=dt.datetime.utcnow)

    transactions = db.relationship("Transaction", backref="merchant", lazy="dynamic")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "external_id": self.external_id,
            "name": self.name,
            "category": self.category,
            "risk_rating": self.risk_rating,
        }
