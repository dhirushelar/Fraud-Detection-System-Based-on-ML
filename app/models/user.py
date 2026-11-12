"""
Fraud Detection System (FDS) - User Models
Author: Alpha
"""

import datetime as dt
from werkzeug.security import check_password_hash, generate_password_hash

from app.extensions import db


class AuthUser(db.Model):
    """System RBAC user accounts for Fraud Analysts and Administrators."""
    __tablename__ = "auth_users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False, index=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default="analyst", index=True)
    is_active = db.Column(db.Boolean, nullable=False, default=True)
    created_at = db.Column(db.DateTime, nullable=False, default=dt.datetime.utcnow)
    updated_at = db.Column(
        db.DateTime,
        nullable=False,
        default=dt.datetime.utcnow,
        onupdate=dt.datetime.utcnow,
    )

    # Relationships
    assigned_alerts = db.relationship("Alert", backref="assigned_analyst", lazy="dynamic")
    feedbacks = db.relationship("AnalystFeedback", backref="analyst", lazy="dynamic")

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "role": self.role,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class User(db.Model):
    """Customer bank account holder / cardholder entity."""
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    external_id = db.Column(db.String(64), unique=True, nullable=False, index=True)
    first_name = db.Column(db.String(100), nullable=True)
    last_name = db.Column(db.String(100), nullable=True)
    email = db.Column(db.String(255), nullable=True)
    account_created_at = db.Column(db.DateTime, nullable=False, default=dt.datetime.utcnow)
    risk_status = db.Column(db.String(30), nullable=False, default="NORMAL", index=True)
    home_lat = db.Column(db.Float, nullable=True)
    home_lon = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=dt.datetime.utcnow)
    updated_at = db.Column(
        db.DateTime,
        nullable=False,
        default=dt.datetime.utcnow,
        onupdate=dt.datetime.utcnow,
    )

    transactions = db.relationship("Transaction", backref="user", lazy="dynamic")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "external_id": self.external_id,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "email": self.email,
            "risk_status": self.risk_status,
            "home_lat": self.home_lat,
            "home_lon": self.home_lon,
        }
