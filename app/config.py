"""
Fraud Detection System (FDS) - Configuration Module
Author: Alpha
"""

import os
from datetime import timedelta
from typing import Type


class BaseConfig:
    """Base application settings."""
    SECRET_KEY = os.getenv("SECRET_KEY", "alpha_fds_default_insecure_secret_key_12345")
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "alpha_fds_jwt_super_secret_key_67890")
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(
        minutes=int(os.getenv("JWT_ACCESS_TOKEN_EXPIRES_MINUTES", "60"))
    )

    # Database
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL",
        "sqlite:///fraud_detection.db"
    )
    # Fix for Heroku/Render postgres:// vs postgresql://
    if SQLALCHEMY_DATABASE_URI.startswith("postgres://"):
        SQLALCHEMY_DATABASE_URI = SQLALCHEMY_DATABASE_URI.replace("postgres://", "postgresql://", 1)

    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_pre_ping": True,
    }

    # Model & Artifacts
    MODEL_ARTIFACTS_DIR = os.getenv("MODEL_ARTIFACTS_DIR", "ml/artifacts")
    DEFAULT_MODEL_VERSION = os.getenv("DEFAULT_MODEL_VERSION", "v1.0.0")

    # Rate Limiting Defaults
    RATELIMIT_DEFAULT = os.getenv("RATELIMIT_DEFAULT", "100 per minute")
    RATELIMIT_PREDICT = os.getenv("RATELIMIT_PREDICT", "30 per second")
    RATELIMIT_STORAGE_URI = os.getenv("RATELIMIT_STORAGE_URI", "memory://")

    # Alert & Risk Threshold Defaults
    CRITICAL_RISK_THRESHOLD = float(os.getenv("CRITICAL_RISK_THRESHOLD", "85.0"))
    HIGH_RISK_THRESHOLD = float(os.getenv("HIGH_RISK_THRESHOLD", "70.0"))
    MEDIUM_RISK_THRESHOLD = float(os.getenv("MEDIUM_RISK_THRESHOLD", "45.0"))


class DevelopmentConfig(BaseConfig):
    DEBUG = True


class ProductionConfig(BaseConfig):
    DEBUG = False


class TestingConfig(BaseConfig):
    TESTING = True
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    RATELIMIT_ENABLED = False
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(minutes=15)


CONFIG_MAP = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
    "default": DevelopmentConfig,
}


def get_config(env_name: str | None = None) -> Type[BaseConfig]:
    env = env_name or os.getenv("FLASK_ENV", "development").lower()
    return CONFIG_MAP.get(env, DevelopmentConfig)
