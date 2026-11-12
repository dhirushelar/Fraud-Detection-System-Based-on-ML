"""
Fraud Detection System (FDS) - Application Factory
Author: Alpha
"""

import os
from flask import Flask, jsonify

from app.api import api_bp
from app.api.errors import register_error_handlers
from app.auth import auth_bp
from app.config import get_config
from app.dashboard import dashboard_bp
from app.extensions import cors, db, jwt, limiter


def create_app(config_name: str | None = None) -> Flask:
    """Constructs and configures the Flask application using the factory pattern."""
    app = Flask(__name__)

    # Load configuration
    cfg_class = get_config(config_name)
    app.config.from_object(cfg_class)

    # Initialize extensions
    db.init_app(app)
    jwt.init_app(app)
    cors.init_app(app, resources={r"/*": {"origins": "*"}})
    limiter.init_app(app)

    # Register structured error handlers
    register_error_handlers(app)

    # Register blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(auth_bp, url_prefix="/api/v1/auth", name="auth_v1")
    app.register_blueprint(api_bp)
    app.register_blueprint(dashboard_bp)

    # Root health check shortcut for infrastructure and container probes
    @app.route("/health", methods=["GET"])
    def root_health():
        return jsonify({
            "status": "UP",
            "service": "Fraud Detection System (FDS)",
            "environment": app.config.get("FLASK_ENV", "production"),
        }), 200

    # Auto-create tables and bootstrap default credentials if unseeded
    with app.app_context():
        try:
            db.create_all()
            from app.models.user import AuthUser
            from app.models.prediction import ModelVersion

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

            if not ModelVersion.query.filter_by(version_tag="v1.0.0").first():
                mv = ModelVersion(
                    version_tag="v1.0.0",
                    model_type="IsolationForest",
                    threshold=81.5,
                    is_active=True,
                )
                db.session.add(mv)

            db.session.commit()
        except Exception:
            db.session.rollback()

    return app
