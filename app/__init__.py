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

    # Auto-create tables if running in SQLite or testing mode
    with app.app_context():
        if "sqlite" in app.config.get("SQLALCHEMY_DATABASE_URI", ""):
            db.create_all()

    return app
