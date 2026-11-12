"""
Fraud Detection System (FDS) - API Error Handlers
Author: Alpha
"""

import datetime as dt
from flask import jsonify


def register_error_handlers(app):
    """Registers structured JSON error responses across the Flask application."""

    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({
            "error": "Bad Request",
            "message": str(error.description if hasattr(error, "description") else error),
            "status_code": 400,
            "timestamp": dt.datetime.utcnow().isoformat(),
        }), 400

    @app.errorhandler(401)
    def unauthorized(error):
        return jsonify({
            "error": "Unauthorized",
            "message": "Authentication required to access this resource.",
            "status_code": 401,
            "timestamp": dt.datetime.utcnow().isoformat(),
        }), 401

    @app.errorhandler(403)
    def forbidden(error):
        return jsonify({
            "error": "Forbidden",
            "message": "You do not have permission to execute this action.",
            "status_code": 403,
            "timestamp": dt.datetime.utcnow().isoformat(),
        }), 403

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            "error": "Not Found",
            "message": "The requested resource was not found on the server.",
            "status_code": 404,
            "timestamp": dt.datetime.utcnow().isoformat(),
        }), 404

    @app.errorhandler(429)
    def ratelimit_handler(error):
        return jsonify({
            "error": "Too Many Requests",
            "message": "Rate limit exceeded. Please throttle your transaction volume.",
            "status_code": 429,
            "timestamp": dt.datetime.utcnow().isoformat(),
        }), 429

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({
            "error": "Internal Server Error",
            "message": "An unexpected server error occurred. Telemetry has been logged.",
            "status_code": 500,
            "timestamp": dt.datetime.utcnow().isoformat(),
        }), 500
