"""
Fraud Detection System (FDS) - Auth Utilities & Role Decorators
Author: Alpha
"""

from functools import wraps
from flask import jsonify
from flask_jwt_extended import get_jwt, verify_jwt_in_request


def role_required(required_role: str):
    """
    Enforces JWT authentication and verifies that the user holds the required role.
    Role hierarchy: 'admin' has access to both 'admin' and 'analyst' endpoints.
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            verify_jwt_in_request()
            claims = get_jwt()
            user_role = claims.get("role", "analyst")

            if required_role == "admin" and user_role != "admin":
                return jsonify({
                    "error": "Forbidden",
                    "message": f"Administrator privileges required for this endpoint.",
                    "status_code": 403,
                }), 403

            return fn(*args, **kwargs)
        return wrapper
    return decorator
