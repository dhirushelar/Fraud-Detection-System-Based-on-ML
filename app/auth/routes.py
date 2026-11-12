"""
Fraud Detection System (FDS) - Authentication Routes
Author: Alpha
"""

import datetime as dt
from flask import Blueprint, jsonify, request
from flask_jwt_extended import (
    create_access_token,
    get_jwt,
    get_jwt_identity,
    jwt_required,
)
from pydantic import ValidationError

from app.extensions import db
from app.models.user import AuthUser
from app.schemas.auth import LoginSchema, RegisterSchema

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")


@auth_bp.route("/register", methods=["POST"])
def register():
    """Registers a new fraud analyst or admin account."""
    payload = request.get_json(silent=True) or {}
    try:
        data = RegisterSchema(**payload)
    except ValidationError as err:
        return jsonify({"error": "Validation Error", "details": err.errors(), "status_code": 422}), 422

    if AuthUser.query.filter((AuthUser.username == data.username) | (AuthUser.email == data.email)).first():
        return jsonify({"error": "Conflict", "message": "Username or email already exists.", "status_code": 409}), 409

    new_user = AuthUser(
        username=data.username,
        email=data.email,
        role=data.role,
        is_active=True,
    )
    new_user.set_password(data.password)

    db.session.add(new_user)
    db.session.commit()

    return jsonify({
        "message": "User registered successfully.",
        "user": new_user.to_dict(),
        "status_code": 201,
    }), 201


@auth_bp.route("/login", methods=["POST"])
def login():
    """Authenticates analyst/admin credentials and issues a signed JWT token."""
    payload = request.get_json(silent=True) or {}
    try:
        data = LoginSchema(**payload)
    except ValidationError as err:
        return jsonify({"error": "Validation Error", "details": err.errors(), "status_code": 422}), 422

    user = AuthUser.query.filter_by(username=data.username).first()
    if not user or not user.check_password(data.password):
        return jsonify({"error": "Unauthorized", "message": "Invalid username or password.", "status_code": 401}), 401

    if not user.is_active:
        return jsonify({"error": "Forbidden", "message": "User account has been deactivated.", "status_code": 403}), 403

    # Generate JWT with custom role claim
    access_token = create_access_token(
        identity=str(user.id),
        additional_claims={
            "username": user.username,
            "role": user.role,
        },
    )

    return jsonify({
        "access_token": access_token,
        "token_type": "Bearer",
        "user": user.to_dict(),
        "status_code": 200,
    }), 200


@auth_bp.route("/me", methods=["GET"])
@jwt_required()
def me():
    """Returns profile information for the authenticated user."""
    user_id = int(get_jwt_identity())
    user = AuthUser.query.get(user_id)
    if not user:
        return jsonify({"error": "Not Found", "message": "User not found.", "status_code": 404}), 404

    return jsonify({"user": user.to_dict(), "status_code": 200}), 200
