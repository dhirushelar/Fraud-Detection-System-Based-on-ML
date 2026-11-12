"""
Fraud Detection System (FDS) - Auth Package
Author: Alpha
"""

from app.auth.routes import auth_bp
from app.auth.utils import role_required

__all__ = ["auth_bp", "role_required"]
