"""
Fraud Detection System (FDS) - Authentication Pydantic Schemas
Author: Alpha
"""

from typing import Optional
from pydantic import BaseModel, Field


class LoginSchema(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)


class RegisterSchema(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., min_length=5, max_length=255, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    password: str = Field(..., min_length=8)
    role: str = Field(default="analyst", pattern="^(analyst|admin)$")


class TokenResponseSchema(BaseModel):
    access_token: str
    token_type: str = "Bearer"
    expires_in_minutes: int
    user_info: dict
