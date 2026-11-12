"""
Fraud Detection System (FDS) - Transaction Ingestion Pydantic Schemas
Author: Alpha
"""

import datetime as dt
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class TransactionIngestSchema(BaseModel):
    transaction_uid: str = Field(..., min_length=4, max_length=64)
    user_external_id: str = Field(..., min_length=3, max_length=64)
    merchant_external_id: str = Field(..., min_length=3, max_length=64)
    amount: float = Field(..., gt=0.0, description="Amount must be positive.")
    currency: str = Field(default="USD", min_length=3, max_length=3)
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp")
    location_lat: Optional[float] = Field(None, ge=-90.0, le=90.0)
    location_lon: Optional[float] = Field(None, ge=-180.0, le=180.0)
    device_id: str = Field(..., min_length=3, max_length=128)
    ip_address: Optional[str] = Field(None, max_length=45)
    channel: str = Field(default="ONLINE", pattern="^(ONLINE|POS|ATM|MOBILE)$")

    @field_validator("timestamp")
    @classmethod
    def validate_iso_timestamp(cls, v: str) -> str:
        try:
            # Parses standard ISO formats including Z and offset
            dt.datetime.fromisoformat(v.replace("Z", "+00:00"))
            return v
        except Exception:
            raise ValueError("Timestamp must be a valid ISO 8601 formatted datetime string.")


class BatchTransactionIngestSchema(BaseModel):
    transactions: List[TransactionIngestSchema] = Field(..., min_length=1, max_length=5000)
