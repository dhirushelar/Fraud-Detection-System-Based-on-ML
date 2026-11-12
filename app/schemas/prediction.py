"""
Fraud Detection System (FDS) - Prediction Response Pydantic Schemas
Author: Alpha
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ReasonCodeSchema(BaseModel):
    code: str
    description: str
    impact: float


class PredictionResponseSchema(BaseModel):
    transaction_uid: str
    risk_score: float = Field(..., ge=0.0, le=100.0)
    raw_anomaly_score: float
    is_flagged: bool
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    decision_threshold: float
    model_version: str
    reason_codes: List[ReasonCodeSchema]
    latency_ms: float
    alert_uid: Optional[str] = None
