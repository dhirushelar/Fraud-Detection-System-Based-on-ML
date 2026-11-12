"""
Fraud Detection System (FDS) - Alert Schemas
Author: Alpha
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class AlertUpdateSchema(BaseModel):
    status: str = Field(
        ...,
        pattern="^(OPEN|INVESTIGATING|RESOLVED_FRAUD|RESOLVED_FALSE_POSITIVE|DISMISSED)$",
    )
    feedback_verdict: Optional[str] = Field(
        None,
        pattern="^(TRUE_FRAUD|FALSE_POSITIVE|SUSPICIOUS_UNCONFIRMED)$",
    )
    reason_tags: Optional[List[str]] = Field(default=[])
    notes: Optional[str] = Field(None, max_length=2000)
