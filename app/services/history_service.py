"""
Fraud Detection System (FDS) - User History Service
Author: Alpha
Description: Low-latency historical user lookup to feed the dynamic feature engineering pipeline.
"""

import datetime as dt
from typing import Any, Dict, List, Optional
import numpy as np

from app.extensions import db
from app.models.transaction import Transaction
from app.models.user import User


def _to_naive_utc(d: dt.datetime) -> dt.datetime:
    """Ensure datetime is offset-naive UTC for consistent cross-database comparisons."""
    if d is None:
        return d
    if d.tzinfo is not None:
        return d.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return d


class HistoryService:
    """Extracts historical transactional context for a given account holder."""

    @staticmethod
    def get_user_history_snapshot(
        user_id: int,
        current_time: dt.datetime,
        lookback_days: int = 30,
    ) -> Dict[str, Any]:
        """
        Retrieves user historical statistics required for real-time feature engineering.
        """
        current_time = _to_naive_utc(current_time)
        cutoff = current_time - dt.timedelta(days=lookback_days)

        # Query past transactions for this user strictly before current_time
        past_txs = (
            Transaction.query.filter(
                Transaction.user_id == user_id,
                Transaction.timestamp < current_time,
                Transaction.timestamp >= cutoff,
            )
            .order_by(Transaction.timestamp.desc())
            .all()
        )

        if not past_txs:
            return {
                "tx_velocity_1h": 0,
                "tx_velocity_24h": 0,
                "user_rolling_mean_amount": 65.0,
                "user_rolling_std_amount": 25.0,
                "last_lat": None,
                "last_lon": None,
                "known_devices": [],
                "known_merchants": [],
            }

        t_1h = current_time - dt.timedelta(hours=1)
        t_24h = current_time - dt.timedelta(hours=24)

        v_1h = sum(1 for tx in past_txs if _to_naive_utc(tx.timestamp) >= t_1h)
        v_24h = sum(1 for tx in past_txs if _to_naive_utc(tx.timestamp) >= t_24h)

        amounts = [float(tx.amount) for tx in past_txs]
        mean_amt = float(np.mean(amounts))
        std_amt = float(np.std(amounts)) if len(amounts) > 1 else 25.0

        most_recent = past_txs[0]
        known_devices = list({tx.device_id for tx in past_txs})
        known_merchants = list({str(tx.merchant_id) for tx in past_txs})

        return {
            "tx_velocity_1h": v_1h,
            "tx_velocity_24h": v_24h,
            "user_rolling_mean_amount": round(mean_amt, 2),
            "user_rolling_std_amount": round(max(5.0, std_amt), 2),
            "last_lat": most_recent.location_lat,
            "last_lon": most_recent.location_lon,
            "known_devices": known_devices,
            "known_merchants": known_merchants,
        }
