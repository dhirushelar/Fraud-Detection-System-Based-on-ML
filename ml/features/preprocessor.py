"""
Fraud Detection System (FDS) - Feature Engineering Preprocessors
Author: Alpha
Description: Scikit-learn compliant feature transformers for time, behavioral
             velocity, rolling stats, geodetic distance, and novelty signals.
             Guarantees zero data leakage via sequential history accumulation.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def calculate_haversine_distance(
    lat1: np.ndarray | float,
    lon1: np.ndarray | float,
    lat2: np.ndarray | float,
    lon2: np.ndarray | float,
) -> np.ndarray | float:
    """Vectorized Haversine distance in kilometers between two GPS coordinates."""
    r = 6371.0  # Earth's mean radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a)))
    return r * c


class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts calendar, day-of-week, hour-of-day, and cyclical trigonometric features."""

    def __init__(self, timestamp_col: str = "timestamp") -> None:
        self.timestamp_col = timestamp_col
        self.feature_names_: List[str] = [
            "hour",
            "day_of_week",
            "is_night",
            "is_weekend",
            "sin_hour",
            "cos_hour",
            "sin_day",
            "cos_day",
        ]

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "TemporalFeatureExtractor":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[self.timestamp_col]):
            timestamps = pd.to_datetime(df[self.timestamp_col], utc=True)
        else:
            timestamps = df[self.timestamp_col]

        hour = timestamps.dt.hour.values
        day_of_week = timestamps.dt.dayofweek.values

        # Night definition: 00:00 - 05:59 inclusive
        is_night = ((hour >= 0) & (hour <= 5)).astype(int)
        # Weekend definition: Saturday (5) or Sunday (6)
        is_weekend = (day_of_week >= 5).astype(int)

        # Continuous periodic representations
        sin_hour = np.sin(2.0 * np.pi * hour / 24.0)
        cos_hour = np.cos(2.0 * np.pi * hour / 24.0)
        sin_day = np.sin(2.0 * np.pi * day_of_week / 7.0)
        cos_day = np.cos(2.0 * np.pi * day_of_week / 7.0)

        out = pd.DataFrame(
            {
                "hour": hour,
                "day_of_week": day_of_week,
                "is_night": is_night,
                "is_weekend": is_weekend,
                "sin_hour": sin_hour,
                "cos_hour": cos_hour,
                "sin_day": sin_day,
                "cos_day": cos_day,
            },
            index=df.index,
        )
        return out

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        return self.feature_names_


class ContextFeatureExtractor(BaseEstimator, TransformerMixin):
    """Computes geodetic movement and device/merchant novelty relative to past history."""

    def __init__(
        self,
        user_col: str = "user_id",
        timestamp_col: str = "timestamp",
        lat_col: str = "location_lat",
        lon_col: str = "location_lon",
        device_col: str = "device_id",
        merchant_col: str = "merchant_id",
    ) -> None:
        self.user_col = user_col
        self.timestamp_col = timestamp_col
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.device_col = device_col
        self.merchant_col = merchant_col
        self.feature_names_: List[str] = [
            "distance_from_last_tx_km",
            "is_new_device",
            "is_new_merchant",
        ]

        # Learned state from training set
        self.known_user_devices_: Dict[str, Set[str]] = {}
        self.known_user_merchants_: Dict[str, Set[str]] = {}
        self.user_last_locations_: Dict[str, Tuple[float, float]] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "ContextFeatureExtractor":
        self.known_user_devices_ = {}
        self.known_user_merchants_ = {}
        self.user_last_locations_ = {}

        df = X.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[self.timestamp_col]):
            df["_parsed_ts"] = pd.to_datetime(df[self.timestamp_col], utc=True)
        else:
            df["_parsed_ts"] = df[self.timestamp_col]

        df = df.sort_values(by="_parsed_ts")

        for _, row in df.iterrows():
            uid = str(row[self.user_col])
            dev = str(row[self.device_col])
            mcht = str(row[self.merchant_col])
            lat = float(row[self.lat_col])
            lon = float(row[self.lon_col])

            if uid not in self.known_user_devices_:
                self.known_user_devices_[uid] = set()
                self.known_user_merchants_[uid] = set()

            self.known_user_devices_[uid].add(dev)
            self.known_user_merchants_[uid].add(mcht)
            self.user_last_locations_[uid] = (lat, lon)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[self.timestamp_col]):
            df["_parsed_ts"] = pd.to_datetime(df[self.timestamp_col], utc=True)
        else:
            df["_parsed_ts"] = df[self.timestamp_col]

        # Working state copies to handle streaming or sequential evaluation
        running_devices: Dict[str, Set[str]] = {
            k: set(v) for k, v in self.known_user_devices_.items()
        }
        running_merchants: Dict[str, Set[str]] = {
            k: set(v) for k, v in self.known_user_merchants_.items()
        }
        running_locations: Dict[str, Tuple[float, float]] = dict(
            self.user_last_locations_
        )

        distances: List[float] = []
        is_new_device: List[int] = []
        is_new_merchant: List[int] = []

        # Iterate strictly in time order to prevent forward data leakage
        for _, row in df.iterrows():
            uid = str(row[self.user_col])
            dev = str(row[self.device_col])
            mcht = str(row[self.merchant_col])
            lat = float(row[self.lat_col])
            lon = float(row[self.lon_col])

            # Distance computation
            if uid in running_locations:
                last_lat, last_lon = running_locations[uid]
                dist = float(calculate_haversine_distance(last_lat, last_lon, lat, lon))
            else:
                dist = 0.0
            distances.append(round(dist, 3))
            running_locations[uid] = (lat, lon)

            # Novelty detection
            if uid in running_devices:
                dev_novel = 1 if dev not in running_devices[uid] else 0
                mcht_novel = 1 if mcht not in running_merchants[uid] else 0
            else:
                dev_novel = 1
                mcht_novel = 1
                running_devices[uid] = set()
                running_merchants[uid] = set()

            is_new_device.append(dev_novel)
            is_new_merchant.append(mcht_novel)

            # Update running user state
            running_devices[uid].add(dev)
            running_merchants[uid].add(mcht)

        out = pd.DataFrame(
            {
                "distance_from_last_tx_km": distances,
                "is_new_device": is_new_device,
                "is_new_merchant": is_new_merchant,
            },
            index=df.index,
        )
        return out

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        return self.feature_names_


class BehavioralFeatureExtractor(BaseEstimator, TransformerMixin):
    """Computes transaction velocity and historical rolling statistics per user."""

    def __init__(
        self,
        user_col: str = "user_id",
        amount_col: str = "amount",
        timestamp_col: str = "timestamp",
        epsilon: float = 1e-4,
    ) -> None:
        self.user_col = user_col
        self.amount_col = amount_col
        self.timestamp_col = timestamp_col
        self.epsilon = epsilon
        self.feature_names_: List[str] = [
            "tx_velocity_1h",
            "tx_velocity_24h",
            "user_rolling_mean_amount",
            "user_rolling_std_amount",
            "amount_zscore",
            "amount_to_mean_ratio",
        ]
        self.global_mean_amount_: float = 50.0
        self.global_std_amount_: float = 30.0

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "BehavioralFeatureExtractor":
        amounts = X[self.amount_col].astype(float)
        self.global_mean_amount_ = float(amounts.mean())
        self.global_std_amount_ = float(amounts.std() or 20.0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[self.timestamp_col]):
            df["_parsed_ts"] = pd.to_datetime(df[self.timestamp_col], utc=True)
        else:
            df["_parsed_ts"] = df[self.timestamp_col]

        # Store epoch in seconds for fast time-window calculations
        df["_epoch_sec"] = df["_parsed_ts"].astype("int64") // 10**9

        velocity_1h: List[int] = []
        velocity_24h: List[int] = []
        rolling_means: List[float] = []
        rolling_stds: List[float] = []
        z_scores: List[float] = []
        ratios: List[float] = []

        # Maintain past transactions per user: list of (epoch_sec, amount)
        user_history: Dict[str, List[Tuple[int, float]]] = {}

        for _, row in df.iterrows():
            uid = str(row[self.user_col])
            amt = float(row[self.amount_col])
            t_curr = int(row["_epoch_sec"])

            if uid not in user_history:
                user_history[uid] = []

            past = user_history[uid]

            # Velocity in last 1 hour (3600s) and last 24 hours (86400s)
            c_1h = sum(1 for (t_past, _) in past if (t_curr - t_past) <= 3600)
            c_24h = sum(1 for (t_past, _) in past if (t_curr - t_past) <= 86400)

            # Historical amounts for rolling statistics
            if past:
                past_amts = [p_amt for (_, p_amt) in past]
                mean_val = float(np.mean(past_amts))
                std_val = float(np.std(past_amts)) if len(past_amts) > 1 else self.global_std_amount_
            else:
                mean_val = self.global_mean_amount_
                std_val = self.global_std_amount_

            z = (amt - mean_val) / (std_val + self.epsilon)
            ratio = amt / (mean_val + self.epsilon)

            velocity_1h.append(c_1h)
            velocity_24h.append(c_24h)
            rolling_means.append(round(mean_val, 2))
            rolling_stds.append(round(std_val, 2))
            z_scores.append(round(z, 3))
            ratios.append(round(ratio, 3))

            # Append current event to user's history for subsequent rows
            past.append((t_curr, amt))

        out = pd.DataFrame(
            {
                "tx_velocity_1h": velocity_1h,
                "tx_velocity_24h": velocity_24h,
                "user_rolling_mean_amount": rolling_means,
                "user_rolling_std_amount": rolling_stds,
                "amount_zscore": z_scores,
                "amount_to_mean_ratio": ratios,
            },
            index=df.index,
        )
        return out

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        return self.feature_names_
