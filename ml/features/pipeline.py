"""
Fraud Detection System (FDS) - Feature Engineering Pipeline
Author: Alpha
Description: Complete Sklearn Pipeline combining temporal, behavioral, and
             contextual transformers with RobustScaler. Supports both batch
             dataframe processing and sub-millisecond single-transaction enrichment.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from ml.features.preprocessor import (
    BehavioralFeatureExtractor,
    ContextFeatureExtractor,
    TemporalFeatureExtractor,
    calculate_haversine_distance,
)


class FraudFeatureEngineeringPipeline(BaseEstimator, TransformerMixin):
    """
    High-level composite pipeline that transforms raw transaction logs into
    scaled, model-ready anomaly feature vectors.
    """

    def __init__(self, with_scaling: bool = True) -> None:
        self.with_scaling = with_scaling
        self.temporal_extractor = TemporalFeatureExtractor()
        self.context_extractor = ContextFeatureExtractor()
        self.behavioral_extractor = BehavioralFeatureExtractor()
        self.scaler = RobustScaler()

        self.feature_columns_: List[str] = []
        self.is_fitted_: bool = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FraudFeatureEngineeringPipeline":
        # Fit sub-extractors
        self.temporal_extractor.fit(X, y)
        self.context_extractor.fit(X, y)
        self.behavioral_extractor.fit(X, y)

        # Generate unscaled features to fit the scaler
        feat_df = self._extract_all_features(X)
        self.feature_columns_ = list(feat_df.columns)

        if self.with_scaling:
            self.scaler.fit(feat_df.values)

        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Pipeline must be fitted before calling transform.")

        feat_df = self._extract_all_features(X)

        if self.with_scaling:
            scaled_vals = self.scaler.transform(feat_df.values)
            return pd.DataFrame(scaled_vals, columns=self.feature_columns_, index=X.index)

        return feat_df

    def _extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Internal helper to construct full feature matrix from sub-extractors."""
        temp_df = self.temporal_extractor.transform(df)
        cont_df = self.context_extractor.transform(df)
        behav_df = self.behavioral_extractor.transform(df)

        # Raw amount features
        raw_amount = df["amount"].astype(float).values
        log_amount = np.log1p(np.maximum(0.0, raw_amount))

        core_df = pd.DataFrame(
            {
                "amount": raw_amount,
                "log_amount": log_amount,
            },
            index=df.index,
        )

        combined = pd.concat([core_df, temp_df, cont_df, behav_df], axis=1)
        return combined

    def transform_single(
        self,
        tx: Dict[str, Any],
        user_history: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Fast real-time transformation for a single incoming transaction dict.
        Combines current payload with historical state fetched from DB.
        Returns:
            scaled_array: 1xP numpy array ready for model inference
            raw_features: dict of unscaled feature values for explanations / snapshots
        """
        user_history = user_history or {}

        # 1. Parse timestamp and temporal features
        ts = pd.to_datetime(tx["timestamp"], utc=True)
        hour = ts.hour
        day_of_week = ts.dayofweek
        is_night = 1 if (0 <= hour <= 5) else 0
        is_weekend = 1 if (day_of_week >= 5) else 0

        sin_hour = float(np.sin(2.0 * np.pi * hour / 24.0))
        cos_hour = float(np.cos(2.0 * np.pi * hour / 24.0))
        sin_day = float(np.sin(2.0 * np.pi * day_of_week / 7.0))
        cos_day = float(np.cos(2.0 * np.pi * day_of_week / 7.0))

        # 2. Context features
        lat = float(tx.get("location_lat", 0.0))
        lon = float(tx.get("location_lon", 0.0))
        last_lat = user_history.get("last_lat")
        last_lon = user_history.get("last_lon")

        if last_lat is not None and last_lon is not None:
            distance_km = float(calculate_haversine_distance(last_lat, last_lon, lat, lon))
        else:
            distance_km = 0.0

        known_devices = set(user_history.get("known_devices", []))
        known_merchants = set(user_history.get("known_merchants", []))

        is_new_device = 1 if (tx.get("device_id") not in known_devices) else 0
        is_new_merchant = 1 if (tx.get("merchant_id") not in known_merchants) else 0

        # 3. Behavioral features
        amount = float(tx["amount"])
        log_amount = float(np.log1p(max(0.0, amount)))

        velocity_1h = int(user_history.get("tx_velocity_1h", 0))
        velocity_24h = int(user_history.get("tx_velocity_24h", 0))

        rolling_mean = float(user_history.get("user_rolling_mean_amount", self.behavioral_extractor.global_mean_amount_))
        rolling_std = float(user_history.get("user_rolling_std_amount", self.behavioral_extractor.global_std_amount_))

        amount_zscore = float((amount - rolling_mean) / (rolling_std + 1e-4))
        amount_to_mean_ratio = float(amount / (rolling_mean + 1e-4))

        raw_dict = {
            "amount": amount,
            "log_amount": log_amount,
            "hour": float(hour),
            "day_of_week": float(day_of_week),
            "is_night": float(is_night),
            "is_weekend": float(is_weekend),
            "sin_hour": sin_hour,
            "cos_hour": cos_hour,
            "sin_day": sin_day,
            "cos_day": cos_day,
            "distance_from_last_tx_km": round(distance_km, 3),
            "is_new_device": float(is_new_device),
            "is_new_merchant": float(is_new_merchant),
            "tx_velocity_1h": float(velocity_1h),
            "tx_velocity_24h": float(velocity_24h),
            "user_rolling_mean_amount": round(rolling_mean, 2),
            "user_rolling_std_amount": round(rolling_std, 2),
            "amount_zscore": round(amount_zscore, 3),
            "amount_to_mean_ratio": round(amount_to_mean_ratio, 3),
        }

        # Build ordered vector
        ordered_vals = [raw_dict[col] for col in self.feature_columns_]
        vec = np.array([ordered_vals], dtype=float)

        if self.with_scaling and self.is_fitted_:
            scaled_vec = self.scaler.transform(vec)
            return scaled_vec, raw_dict

        return vec, raw_dict

    def get_feature_names(self) -> List[str]:
        return list(self.feature_columns_)
