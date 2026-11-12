"""
Fraud Detection System (FDS) - Isolation Forest Anomaly Detection Model
Author: Alpha
Description: Tuned Isolation Forest implementation with monotonic sigmoid-based
             risk calibration (0.0 to 100.0), decision thresholding, and
             single/batch inference methods.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest


class IsolationForestFraudModel:
    """
    Primary Anomaly Detection Model utilizing an ensemble of Isolation Trees.
    Transforms raw tree path lengths into an interpretable 0-100 calibrated risk score.
    """

    def __init__(
        self,
        contamination: float = 0.005,
        n_estimators: int = 150,
        max_samples: int | str = 256,
        random_state: int = 42,
        decision_threshold: float = 65.0,
    ) -> None:
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.decision_threshold = decision_threshold

        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1,
        )

        # Calibration parameters (learned during fit)
        self.score_median_: float = 0.5
        self.score_scale_: float = 0.1
        self.is_fitted_: bool = False

    def fit(self, X: np.ndarray | Any, y: Optional[Any] = None) -> "IsolationForestFraudModel":
        """Fits the Isolation Forest on feature matrix X."""
        X_arr = np.asarray(X)
        self.model.fit(X_arr)

        # Calibrate risk scores based on training distribution
        raw_scores = -self.model.score_samples(X_arr)
        self.score_median_ = float(np.median(raw_scores))
        q75, q25 = np.percentile(raw_scores, [75, 25])
        iqr = max(1e-4, q75 - q25)
        self.score_scale_ = float(iqr)

        self.is_fitted_ = True
        return self

    def raw_anomaly_scores(self, X: np.ndarray | Any) -> np.ndarray:
        """Returns raw anomaly scores (higher = more anomalous)."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before scoring.")
        X_arr = np.asarray(X)
        return -self.model.score_samples(X_arr)

    def predict_risk_scores(self, X: np.ndarray | Any) -> np.ndarray:
        """
        Maps raw anomaly scores monotonically to [0.0, 100.0] risk range.
        Normal transactions fall around 10-40; anomalies surge towards 75-100.
        """
        raw = self.raw_anomaly_scores(X)
        z = (raw - self.score_median_) / (self.score_scale_ + 1e-6)
        # Sigmoid transfer centered slightly above median
        calibrated = 100.0 / (1.0 + np.exp(-1.8 * (z - 1.2)))
        return np.clip(np.round(calibrated, 2), 0.0, 100.0)

    def predict(self, X: np.ndarray | Any) -> np.ndarray:
        """Binary anomaly classification: 1 (Fraud Alert), 0 (Normal)."""
        risk_scores = self.predict_risk_scores(X)
        return (risk_scores >= self.decision_threshold).astype(int)

    def score_single(self, vector: np.ndarray) -> Tuple[float, float, bool]:
        """
        Low-latency scoring helper for real-time API.
        Returns: (raw_anomaly_score, calibrated_risk_score, is_flagged)
        """
        vec = np.asarray(vector).reshape(1, -1)
        raw = float(self.raw_anomaly_scores(vec)[0])
        risk = float(self.predict_risk_scores(vec)[0])
        flagged = bool(risk >= self.decision_threshold)
        return raw, risk, flagged

    def save(self, filepath: str) -> None:
        """Serializes fitted model and calibration metadata to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        payload = {
            "model": self.model,
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "random_state": self.random_state,
            "decision_threshold": self.decision_threshold,
            "score_median": self.score_median_,
            "score_scale": self.score_scale_,
            "is_fitted": self.is_fitted_,
        }
        joblib.dump(payload, filepath)

    @classmethod
    def load(cls, filepath: str) -> "IsolationForestFraudModel":
        """Deserializes model artifact from disk."""
        payload = joblib.load(filepath)
        instance = cls(
            contamination=payload["contamination"],
            n_estimators=payload["n_estimators"],
            max_samples=payload["max_samples"],
            random_state=payload["random_state"],
            decision_threshold=payload["decision_threshold"],
        )
        instance.model = payload["model"]
        instance.score_median_ = payload["score_median"]
        instance.score_scale_ = payload["score_scale"]
        instance.is_fitted_ = payload["is_fitted"]
        return instance
