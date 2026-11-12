"""
Fraud Detection System (FDS) - Baseline Anomaly Detection Models
Author: Alpha
Description: Local Outlier Factor (LOF) and One-Class SVM baseline models
             for benchmark comparison against the primary Isolation Forest.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


class LOFFraudModel:
    """Local Outlier Factor (LOF) baseline with novelty detection enabled."""

    def __init__(
        self,
        n_neighbors: int = 25,
        contamination: float = 0.005,
        decision_threshold: float = 65.0,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.decision_threshold = decision_threshold

        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True,
            n_jobs=-1,
        )
        self.score_median_: float = 0.0
        self.score_scale_: float = 1.0
        self.is_fitted_: bool = False

    def fit(self, X: np.ndarray | Any, y: Optional[Any] = None) -> "LOFFraudModel":
        X_arr = np.asarray(X)
        self.model.fit(X_arr)
        raw = -self.model.decision_function(X_arr)
        self.score_median_ = float(np.median(raw))
        q75, q25 = np.percentile(raw, [75, 25])
        self.score_scale_ = float(max(1e-4, q75 - q25))
        self.is_fitted_ = True
        return self

    def raw_anomaly_scores(self, X: np.ndarray | Any) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("LOF model must be fitted first.")
        return -self.model.decision_function(np.asarray(X))

    def predict_risk_scores(self, X: np.ndarray | Any) -> np.ndarray:
        raw = self.raw_anomaly_scores(X)
        z = (raw - self.score_median_) / (self.score_scale_ + 1e-6)
        calibrated = 100.0 / (1.0 + np.exp(-1.8 * (z - 1.2)))
        return np.clip(np.round(calibrated, 2), 0.0, 100.0)

    def predict(self, X: np.ndarray | Any) -> np.ndarray:
        risk = self.predict_risk_scores(X)
        return (risk >= self.decision_threshold).astype(int)


class OneClassSVMFraudModel:
    """One-Class Support Vector Machine (OC-SVM) with RBF kernel baseline."""

    def __init__(
        self,
        kernel: str = "rbf",
        gamma: str | float = "scale",
        nu: float = 0.01,
        decision_threshold: float = 65.0,
    ) -> None:
        self.kernel = kernel
        self.gamma = gamma
        self.nu = nu
        self.decision_threshold = decision_threshold

        self.model = OneClassSVM(
            kernel=kernel,
            gamma=gamma,
            nu=nu,
        )
        self.score_median_: float = 0.0
        self.score_scale_: float = 1.0
        self.is_fitted_: bool = False

    def fit(self, X: np.ndarray | Any, y: Optional[Any] = None) -> "OneClassSVMFraudModel":
        X_arr = np.asarray(X)
        self.model.fit(X_arr)
        raw = -self.model.decision_function(X_arr)
        self.score_median_ = float(np.median(raw))
        q75, q25 = np.percentile(raw, [75, 25])
        self.score_scale_ = float(max(1e-4, q75 - q25))
        self.is_fitted_ = True
        return self

    def raw_anomaly_scores(self, X: np.ndarray | Any) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("One-Class SVM must be fitted first.")
        return -self.model.decision_function(np.asarray(X))

    def predict_risk_scores(self, X: np.ndarray | Any) -> np.ndarray:
        raw = self.raw_anomaly_scores(X)
        z = (raw - self.score_median_) / (self.score_scale_ + 1e-6)
        calibrated = 100.0 / (1.0 + np.exp(-1.8 * (z - 1.2)))
        return np.clip(np.round(calibrated, 2), 0.0, 100.0)

    def predict(self, X: np.ndarray | Any) -> np.ndarray:
        risk = self.predict_risk_scores(X)
        return (risk >= self.decision_threshold).astype(int)
