"""
Fraud Detection System (FDS) - Hybrid Anomaly Scoring Ensemble
Author: Alpha
Description: Weighted ensemble combining Isolation Forest, Local Outlier Factor,
             and One-Class SVM scores to balance tree partitioning, local density,
             and boundary-based outlier detection.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ml.models.baselines import LOFFraudModel, OneClassSVMFraudModel
from ml.models.isolation_forest import IsolationForestFraudModel


class HybridAnomalyEnsemble:
    """
    Weighted ensemble of multiple anomaly detectors.

    Design Note & Architecture Decision:
    -----------------------------------
    While the ensemble aggregates multiple mathematical perspectives:
    1. Isolation Forest: Fast recursive tree partitioning (O(n log n)).
    2. Local Outlier Factor (LOF): Density-based k-nearest neighborhood.
    3. One-Class SVM: Non-linear hyper-spherical margin boundary.

    We select Isolation Forest as the PRIMARY production engine because:
    - LOF and OCSVM suffer from O(N^2) or O(N^3) complexity during fitting and
      high latency (> 25ms) on live scoring paths.
    - Isolation Forest scores in < 2ms per transaction, which satisfies strict
      payment gateway SLAs.
    - Isolation Forest maps cleanly to TreeSHAP for instant feature attribution.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        decision_threshold: float = 65.0,
    ) -> None:
        self.weights = weights or {
            "isolation_forest": 0.60,
            "lof": 0.25,
            "ocsvm": 0.15,
        }
        self.decision_threshold = decision_threshold

        self.if_model = IsolationForestFraudModel(decision_threshold=decision_threshold)
        self.lof_model = LOFFraudModel(decision_threshold=decision_threshold)
        self.ocsvm_model = OneClassSVMFraudModel(decision_threshold=decision_threshold)

        self.is_fitted_: bool = False

    def fit(self, X: np.ndarray | Any, y: Optional[Any] = None) -> "HybridAnomalyEnsemble":
        X_arr = np.asarray(X)
        self.if_model.fit(X_arr)
        self.lof_model.fit(X_arr)
        self.ocsvm_model.fit(X_arr)
        self.is_fitted_ = True
        return self

    def predict_risk_scores(self, X: np.ndarray | Any) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("Ensemble must be fitted before predicting.")

        X_arr = np.asarray(X)
        s_if = self.if_model.predict_risk_scores(X_arr)
        s_lof = self.lof_model.predict_risk_scores(X_arr)
        s_svm = self.ocsvm_model.predict_risk_scores(X_arr)

        w_if = self.weights.get("isolation_forest", 0.6)
        w_lof = self.weights.get("lof", 0.25)
        w_svm = self.weights.get("ocsvm", 0.15)
        total_w = w_if + w_lof + w_svm

        ensemble_score = (w_if * s_if + w_lof * s_lof + w_svm * s_svm) / total_w
        return np.clip(np.round(ensemble_score, 2), 0.0, 100.0)

    def predict(self, X: np.ndarray | Any) -> np.ndarray:
        scores = self.predict_risk_scores(X)
        return (scores >= self.decision_threshold).astype(int)
