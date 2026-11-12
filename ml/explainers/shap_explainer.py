"""
Fraud Detection System (FDS) - SHAP Explainer & Reason Code Generator
Author: Alpha
Description: Computes local feature attribution via TreeSHAP on Isolation Forest
             models and synthesizes top human-readable reason codes for fraud analysts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap

from ml.models.isolation_forest import IsolationForestFraudModel


class FraudSHAPExplainer:
    """
    Computes local feature attributions using TreeSHAP on Isolation Forest estimators.
    Translates raw attribution values into actionable English explanation codes.
    """

    def __init__(
        self,
        model: IsolationForestFraudModel,
        feature_names: List[str],
        background_samples: Optional[np.ndarray] = None,
    ) -> None:
        self.model = model
        self.feature_names = feature_names

        # TreeExplainer natively inspects sklearn IsolationForest decision trees
        try:
            self.explainer = shap.TreeExplainer(model.model)
        except Exception:
            # Fallback with background data if needed
            if background_samples is not None:
                self.explainer = shap.Explainer(model.model.score_samples, background_samples)
            else:
                self.explainer = None

    def explain_transaction(
        self,
        scaled_vector: np.ndarray,
        raw_features: Dict[str, Any],
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Calculates local SHAP values and derives top-k analyst reason codes.
        Returns:
            dict containing shap_values_dict, top_reason_codes, and explanation_summary.
        """
        vec = np.asarray(scaled_vector).reshape(1, -1)

        # 1. Compute SHAP values
        if self.explainer is not None:
            shap_vals = self.explainer.shap_values(vec)
            if isinstance(shap_vals, list):
                # Multi-output handle
                shap_vals = shap_vals[0]
            if len(shap_vals.shape) > 1:
                shap_vals = shap_vals[0]
        else:
            # Heuristic attribution fallback based on standardized deviance
            shap_vals = np.abs(vec[0])

        # Note: In Isolation Forest, shorter path length means anomaly.
        # We invert so positive SHAP values indicate contribution toward ANOMALY.
        anomaly_contributions = -1.0 * np.asarray(shap_vals)

        # 2. Map contributions to feature names
        contributions_dict = {
            name: float(round(anomaly_contributions[i], 4))
            for i, name in enumerate(self.feature_names)
            if i < len(anomaly_contributions)
        }

        # 3. Sort features by highest anomalous contribution
        sorted_feats = sorted(
            contributions_dict.items(), key=lambda x: x[1], reverse=True
        )

        # 4. Generate contextual, plain-English reason codes
        reason_codes: List[Dict[str, Any]] = []
        for feat_name, impact in sorted_feats[:top_k]:
            reason = self._synthesize_reason_code(feat_name, impact, raw_features)
            if reason:
                reason_codes.append(reason)

        return {
            "feature_attributions": contributions_dict,
            "sorted_contributions": [
                {"feature": k, "impact": v} for k, v in sorted_feats
            ],
            "reason_codes": reason_codes,
        }

    def _synthesize_reason_code(
        self,
        feature_name: str,
        impact_score: float,
        raw: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Translates an anomalous feature signal into a concrete analyst narrative."""
        amount = raw.get("amount", 0.0)
        zscore = raw.get("amount_zscore", 0.0)
        mean_amt = raw.get("user_rolling_mean_amount", 50.0)
        dist_km = raw.get("distance_from_last_tx_km", 0.0)
        v1h = int(raw.get("tx_velocity_1h", 0))
        v24h = int(raw.get("tx_velocity_24h", 0))
        hour = int(raw.get("hour", 12))
        is_night = int(raw.get("is_night", 0))
        new_dev = int(raw.get("is_new_device", 0))
        new_mcht = int(raw.get("is_new_merchant", 0))

        if feature_name in ["amount", "log_amount", "amount_zscore", "amount_to_mean_ratio"]:
            multiple = round(amount / max(1.0, mean_amt), 1)
            msg = f"Transaction amount (${amount:,.2f}) is {multiple}x higher than user average (${mean_amt:,.2f}) with Z-score {zscore:.1f}."
            return {"code": "ABNORMAL_HIGH_AMOUNT", "description": msg, "impact": impact_score}

        if feature_name in ["tx_velocity_1h", "tx_velocity_24h"]:
            msg = f"High velocity burst: {v1h} transactions in the last hour ({v24h} in 24 hours)."
            return {"code": "VELOCITY_BURST", "description": msg, "impact": impact_score}

        if feature_name == "distance_from_last_tx_km" and dist_km > 50.0:
            msg = f"Geographic anomaly: Transaction located {dist_km:,.1f} km away from recent activity."
            return {"code": "GEODETIC_DISPARITY", "description": msg, "impact": impact_score}

        if feature_name == "is_new_device" and new_dev == 1:
            msg = "Transaction initiated from an unrecognized device ID never before used by this account."
            return {"code": "UNRECOGNIZED_DEVICE", "description": msg, "impact": impact_score}

        if feature_name in ["is_night", "hour", "sin_hour", "cos_hour"] and is_night == 1:
            msg = f"Transaction executed at odd hour ({hour:02d}:00), outside the account holder's customary schedule."
            return {"code": "ODD_HOUR_TRANSACTION", "description": msg, "impact": impact_score}

        if feature_name == "is_new_merchant" and new_mcht == 1:
            msg = "First interaction with this merchant, deviating from customary vendor history."
            return {"code": "NOVEL_MERCHANT", "description": msg, "impact": impact_score}

        # Generic descriptive fallback
        val = raw.get(feature_name, "N/A")
        return {
            "code": f"UNUSUAL_{feature_name.upper()}",
            "description": f"Feature '{feature_name}' exhibited an anomalous reading of {val}.",
            "impact": impact_score,
        }
