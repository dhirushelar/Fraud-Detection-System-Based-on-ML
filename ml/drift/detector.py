"""
Fraud Detection System (FDS) - Data Drift Detection Module
Author: Alpha
Description: Calculates Population Stability Index (PSI) and Kolmogorov-Smirnov (KS)
             two-sample tests to detect covariate drift in financial transaction streams.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def calculate_psi(
    expected: np.ndarray | pd.Series | list,
    actual: np.ndarray | pd.Series | list,
    num_buckets: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """
    Computes Population Stability Index (PSI) between baseline (expected)
    and production inference (actual) continuous distributions.

    PSI Interpretation:
    - PSI < 0.10: Stable, negligible drift
    - 0.10 <= PSI < 0.25: Moderate shift, investigation recommended
    - PSI >= 0.25: Significant distribution drift, automated retrain required
    """
    exp_arr = np.asarray(expected, dtype=float)
    act_arr = np.asarray(actual, dtype=float)

    # Remove non-finite values if any
    exp_arr = exp_arr[np.isfinite(exp_arr)]
    act_arr = act_arr[np.isfinite(act_arr)]

    if len(exp_arr) == 0 or len(act_arr) == 0:
        return 0.0

    # Determine quantile bucket breakpoints based on expected baseline
    quantiles = np.linspace(0, 100, num_buckets + 1)
    bins = np.percentile(exp_arr, quantiles)
    # Ensure strictly increasing bins
    bins[0] = -np.inf
    bins[-1] = np.inf
    for i in range(1, len(bins) - 1):
        if bins[i] <= bins[i - 1]:
            bins[i] = bins[i - 1] + 1e-5

    # Compute observed counts per bucket
    exp_counts, _ = np.histogram(exp_arr, bins=bins)
    act_counts, _ = np.histogram(act_arr, bins=bins)

    # Convert to proportions with smoothing epsilon to prevent division by zero
    exp_pct = np.maximum(exp_counts / len(exp_arr), epsilon)
    act_pct = np.maximum(act_counts / len(act_arr), epsilon)

    # Renormalize
    exp_pct /= np.sum(exp_pct)
    act_pct /= np.sum(act_pct)

    # PSI summation
    psi_value = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(round(max(0.0, psi_value), 4))


def calculate_ks_test(
    expected: np.ndarray | pd.Series | list,
    actual: np.ndarray | pd.Series | list,
) -> Tuple[float, float]:
    """
    Performs Kolmogorov-Smirnov two-sample test for continuous distribution equality.
    Returns: (ks_statistic, p_value)
    """
    exp_arr = np.asarray(expected, dtype=float)
    act_arr = np.asarray(actual, dtype=float)
    exp_arr = exp_arr[np.isfinite(exp_arr)]
    act_arr = act_arr[np.isfinite(act_arr)]

    if len(exp_arr) < 5 or len(act_arr) < 5:
        return 0.0, 1.0

    res = stats.ks_2samp(exp_arr, act_arr)
    return float(round(res.statistic, 4)), float(round(res.pvalue, 6))


class DataDriftMonitor:
    """Evaluates multi-feature drift against a reference baseline dataset."""

    def __init__(
        self,
        reference_data: pd.DataFrame,
        numeric_features: Optional[List[str]] = None,
        psi_threshold: float = 0.25,
        ks_pvalue_threshold: float = 0.01,
    ) -> None:
        self.reference_data = reference_data
        self.numeric_features = numeric_features or [
            c for c in reference_data.columns if np.issubdtype(reference_data[c].dtype, np.number)
        ]
        self.psi_threshold = psi_threshold
        self.ks_pvalue_threshold = ks_pvalue_threshold

    def evaluate_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs PSI and KS checks across all tracked numeric features.
        Returns detailed per-feature telemetry and a global drift status flag.
        """
        feature_reports: List[Dict[str, Any]] = []
        drift_detected = False

        for feat in self.numeric_features:
            if feat not in current_data.columns:
                continue

            ref_vals = self.reference_data[feat].dropna().values
            curr_vals = current_data[feat].dropna().values

            psi = calculate_psi(ref_vals, curr_vals)
            ks_stat, ks_pval = calculate_ks_test(ref_vals, curr_vals)

            feat_drift = (psi >= self.psi_threshold) or (ks_pval < self.ks_pvalue_threshold and ks_stat > 0.15)
            if feat_drift:
                drift_detected = True

            feature_reports.append(
                {
                    "feature_name": feat,
                    "psi": psi,
                    "psi_drift": psi >= self.psi_threshold,
                    "ks_statistic": ks_stat,
                    "ks_pvalue": ks_pval,
                    "ks_drift": ks_pval < self.ks_pvalue_threshold,
                    "is_drift_detected": feat_drift,
                    "reference_mean": float(round(np.mean(ref_vals), 3)) if len(ref_vals) else 0.0,
                    "current_mean": float(round(np.mean(curr_vals), 3)) if len(curr_vals) else 0.0,
                }
            )

        return {
            "is_drift_detected": drift_detected,
            "psi_threshold": self.psi_threshold,
            "ks_pvalue_threshold": self.ks_pvalue_threshold,
            "total_features_evaluated": len(feature_reports),
            "drifted_features_count": sum(1 for r in feature_reports if r["is_drift_detected"]),
            "features": feature_reports,
            "recommendation": (
                "TRIGGER_RETRAINING: Significant covariate shift detected in production distribution."
                if drift_detected
                else "SYSTEM_STABLE: Feature distributions align with baseline."
            ),
        }
