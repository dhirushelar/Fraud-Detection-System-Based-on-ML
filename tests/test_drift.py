"""
Fraud Detection System (FDS) - Data Drift Detection Tests
Author: Alpha
"""

import numpy as np
import pandas as pd
import pytest

from ml.drift.detector import DataDriftMonitor, calculate_ks_test, calculate_psi


def test_psi_identical_distribution_is_stable():
    """Verify PSI is near zero for identically distributed data."""
    np.random.seed(42)
    expected = np.random.normal(loc=100.0, scale=20.0, size=2000)
    actual = np.random.normal(loc=100.0, scale=20.0, size=2000)

    psi = calculate_psi(expected, actual, num_buckets=10)
    assert psi < 0.10, f"Expected stable PSI (< 0.10), got {psi}"


def test_psi_shifted_distribution_detects_drift():
    """Verify PSI exceeds 0.25 threshold when mean shifts substantially."""
    np.random.seed(42)
    expected = np.random.normal(loc=100.0, scale=20.0, size=2000)
    # Substantial distribution shift (mean shifted by 2 std deviations)
    actual = np.random.normal(loc=140.0, scale=25.0, size=2000)

    psi = calculate_psi(expected, actual, num_buckets=10)
    assert psi >= 0.25, f"Expected drift PSI (>= 0.25), got {psi}"


def test_ks_test_significance():
    """Verify Kolmogorov-Smirnov test correctly computes p-values for distribution divergence."""
    np.random.seed(42)
    base = np.random.normal(loc=50.0, scale=10.0, size=1500)
    same = np.random.normal(loc=50.0, scale=10.0, size=1500)
    diverged = np.random.exponential(scale=50.0, size=1500)

    _, p_same = calculate_ks_test(base, same)
    stat_div, p_div = calculate_ks_test(base, diverged)

    assert p_same > 0.01, f"Expected non-significant p-value for same distribution, got {p_same}"
    assert p_div < 0.001, f"Expected highly significant p-value for diverged distribution, got {p_div}"
    assert stat_div > 0.15


def test_data_drift_monitor_workflow():
    """Verify end-to-end DataDriftMonitor generates structured telemetry and actionable alert."""
    np.random.seed(42)
    ref_df = pd.DataFrame({
        "amount": np.random.normal(80.0, 15.0, 1000),
        "tx_velocity_1h": np.random.poisson(1.0, 1000),
        "location_lat": np.random.normal(40.7128, 0.05, 1000),
    })

    # Production stream where amount has drifted significantly
    curr_df = pd.DataFrame({
        "amount": np.random.normal(160.0, 40.0, 1000),  # Drifted
        "tx_velocity_1h": np.random.poisson(1.0, 1000), # Stable
        "location_lat": np.random.normal(40.7128, 0.05, 1000), # Stable
    })

    monitor = DataDriftMonitor(
        reference_data=ref_df,
        numeric_features=["amount", "tx_velocity_1h", "location_lat"],
        psi_threshold=0.25,
        ks_pvalue_threshold=0.01,
    )

    report = monitor.evaluate_drift(curr_df)

    assert report["is_drift_detected"] is True
    assert report["total_features_evaluated"] == 3
    assert report["drifted_features_count"] >= 1
    assert "TRIGGER_RETRAINING" in report["recommendation"]

    # Verify amount specifically flagged
    amt_report = next(f for f in report["features"] if f["feature_name"] == "amount")
    assert amt_report["is_drift_detected"] is True
    assert amt_report["psi"] >= 0.25
