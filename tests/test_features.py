"""
Fraud Detection System (FDS) - Feature Engineering & Data Pipeline Tests
Author: Alpha
"""

import datetime as dt
import numpy as np
import pandas as pd
import pytest

from ml.data_generator import (
    TransactionGenerator,
    generate_dataset,
    haversine_distance,
)
from ml.data_loader import time_based_train_test_split
from ml.features.pipeline import FraudFeatureEngineeringPipeline
from ml.features.preprocessor import (
    BehavioralFeatureExtractor,
    ContextFeatureExtractor,
    TemporalFeatureExtractor,
    calculate_haversine_distance,
)


@pytest.fixture
def sample_raw_dataframe() -> pd.DataFrame:
    """Fixture providing a deterministic small dataset for unit tests."""
    return generate_dataset(n_transactions=500, fraud_rate=0.04, random_state=42)


def test_data_generator_schema_and_distribution():
    """Verify data generator produces valid schema, expected columns, and fraud archetypes."""
    df = generate_dataset(n_transactions=1000, fraud_rate=0.01, random_state=123)

    expected_cols = [
        "transaction_id",
        "user_id",
        "merchant_id",
        "merchant_category",
        "amount",
        "currency",
        "timestamp",
        "location_lat",
        "location_lon",
        "device_id",
        "ip_address",
        "channel",
        "is_fraud",
        "fraud_type",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing expected column: {col}"

    # Fraud rate sanity (around 1%)
    fraud_count = df["is_fraud"].sum()
    assert 5 <= fraud_count <= 25, f"Unexpected fraud count: {fraud_count}"

    # Verify fraud archetypes exist
    fraud_types = set(df[df["is_fraud"] == 1]["fraud_type"].unique())
    assert "velocity_burst" in fraud_types or "odd_hour_high_amount" in fraud_types

    # Verify chronological ordering
    ts_series = pd.to_datetime(df["timestamp"])
    assert ts_series.is_monotonic_increasing, "Transactions are not strictly sorted by timestamp."


def test_haversine_distance_accuracy():
    """Verify Haversine formula calculation against known geographical coordinates."""
    # New York City (40.7128, -74.0060) to Los Angeles (34.0522, -118.2437) ~ 3935 km
    dist = haversine_distance(40.7128, -74.0060, 34.0522, -118.2437)
    assert 3900.0 < dist < 4000.0

    # Same location distance must be 0
    zero_dist = haversine_distance(40.7128, -74.0060, 40.7128, -74.0060)
    assert round(zero_dist, 4) == 0.0

    # Vectorized version consistency
    arr_dist = calculate_haversine_distance(
        np.array([40.7128]),
        np.array([-74.0060]),
        np.array([34.0522]),
        np.array([-118.2437]),
    )
    assert abs(arr_dist[0] - dist) < 1e-4


def test_temporal_feature_extractor():
    """Verify temporal extractor correctly extracts time flags and cyclical encodings."""
    dates = [
        "2026-03-01T02:15:00Z",  # Sunday (weekend=1), 02:00 (night=1)
        "2026-03-02T14:30:00Z",  # Monday (weekend=0), 14:00 (night=0)
    ]
    df = pd.DataFrame({"timestamp": dates})

    extractor = TemporalFeatureExtractor()
    out = extractor.transform(df)

    assert out.loc[0, "hour"] == 2
    assert out.loc[0, "is_night"] == 1
    assert out.loc[0, "is_weekend"] == 1

    assert out.loc[1, "hour"] == 14
    assert out.loc[1, "is_night"] == 0
    assert out.loc[1, "is_weekend"] == 0

    # Sine/cosine bounds [-1, 1]
    assert -1.0 <= out["sin_hour"].min() and out["sin_hour"].max() <= 1.0
    assert -1.0 <= out["cos_hour"].min() and out["cos_hour"].max() <= 1.0


def test_context_feature_extractor_state_tracking():
    """Verify context extractor tracks new devices and distances sequentially."""
    data = [
        {
            "user_id": "usr_A",
            "timestamp": "2026-01-01T10:00:00Z",
            "location_lat": 40.7128,
            "location_lon": -74.0060,
            "device_id": "dev_1",
            "merchant_id": "mcht_1",
        },
        {
            "user_id": "usr_A",
            "timestamp": "2026-01-01T10:15:00Z",
            "location_lat": 40.7128,
            "location_lon": -74.0060,
            "device_id": "dev_1",  # Known device
            "merchant_id": "mcht_1",
        },
        {
            "user_id": "usr_A",
            "timestamp": "2026-01-01T11:00:00Z",
            "location_lat": 34.0522,  # Moved to LA
            "location_lon": -118.2437,
            "device_id": "dev_new_unseen",  # New device
            "merchant_id": "mcht_2",  # New merchant
        },
    ]
    df = pd.DataFrame(data)

    extractor = ContextFeatureExtractor()
    extractor.fit(df.iloc[:1])  # Fit on first row
    out = extractor.transform(df)

    # First row: distance is 0.0
    assert out.loc[0, "distance_from_last_tx_km"] == 0.0

    # Second row: same device -> not new
    assert out.loc[1, "is_new_device"] == 0

    # Third row: unseen device and unseen merchant -> new
    assert out.loc[2, "is_new_device"] == 1
    assert out.loc[2, "is_new_merchant"] == 1
    assert out.loc[2, "distance_from_last_tx_km"] > 3000.0


def test_behavioral_feature_extractor_velocity_and_zscore():
    """Verify transaction velocity and rolling z-scores compute correctly."""
    data = [
        {"user_id": "usr_V", "amount": 100.0, "timestamp": "2026-01-01T12:00:00Z"},
        {"user_id": "usr_V", "amount": 100.0, "timestamp": "2026-01-01T12:10:00Z"},
        {"user_id": "usr_V", "amount": 800.0, "timestamp": "2026-01-01T12:20:00Z"},
        {"user_id": "usr_V", "amount": 50.0, "timestamp": "2026-01-02T15:00:00Z"},
    ]
    df = pd.DataFrame(data)

    extractor = BehavioralFeatureExtractor()
    extractor.fit(df)
    out = extractor.transform(df)

    # At row 0: velocity is 0
    assert out.loc[0, "tx_velocity_1h"] == 0

    # At row 1: 1 transaction occurred in last 1 hour
    assert out.loc[1, "tx_velocity_1h"] == 1

    # At row 2: 2 transactions in last 1 hour
    assert out.loc[2, "tx_velocity_1h"] == 2
    # Amount 800 is far above user's prior 100 average -> positive high z-score
    assert out.loc[2, "amount_zscore"] > 3.0

    # At row 3: 26 hours later -> 1h velocity is 0, 24h velocity is 0
    assert out.loc[3, "tx_velocity_1h"] == 0
    assert out.loc[3, "tx_velocity_24h"] == 0


def test_pipeline_fit_transform_and_single_predict(sample_raw_dataframe):
    """Verify FraudFeatureEngineeringPipeline fits, transforms, and executes single real-time scoring."""
    pipeline = FraudFeatureEngineeringPipeline(with_scaling=True)
    pipeline.fit(sample_raw_dataframe)

    assert pipeline.is_fitted_
    transformed = pipeline.transform(sample_raw_dataframe)

    assert isinstance(transformed, pd.DataFrame)
    assert len(transformed) == len(sample_raw_dataframe)
    assert len(pipeline.get_feature_names()) == transformed.shape[1]

    # Test single-transaction enrichment (sub-millisecond online scoring API path)
    single_tx = {
        "amount": 250.0,
        "timestamp": "2026-01-15T03:30:00Z",
        "location_lat": 40.7128,
        "location_lon": -74.0060,
        "device_id": "dev_test_123",
        "merchant_id": "mcht_test_999",
    }
    user_history = {
        "last_lat": 40.7100,
        "last_lon": -74.0050,
        "known_devices": ["dev_old"],
        "known_merchants": ["mcht_old"],
        "tx_velocity_1h": 4,
        "tx_velocity_24h": 7,
        "user_rolling_mean_amount": 50.0,
        "user_rolling_std_amount": 15.0,
    }

    vec, raw_dict = pipeline.transform_single(single_tx, user_history)

    assert vec.shape == (1, len(pipeline.get_feature_names()))
    assert raw_dict["is_night"] == 1.0
    assert raw_dict["is_new_device"] == 1.0
    assert raw_dict["is_new_merchant"] == 1.0
    assert raw_dict["amount_zscore"] > 10.0


def test_time_based_split_zero_leakage(sample_raw_dataframe):
    """Verify time-based train/test split has zero temporal leakage."""
    train_df, test_df = time_based_train_test_split(sample_raw_dataframe, test_size=0.25)

    assert len(train_df) + len(test_df) == len(sample_raw_dataframe)

    train_max_ts = pd.to_datetime(train_df["timestamp"]).max()
    test_min_ts = pd.to_datetime(test_df["timestamp"]).min()

    assert train_max_ts <= test_min_ts, "Train set overlaps with future test set timestamps!"
