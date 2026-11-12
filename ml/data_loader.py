"""
Fraud Detection System (FDS) - Unified Data Loader
Author: Alpha
Description: Loads either Kaggle Credit Card Fraud dataset if present, or
             synthetically generated multi-entity transaction records.
"""

from __future__ import annotations

import datetime as dt
import os
from typing import Optional, Tuple

import pandas as pd

from ml.data_generator import generate_dataset


class DataLoader:
    """Provides a standardized interface for ingesting raw fraud transaction data."""

    def __init__(
        self,
        raw_data_dir: str = "data/raw",
        synthetic_count: int = 25000,
        fraud_rate: float = 0.005,
        random_state: int = 42,
    ) -> None:
        self.raw_data_dir = raw_data_dir
        self.synthetic_count = synthetic_count
        self.fraud_rate = fraud_rate
        self.random_state = random_state

    def load(self, preferred_filename: Optional[str] = None) -> pd.DataFrame:
        """
        Loads the transaction dataframe.
        Checks for:
        1. Explicitly requested file
        2. Kaggle 'creditcard.csv'
        3. Synthetic 'transactions.csv'
        4. Dynamically generates synthetic transactions if none found
        """
        os.makedirs(self.raw_data_dir, exist_ok=True)

        if preferred_filename:
            target_path = os.path.join(self.raw_data_dir, preferred_filename)
            if os.path.exists(target_path):
                print(f"[i] Loading transactions from specified path: {target_path}")
                return pd.read_csv(target_path)

        # Check for Kaggle credit card dataset
        kaggle_path = os.path.join(self.raw_data_dir, "creditcard.csv")
        if os.path.exists(kaggle_path):
            print(f"[i] Discovered Kaggle dataset at: {kaggle_path}. Standardizing format...")
            return self._standardize_kaggle_dataset(kaggle_path)

        # Check for cached synthetic dataset
        synth_path = os.path.join(self.raw_data_dir, "transactions.csv")
        if os.path.exists(synth_path):
            print(f"[i] Loading existing synthetic transactions from: {synth_path}")
            return pd.read_csv(synth_path)

        # Generate on the fly
        print(f"[i] No dataset found. Generating {self.synthetic_count} synthetic transactions...")
        return generate_dataset(
            n_transactions=self.synthetic_count,
            fraud_rate=self.fraud_rate,
            output_path=synth_path,
            random_state=self.random_state,
        )

    def _standardize_kaggle_dataset(self, path: str) -> pd.DataFrame:
        """Adapts the raw Kaggle PCA dataset (Time, V1..V28, Amount, Class) to standard schema."""
        raw_df = pd.read_csv(path)
        base_dt = dt.datetime(2026, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)

        # Convert relative Time (seconds from first transaction) to timestamps
        timestamps = [
            (base_dt + dt.timedelta(seconds=float(s))).isoformat()
            for s in raw_df["Time"]
        ]

        standardized = pd.DataFrame(
            {
                "transaction_id": [f"tx_kgl_{i:07d}" for i in range(len(raw_df))],
                "user_id": [f"usr_kgl_{(i % 500):05d}" for i in range(len(raw_df))],
                "merchant_id": [f"mcht_kgl_{(i % 100):04d}" for i in range(len(raw_df))],
                "merchant_category": "financial_services",
                "amount": raw_df["Amount"].values,
                "currency": "USD",
                "timestamp": timestamps,
                "location_lat": 40.7128,
                "location_lon": -74.0060,
                "device_id": [f"dev_{(i % 500)}" for i in range(len(raw_df))],
                "ip_address": "127.0.0.1",
                "channel": "ONLINE",
                "is_fraud": raw_df["Class"].values,
                "fraud_type": [
                    "kaggle_label" if c == 1 else "none" for c in raw_df["Class"]
                ],
            }
        )
        return standardized


def time_based_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.20,
    timestamp_col: str = "timestamp",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-based train/test split without random shuffling to strictly avoid
    temporal leakage and lookahead bias.
    """
    df_sorted = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_sorted[timestamp_col]):
        df_sorted["_split_ts"] = pd.to_datetime(df_sorted[timestamp_col], utc=True)
    else:
        df_sorted["_split_ts"] = df_sorted[timestamp_col]

    df_sorted = df_sorted.sort_values(by="_split_ts").reset_index(drop=True)
    df_sorted = df_sorted.drop(columns=["_split_ts"])

    split_idx = int(len(df_sorted) * (1.0 - test_size))
    train_df = df_sorted.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df_sorted.iloc[split_idx:].copy().reset_index(drop=True)

    return train_df, test_df
