"""
Fraud Detection System (FDS) - Synthetic Transaction Data Generator
Author: Alpha
Description: Generates realistic multi-entity financial transactions with natural
             spending variations and ~0.5% injected fraud patterns (velocity bursts,
             odd-hour high amounts, new device + foreign locations).
"""

from __future__ import annotations

import datetime as dt
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class UserProfile:
    user_id: str
    home_lat: float
    home_lon: float
    base_amount_mean: float
    base_amount_std: float
    primary_device_id: str
    secondary_device_id: str
    frequent_merchants: List[str]
    night_transactor: bool  # Shift workers or night owls


@dataclass
class MerchantProfile:
    merchant_id: str
    name: str
    category: str
    risk_factor: float
    base_lat: float
    base_lon: float


CATEGORIES = [
    "grocery",
    "electronics",
    "travel",
    "luxury_goods",
    "restaurant",
    "digital_services",
    "pharmacy",
    "gas_station",
    "apparel",
    "entertainment",
]

CATEGORY_RISK: Dict[str, float] = {
    "grocery": 0.8,
    "electronics": 2.5,
    "travel": 3.0,
    "luxury_goods": 4.0,
    "restaurant": 0.9,
    "digital_services": 2.2,
    "pharmacy": 0.7,
    "gas_station": 1.2,
    "apparel": 1.1,
    "entertainment": 1.3,
}

# Major US metropolitan coordinate anchors (lat, lon)
GEO_ANCHORS: List[Tuple[float, float]] = [
    (40.7128, -74.0060),  # New York
    (34.0522, -118.2437),  # Los Angeles
    (41.8781, -87.6298),   # Chicago
    (29.7604, -95.3698),   # Houston
    (33.4484, -112.0740),  # Phoenix
    (37.7749, -122.4194),  # San Francisco
    (47.6062, -122.3321),  # Seattle
    (25.7617, -80.1918),   # Miami
]


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points on Earth in kilometers."""
    r = 6371.0  # Earth's radius in kilometers
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


class TransactionGenerator:
    """Production-grade synthetic generator for transactional fraud detection modeling."""

    def __init__(
        self,
        n_users: int = 400,
        n_merchants: int = 80,
        fraud_rate: float = 0.005,
        start_date: Optional[dt.datetime] = None,
        duration_days: int = 45,
        random_state: int = 42,
    ) -> None:
        self.n_users = n_users
        self.n_merchants = n_merchants
        self.fraud_rate = fraud_rate
        self.start_date = start_date or dt.datetime(2026, 1, 1, 0, 0, 0)
        self.duration_days = duration_days
        self.random_state = random_state

        # Seed PRNGs
        random.seed(random_state)
        np.random.seed(random_state)

        # Build Profiles
        self.merchants = self._generate_merchants()
        self.users = self._generate_users()

    def _generate_merchants(self) -> List[MerchantProfile]:
        merchants: List[MerchantProfile] = []
        for i in range(1, self.n_merchants + 1):
            merchant_id = f"mcht_{i:04d}"
            category = random.choice(CATEGORIES)
            anchor = random.choice(GEO_ANCHORS)
            merchants.append(
                MerchantProfile(
                    merchant_id=merchant_id,
                    name=f"Merchant {category.capitalize()} {i}",
                    category=category,
                    risk_factor=CATEGORY_RISK.get(category, 1.0),
                    base_lat=anchor[0] + random.uniform(-0.15, 0.15),
                    base_lon=anchor[1] + random.uniform(-0.15, 0.15),
                )
            )
        return merchants

    def _generate_users(self) -> List[UserProfile]:
        users: List[UserProfile] = []
        merchant_ids = [m.merchant_id for m in self.merchants]
        for i in range(1, self.n_users + 1):
            user_id = f"usr_{i:05d}"
            anchor = random.choice(GEO_ANCHORS)
            home_lat = anchor[0] + random.gauss(0, 0.08)
            home_lon = anchor[1] + random.gauss(0, 0.08)
            mean_amt = float(random.uniform(25.0, 180.0))
            std_amt = float(mean_amt * random.uniform(0.25, 0.6))

            users.append(
                UserProfile(
                    user_id=user_id,
                    home_lat=home_lat,
                    home_lon=home_lon,
                    base_amount_mean=mean_amt,
                    base_amount_std=std_amt,
                    primary_device_id=f"dev_{user_id}_primary",
                    secondary_device_id=f"dev_{user_id}_secondary",
                    frequent_merchants=random.sample(
                        merchant_ids, k=min(6, len(merchant_ids))
                    ),
                    night_transactor=random.random() < 0.07,
                )
            )
        return users

    def generate(self, n_transactions: int = 25000) -> pd.DataFrame:
        """Generate transactions with natural distributions and realistic fraud patterns."""
        records: List[Dict] = []
        target_fraud_count = max(1, int(n_transactions * self.fraud_rate))
        normal_count = n_transactions - target_fraud_count

        total_seconds = self.duration_days * 86400
        start_ts = self.start_date.timestamp()

        # Track per-user state for realistic continuity
        user_last_ts: Dict[str, float] = {}

        # ---------------------------------------------------------------------
        # 1. Generate Normal Transactions
        # ---------------------------------------------------------------------
        for i in range(normal_count):
            user = random.choice(self.users)

            # Biased temporal distribution (daytime peak, evening activity)
            offset = random.uniform(0, total_seconds)
            cand_dt = dt.datetime.fromtimestamp(start_ts + offset)

            # Hour shaping: normal transactions peak between 08:00 and 22:00
            if not user.night_transactor:
                while cand_dt.hour in [1, 2, 3, 4] and random.random() < 0.85:
                    offset = random.uniform(0, total_seconds)
                    cand_dt = dt.datetime.fromtimestamp(start_ts + offset)

            # Amount: log-normal around user baseline
            amount = max(
                2.50,
                float(
                    np.random.normal(user.base_amount_mean, user.base_amount_std)
                ),
            )
            amount = round(min(amount, user.base_amount_mean * 4.0), 2)

            # Merchant selection
            if random.random() < 0.70 and user.frequent_merchants:
                merchant_id = random.choice(user.frequent_merchants)
            else:
                merchant_id = random.choice(self.merchants).merchant_id

            mcht = next(m for m in self.merchants if m.merchant_id == merchant_id)

            # Location: normally distributed around home with small jitter
            lat = round(user.home_lat + random.gauss(0, 0.03), 6)
            lon = round(user.home_lon + random.gauss(0, 0.03), 6)

            # Device
            device = (
                user.primary_device_id
                if random.random() < 0.85
                else user.secondary_device_id
            )

            channel = random.choices(
                ["ONLINE", "POS", "MOBILE"], weights=[0.45, 0.35, 0.20]
            )[0]

            records.append(
                {
                    "transaction_id": f"tx_{i:07d}",
                    "user_id": user.user_id,
                    "merchant_id": mcht.merchant_id,
                    "merchant_category": mcht.category,
                    "amount": amount,
                    "currency": "USD",
                    "timestamp": cand_dt.isoformat(),
                    "_epoch": cand_dt.timestamp(),
                    "location_lat": lat,
                    "location_lon": lon,
                    "device_id": device,
                    "ip_address": f"192.168.{random.randint(1, 254)}.{random.randint(1, 254)}",
                    "channel": channel,
                    "is_fraud": 0,
                    "fraud_type": "none",
                }
            )

        # ---------------------------------------------------------------------
        # 2. Inject Realistic Fraud Patterns (~0.5%)
        # ---------------------------------------------------------------------
        fraud_id_counter = normal_count
        fraud_archetypes = [
            "velocity_burst",
            "odd_hour_high_amount",
            "new_device_foreign_location",
        ]

        remaining_fraud = target_fraud_count
        while remaining_fraud > 0:
            user = random.choice(self.users)
            archetype = random.choice(fraud_archetypes)
            base_time = start_ts + random.uniform(86400, total_seconds - 86400)
            base_dt = dt.datetime.fromtimestamp(base_time)

            if archetype == "velocity_burst":
                # Pattern 1: Rapid succession of 3-5 transactions within 20 mins
                burst_size = min(remaining_fraud, random.randint(3, 5))
                target_mcht = random.choice(self.merchants)
                burst_device = (
                    f"dev_rogue_{random.randint(1000, 9999)}"
                    if random.random() < 0.5
                    else user.primary_device_id
                )

                for b in range(burst_size):
                    tx_time = base_dt + dt.timedelta(
                        minutes=random.randint(2, 6) * b,
                        seconds=random.randint(5, 45),
                    )
                    amount = round(
                        user.base_amount_mean * random.uniform(2.5, 6.0), 2
                    )
                    records.append(
                        {
                            "transaction_id": f"tx_{fraud_id_counter:07d}",
                            "user_id": user.user_id,
                            "merchant_id": target_mcht.merchant_id,
                            "merchant_category": target_mcht.category,
                            "amount": amount,
                            "currency": "USD",
                            "timestamp": tx_time.isoformat(),
                            "_epoch": tx_time.timestamp(),
                            "location_lat": round(
                                user.home_lat + random.gauss(0, 0.05), 6
                            ),
                            "location_lon": round(
                                user.home_lon + random.gauss(0, 0.05), 6
                            ),
                            "device_id": burst_device,
                            "ip_address": f"10.0.{random.randint(1, 254)}.{random.randint(1, 254)}",
                            "channel": "ONLINE",
                            "is_fraud": 1,
                            "fraud_type": "velocity_burst",
                        }
                    )
                    fraud_id_counter += 1

                remaining_fraud -= burst_size

            elif archetype == "odd_hour_high_amount":
                # Pattern 2: High amount (6x - 14x normal) between 01:00 AM and 04:30 AM
                odd_hour_dt = base_dt.replace(
                    hour=random.choice([1, 2, 3, 4]),
                    minute=random.randint(5, 55),
                    second=random.randint(0, 59),
                )
                mcht = random.choice(
                    [
                        m
                        for m in self.merchants
                        if m.category in ["luxury_goods", "electronics", "travel"]
                    ]
                    or self.merchants
                )
                amount = round(
                    user.base_amount_mean * random.uniform(6.5, 14.0) + 150.0, 2
                )

                records.append(
                    {
                        "transaction_id": f"tx_{fraud_id_counter:07d}",
                        "user_id": user.user_id,
                        "merchant_id": mcht.merchant_id,
                        "merchant_category": mcht.category,
                        "amount": amount,
                        "currency": "USD",
                        "timestamp": odd_hour_dt.isoformat(),
                        "_epoch": odd_hour_dt.timestamp(),
                        "location_lat": round(
                            user.home_lat + random.gauss(0, 0.04), 6
                        ),
                        "location_lon": round(
                            user.home_lon + random.gauss(0, 0.04), 6
                        ),
                        "device_id": user.primary_device_id,
                        "ip_address": f"172.16.{random.randint(1, 254)}.{random.randint(1, 254)}",
                        "channel": "ONLINE",
                        "is_fraud": 1,
                        "fraud_type": "odd_hour_high_amount",
                    }
                )
                fraud_id_counter += 1
                remaining_fraud -= 1

            elif archetype == "new_device_foreign_location":
                # Pattern 3: Brand new device at distant geographic location (> 800 km)
                foreign_anchor = random.choice(
                    [
                        a
                        for a in GEO_ANCHORS
                        if haversine_distance(
                            user.home_lat, user.home_lon, a[0], a[1]
                        )
                        > 700.0
                    ]
                )
                new_device = f"dev_unrecognized_{random.randint(10000, 99999)}"
                mcht = random.choice(self.merchants)
                amount = round(
                    user.base_amount_mean * random.uniform(3.0, 8.0), 2
                )

                records.append(
                    {
                        "transaction_id": f"tx_{fraud_id_counter:07d}",
                        "user_id": user.user_id,
                        "merchant_id": mcht.merchant_id,
                        "merchant_category": mcht.category,
                        "amount": amount,
                        "currency": "USD",
                        "timestamp": base_dt.isoformat(),
                        "_epoch": base_dt.timestamp(),
                        "location_lat": round(
                            foreign_anchor[0] + random.gauss(0, 0.05), 6
                        ),
                        "location_lon": round(
                            foreign_anchor[1] + random.gauss(0, 0.05), 6
                        ),
                        "device_id": new_device,
                        "ip_address": f"198.51.100.{random.randint(1, 254)}",
                        "channel": "ONLINE",
                        "is_fraud": 1,
                        "fraud_type": "new_device_foreign_location",
                    }
                )
                fraud_id_counter += 1
                remaining_fraud -= 1

        df = pd.DataFrame(records)

        # Sort strictly chronologically by epoch timestamp (avoids temporal leakage)
        df = df.sort_values(by="_epoch").reset_index(drop=True)
        df = df.drop(columns=["_epoch"])
        return df


def generate_dataset(
    n_transactions: int = 25000,
    fraud_rate: float = 0.005,
    output_path: Optional[str] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Convenience helper to generate transactions and optionally save to CSV."""
    gen = TransactionGenerator(
        n_users=350,
        n_merchants=75,
        fraud_rate=fraud_rate,
        random_state=random_state,
    )
    df = gen.generate(n_transactions=n_transactions)
    if output_path:
        df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    df = generate_dataset(n_transactions=5000, output_path="data/raw/sample_transactions.csv")
    print(f"Generated {len(df)} transactions.")
    print("Fraud distribution:")
    print(df["is_fraud"].value_counts(normalize=True))
    print("Fraud type breakdown:")
    print(df[df["is_fraud"] == 1]["fraud_type"].value_counts())
