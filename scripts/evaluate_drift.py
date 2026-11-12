"""
Fraud Detection System (FDS) - Data Drift Evaluation CLI Script
Author: Alpha
Description: Runs Population Stability Index (PSI) and Kolmogorov-Smirnov (KS)
             tests between baseline training distribution and current transaction streams.
"""

import argparse
import json
import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from ml.data_generator import generate_dataset
from ml.drift.detector import DataDriftMonitor


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate covariate data drift using PSI and KS-tests.")
    parser.add_argument(
        "--reference",
        type=str,
        default="data/raw/transactions.csv",
        help="Path to baseline reference dataset (default: data/raw/transactions.csv)",
    )
    parser.add_argument(
        "--current",
        type=str,
        default=None,
        help="Path to current inference dataset (if omitted, a simulated shifted sample is evaluated)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/drift_report.json",
        help="Path to save output JSON telemetry (default: data/processed/drift_report.json)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.reference):
        print(f"[!] Reference data not found at {args.reference}. Generating baseline...")
        ref_df = generate_dataset(n_transactions=5000, random_state=42)
        os.makedirs(os.path.dirname(args.reference), exist_ok=True)
        ref_df.to_csv(args.reference, index=False)
    else:
        ref_df = pd.read_csv(args.reference)

    if args.current and os.path.exists(args.current):
        curr_df = pd.read_csv(args.current)
    else:
        print("[i] Generating production inference stream with simulated distribution shift...")
        # Higher fraud rate (1.5%) and slightly higher base spend to simulate inflation/shift
        curr_df = generate_dataset(n_transactions=2000, fraud_rate=0.015, random_state=999)

    monitor = DataDriftMonitor(
        reference_data=ref_df,
        numeric_features=["amount", "location_lat", "location_lon"],
        psi_threshold=0.25,
        ks_pvalue_threshold=0.01,
    )

    report = monitor.evaluate_drift(curr_df)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print("\n========================= Data Drift Telemetry Report =========================")
    print(f" Global Status:             {'⚠️ DRIFT DETECTED' if report['is_drift_detected'] else '✅ STABLE'}")
    print(f" Recommendation:            {report['recommendation']}")
    print(f" Features Tracked:          {report['total_features_evaluated']}")
    print(f" Features Drifted:          {report['drifted_features_count']}")
    print("-------------------------------------------------------------------------------")
    print(f"{'Feature':<20} | {'PSI':<8} | {'PSI Drift':<10} | {'KS Stat':<8} | {'KS p-val':<10} | {'Status'}")
    print("-------------------------------------------------------------------------------")
    for feat in report["features"]:
        status = "DRIFT" if feat["is_drift_detected"] else "STABLE"
        print(
            f"{feat['feature_name']:<20} | "
            f"{feat['psi']:<8.4f} | "
            f"{str(feat['psi_drift']):<10} | "
            f"{feat['ks_statistic']:<8.4f} | "
            f"{feat['ks_pvalue']:<10.6f} | "
            f"{status}"
        )
    print("===============================================================================\n")
    print(f"[✓] Saved complete telemetry to: {args.output}")


if __name__ == "__main__":
    main()
