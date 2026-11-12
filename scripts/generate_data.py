"""
Fraud Detection System (FDS) - Data Generation CLI Script
Author: Alpha
Description: CLI tool to generate synthetic financial transactions with realistic
             fraud patterns and output to data/raw/transactions.csv.
"""

import argparse
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.data_generator import generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic financial transaction dataset with realistic fraud patterns."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=25000,
        help="Number of transactions to generate (default: 25000)",
    )
    parser.add_argument(
        "--fraud-rate",
        type=float,
        default=0.005,
        help="Target fraud rate proportion (default: 0.005 for 0.5%%)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/transactions.csv",
        help="Destination path for generated CSV (default: data/raw/transactions.csv)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"[+] Generating {args.count} transactions (fraud rate: {args.fraud_rate * 100:.2f}%)...")
    df = generate_dataset(
        n_transactions=args.count,
        fraud_rate=args.fraud_rate,
        output_path=args.output,
        random_state=args.seed,
    )

    print(f"[✓] Successfully generated and saved to: {args.output}")
    print("\n--- Summary Statistics ---")
    print(f"Total Transactions: {len(df)}")
    print(f"Unique Users:       {df['user_id'].nunique()}")
    print(f"Unique Merchants:   {df['merchant_id'].nunique()}")
    print(f"Time Range:         {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print(f"Total Amount:       ${df['amount'].sum():,.2f}")
    print(f"Mean Amount:        ${df['amount'].mean():.2f}")
    print(f"Max Amount:         ${df['amount'].max():.2f}")
    print("\n--- Class Breakdown ---")
    print(df["is_fraud"].value_counts())
    print("\n--- Fraud Archetypes Injected ---")
    print(df[df["is_fraud"] == 1]["fraud_type"].value_counts())


if __name__ == "__main__":
    main()
