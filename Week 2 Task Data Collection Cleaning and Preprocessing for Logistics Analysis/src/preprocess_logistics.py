"""
Week 2 - Data Collection, Cleaning, and Preprocessing for Logistics Analysis

YuvaIntern | Logistics Data Analyst Intern

This script demonstrates a reproducible preprocessing pipeline for the
Brazilian E-Commerce Public Dataset by Olist.

The raw Olist files are intentionally not included in this repository.
Place the downloaded CSV files in a local data/raw/ directory before
running the full pipeline.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_orders():
    """Load the Olist orders table."""
    path = RAW_DIR / "olist_orders_dataset.csv"
    return pd.read_csv(path)


def inspect_data(df):
    """Return basic structure and missing-value information."""
    print("Shape:", df.shape)
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isna().sum())
    print("\nDuplicate rows:", df.duplicated().sum())


def parse_timestamps(df):
    """Convert Olist order timestamp columns to pandas datetime."""
    timestamp_columns = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    for column in timestamp_columns:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")

    return df


def create_delivery_features(df):
    """Create delivery lead-time and delay features."""
    if {
        "order_delivered_customer_date",
        "order_purchase_timestamp",
    }.issubset(df.columns):
        df["delivery_lead_time_days"] = (
            df["order_delivered_customer_date"]
            - df["order_purchase_timestamp"]
        ).dt.total_seconds() / (24 * 60 * 60)

    if {
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    }.issubset(df.columns):
        df["delivery_delay_days"] = (
            df["order_delivered_customer_date"]
            - df["order_estimated_delivery_date"]
        ).dt.total_seconds() / (24 * 60 * 60)

    return df


def remove_exact_duplicates(df):
    """Remove exact duplicate records."""
    before = len(df)
    df = df.drop_duplicates().copy()
    print(f"Removed {before - len(df)} exact duplicate rows.")
    return df


def validate_delivery_times(df):
    """Flag impossible negative delivery lead times for investigation."""
    if "delivery_lead_time_days" in df.columns:
        invalid = df["delivery_lead_time_days"] < 0
        print("Negative delivery lead-time records:", int(invalid.sum()))

    return df


def iqr_outlier_bounds(series):
    """Calculate IQR-based lower and upper bounds."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return lower, upper


def report_outliers(df, column):
    """Report IQR outliers without automatically deleting them."""
    if column not in df.columns:
        return

    values = df[column].dropna()
    if values.empty:
        return

    lower, upper = iqr_outlier_bounds(values)
    mask = (values < lower) | (values > upper)

    print(f"\nOutlier review: {column}")
    print(f"IQR lower bound: {lower:.2f}")
    print(f"IQR upper bound: {upper:.2f}")
    print(f"Potential outliers: {int(mask.sum())}")


def standardize_numeric_columns(df, columns):
    """Return standardized numeric features for ML workflows."""
    available = [c for c in columns if c in df.columns]

    if not available:
        return pd.DataFrame(index=df.index)

    clean = df[available].copy()
    clean = clean.fillna(clean.median(numeric_only=True))

    scaler = StandardScaler()
    scaled = scaler.fit_transform(clean)

    return pd.DataFrame(
        scaled,
        columns=[f"{c}_scaled" for c in available],
        index=df.index,
    )


def main():
    """Run the example preprocessing workflow."""
    orders = load_orders()

    inspect_data(orders)

    orders = parse_timestamps(orders)
    orders = remove_exact_duplicates(orders)
    orders = create_delivery_features(orders)
    orders = validate_delivery_times(orders)

    report_outliers(orders, "delivery_lead_time_days")
    report_outliers(orders, "delivery_delay_days")

    scaled = standardize_numeric_columns(
        orders,
        ["delivery_lead_time_days", "delivery_delay_days"],
    )

    processed = pd.concat([orders, scaled], axis=1)

    output_path = OUTPUT_DIR / "orders_preprocessed.csv"
    processed.to_csv(output_path, index=False)

    print(f"\nProcessed dataset saved to: {output_path}")


if __name__ == "__main__":
    main()
