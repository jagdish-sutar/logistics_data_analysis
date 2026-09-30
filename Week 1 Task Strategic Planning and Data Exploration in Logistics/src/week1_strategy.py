"""
Week 1 - Strategic Planning and Data Exploration in Logistics

This file contains illustrative code for the planned logistics analytics
workflow. It is intentionally designed as a methodology example rather
than a complete analysis of the Olist dataset.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_logistics_kpis(df):
    """Calculate proposed logistics KPIs from an analysis-ready dataframe."""
    result = {}

    if "delivery_delay_days" in df.columns:
        result["on_time_delivery_rate_pct"] = (
            (df["delivery_delay_days"] <= 0).mean() * 100
        )
        late = df.loc[df["delivery_delay_days"] > 0, "delivery_delay_days"]
        result["average_delay_days"] = late.mean()

    if "delivery_lead_time_days" in df.columns:
        result["average_delivery_lead_time_days"] = (
            df["delivery_lead_time_days"].mean()
        )

    if {"freight_value", "order_value"}.issubset(df.columns):
        result["average_freight_cost_ratio_pct"] = (
            df["freight_value"] / df["order_value"]
        ).mean() * 100

    return result


def regression_template(df):
    """Illustrative regression workflow for delivery-time prediction."""
    features = [
        "freight_value",
        "order_value",
        "distance_km",
        "warehouse_processing_hours",
    ]
    target = "delivery_lead_time_days"

    if not set(features + [target]).issubset(df.columns):
        raise ValueError(
            "The dataframe must contain the engineered regression fields."
        )

    model_df = df[features + [target]].dropna()
    X = model_df[features]
    y = model_df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R2": r2_score(y_test, predictions),
    }


def clustering_template(df, n_clusters=3):
    """Illustrative K-Means segmentation workflow."""
    features = [
        "delivery_lead_time_days",
        "freight_value",
        "order_value",
    ]

    if not set(features).issubset(df.columns):
        raise ValueError(
            "The dataframe must contain the engineered clustering fields."
        )

    cluster_df = df[features].dropna().copy()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(cluster_df)

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    )
    cluster_df["cluster"] = model.fit_predict(scaled)

    return cluster_df


if __name__ == "__main__":
    print("Week 1 strategic planning templates loaded successfully.")
    print("Use the notebook for the complete documented workflow.")
