"""
Week 3 - Advanced Data Analysis and Visualization in Logistics
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "hypothetical_logistics_dataset.csv"
CHARTS = BASE / "charts"
RESULTS = BASE / "results"
CHARTS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

df = pd.read_csv(DATA)

df.describe(include="all").to_csv(RESULTS / "descriptive_statistics.csv")

plt.figure(figsize=(9,5))
plt.hist(df["Delivery_Time_Days"], bins=25, edgecolor="black")
plt.title("Distribution of Delivery Time")
plt.xlabel("Delivery Time (Days)")
plt.ylabel("Number of Shipments")
plt.tight_layout()
plt.savefig(CHARTS / "delivery_time_distribution.png", dpi=180)
plt.close()

modes = ["Road", "Rail", "Air"]
costs = [df.loc[df["Transport_Mode"] == m, "Transport_Cost"] for m in modes]
plt.figure(figsize=(9,5))
plt.boxplot(costs, labels=modes)
plt.title("Transport Cost by Transport Mode")
plt.xlabel("Transport Mode")
plt.ylabel("Transport Cost")
plt.tight_layout()
plt.savefig(CHARTS / "cost_by_transport_mode.png", dpi=180)
plt.close()

plt.figure(figsize=(9,5))
plt.scatter(df["Shipment_Volume"], df["Delivery_Time_Days"], alpha=0.55)
plt.title("Shipment Volume vs Delivery Time")
plt.xlabel("Shipment Volume")
plt.ylabel("Delivery Time (Days)")
plt.tight_layout()
plt.savefig(CHARTS / "volume_vs_delivery.png", dpi=180)
plt.close()

delay_rate = df.groupby("Region")["Delayed"].mean().sort_values(ascending=False) * 100
plt.figure(figsize=(9,5))
delay_rate.plot(kind="bar")
plt.title("Delay Rate by Region")
plt.xlabel("Region")
plt.ylabel("Delayed Shipments (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(CHARTS / "delay_by_region.png", dpi=180)
plt.close()

corr = df.select_dtypes(include=np.number).corr()
plt.figure(figsize=(8,6))
plt.imshow(corr, interpolation="nearest", aspect="auto")
plt.colorbar(label="Correlation")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(CHARTS / "correlation_matrix.png", dpi=180)
plt.close()

month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
monthly = df.groupby("Month")["Delivery_Time_Days"].mean().reindex(month_order)
plt.figure(figsize=(10,5))
plt.plot(monthly.index, monthly.values, marker="o")
plt.title("Average Delivery Time by Month")
plt.xlabel("Month")
plt.ylabel("Average Delivery Time (Days)")
plt.tight_layout()
plt.savefig(CHARTS / "monthly_delivery_trend.png", dpi=180)
plt.close()

print("Analysis complete.")
