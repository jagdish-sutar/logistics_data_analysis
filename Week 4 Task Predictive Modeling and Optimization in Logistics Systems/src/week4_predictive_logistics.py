"""
Week 4 - Predictive Modeling and Optimization in Logistics

Synthetic educational project:
1. Train and compare regression models.
2. Evaluate with MAE, RMSE and R².
3. Perform 5-fold cross-validation.
4. Tune Random Forest with GridSearchCV.
5. Demonstrate a simple transport-mode optimization scenario.

The dataset is synthetic and does not represent a real logistics operation.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "hypothetical_logistics_week4_dataset.csv"
RESULTS = BASE / "results"
CHARTS = BASE / "charts"
RESULTS.mkdir(exist_ok=True)
CHARTS.mkdir(exist_ok=True)

df = pd.read_csv(DATA)

target = "Delivery_Time_Days"
X = df.drop(columns=[target, "Delayed"])
y = df[target]

categorical = ["Transport_Mode", "Region", "Priority"]
numerical = [
    "Shipment_Volume",
    "Distance_KM",
    "Warehouse_Processing_Hours",
    "Weather_Disruption",
    "Peak_Period",
    "Transport_Cost",
]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", numerical),
    ]
)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=8),
    "Random Forest": RandomForestRegressor(
        random_state=42, n_estimators=200, max_depth=12, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42, n_estimators=150),
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

results = []
predictions = {}

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    predictions[name] = pred

    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "R2": r2_score(y_test, pred),
    })

results_df = pd.DataFrame(results).sort_values("RMSE")
results_df.to_csv(RESULTS / "model_results.csv", index=False)

# Cross-validation for Random Forest
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        random_state=42, n_estimators=200, max_depth=12, n_jobs=-1
    )),
])

cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rmse = np.sqrt(
    -cross_val_score(
        rf_pipeline, X, y, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1
    )
)

pd.DataFrame({
    "Fold": np.arange(1, 6),
    "RMSE": cv_rmse
}).to_csv(RESULTS / "random_forest_cross_validation.csv", index=False)

# Random Forest hyperparameter tuning
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [8, 12, None],
    "model__min_samples_split": [2, 5],
}

rf_base = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42, n_jobs=-1)),
])

grid = GridSearchCV(
    rf_base,
    param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
)
grid.fit(X_train, y_train)

tuned_pred = grid.predict(X_test)
tuned_results = pd.DataFrame([{
    "Model": "Tuned Random Forest",
    "MAE": mean_absolute_error(y_test, tuned_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, tuned_pred)),
    "R2": r2_score(y_test, tuned_pred),
    "Best_Params": str(grid.best_params_),
}])
tuned_results.to_csv(RESULTS / "tuned_random_forest_results.csv", index=False)

# Model comparison chart
plt.figure(figsize=(9, 5))
plt.bar(results_df["Model"], results_df["RMSE"])
plt.title("Model Comparison — RMSE")
plt.xlabel("Model")
plt.ylabel("RMSE (Days)")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(CHARTS / "model_comparison_rmse.png", dpi=180)
plt.close()

# Actual vs predicted using best test-set model
best_model_name = results_df.iloc[0]["Model"]
best_pred = predictions[best_model_name]

plt.figure(figsize=(7, 6))
plt.scatter(y_test, best_pred, alpha=0.55)
min_value = min(y_test.min(), best_pred.min())
max_value = max(y_test.max(), best_pred.max())
plt.plot([min_value, max_value], [min_value, max_value])
plt.title(f"Actual vs Predicted — {best_model_name}")
plt.xlabel("Actual Delivery Time (Days)")
plt.ylabel("Predicted Delivery Time (Days)")
plt.tight_layout()
plt.savefig(CHARTS / "actual_vs_predicted.png", dpi=180)
plt.close()

# Residual distribution
residuals = y_test - best_pred
plt.figure(figsize=(9, 5))
plt.hist(residuals, bins=25, edgecolor="black")
plt.title(f"Residual Distribution — {best_model_name}")
plt.xlabel("Actual - Predicted (Days)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(CHARTS / "residual_distribution.png", dpi=180)
plt.close()

# Cross-validation chart
plt.figure(figsize=(8, 5))
plt.bar(
    [f"Fold {i}" for i in range(1, 6)],
    cv_rmse,
)
plt.axhline(cv_rmse.mean(), linestyle="--", label=f"Mean RMSE = {cv_rmse.mean():.3f}")
plt.title("Random Forest 5-Fold Cross-Validation")
plt.xlabel("Fold")
plt.ylabel("RMSE (Days)")
plt.legend()
plt.tight_layout()
plt.savefig(CHARTS / "cross_validation.png", dpi=180)
plt.close()

# Simple transport-mode optimization demonstration
# Representative shipment with fixed non-mode attributes.
scenario = pd.DataFrame({
    "Transport_Mode": ["Road", "Rail", "Air"],
    "Region": ["Central"] * 3,
    "Priority": ["Standard"] * 3,
    "Shipment_Volume": [250] * 3,
    "Distance_KM": [1000] * 3,
    "Warehouse_Processing_Hours": [8] * 3,
    "Weather_Disruption": [0] * 3,
    "Peak_Period": [0] * 3,
    "Transport_Cost": [380, 520, 900],
})

# Use the best pipeline from the test comparison.
best_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", models[best_model_name]),
])
best_pipeline.fit(X_train, y_train)

scenario["Predicted_Delivery_Days"] = best_pipeline.predict(scenario)

required_days = 5
late_penalty = 75
scenario["Late_Days"] = np.maximum(
    scenario["Predicted_Delivery_Days"] - required_days, 0
)
scenario["Objective_Cost"] = (
    scenario["Transport_Cost"] + scenario["Late_Days"] * late_penalty
)

scenario.to_csv(RESULTS / "transport_optimization_scenario.csv", index=False)

plt.figure(figsize=(8, 5))
plt.bar(scenario["Transport_Mode"], scenario["Objective_Cost"])
plt.title("Illustrative Transport-Mode Optimization Objective")
plt.xlabel("Transport Mode")
plt.ylabel("Cost + Late Delivery Penalty")
plt.tight_layout()
plt.savefig(CHARTS / "optimization_mode_selection.png", dpi=180)
plt.close()

print("Model results:")
print(results_df)
print("\\nCross-validation RMSE:", cv_rmse)
print("\\nTuned Random Forest:")
print(tuned_results)
print("\\nOptimization scenario:")
print(scenario)
