"""
SmartChargeAI -- Module A: Demand Forecasting Model
=====================================================
Trains an XGBoost regression model on featured_demand.csv to predict
charging_demand_kw.  Uses a chronological train/test split so that
the model is evaluated on genuinely future data.

Outputs
-------
  outputs/model/model_xgb.json          — saved model
  outputs/model/metrics.json            — MAE, RMSE, MAPE, R²
  outputs/model/feature_importances.csv — per-feature importances
  outputs/model/actual_vs_pred.png      — scatter plot
  outputs/model/residual_dist.png       — residual distribution
  outputs/model/hourly_error.png        — error by hour-of-day
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# ──────────────────────────────────────────────
# 0. Paths & config
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "featured_demand.csv")
OUT_DIR = os.path.join(BASE_DIR, "outputs", "model")
os.makedirs(OUT_DIR, exist_ok=True)

# Matplotlib style
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "Helvetica"],
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "#f9f9f9",
    "axes.edgecolor": "#cccccc",
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
})

COLORS = {
    "primary": "#1a73e8",
    "secondary": "#e8710a",
    "accent": "#0d652d",
    "error": "#e63946",
}


# ──────────────────────────────────────────────
# 1. Load data
# ──────────────────────────────────────────────
print("=" * 70)
print("1. LOADING FEATURED DATA")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])

print(f"  Shape           : {df.shape}")
print(f"  Date range      : {df['timestamp'].min()} --> {df['timestamp'].max()}")
print(f"  Zones           : {df['zone_id'].nunique()}")


# ──────────────────────────────────────────────
# 2. Define features & target
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("2. FEATURE / TARGET SETUP")
print("=" * 70)

TARGET = "charging_demand_kw"

FEATURE_COLS = [
    # Time features
    "hour", "day_of_week", "is_weekend", "month",
    # Raw grid features
    "grid_total_load_kw", "temperature_celsius", "is_holiday",
    # Zone-level static features
    "ev_count_current", "feeder_capacity_kw", "ev_growth_rate_monthly",
    "land_use_encoded",
    # Lag features
    "lag_1h", "lag_24h", "lag_168h",
    # Rolling features
    "rolling_mean_3h", "rolling_mean_24h", "rolling_std_24h",
    # Derived features
    "is_peak_hour", "is_morning_ramp",
    "demand_to_capacity_ratio", "headroom_kw", "season",
]

print(f"  Target          : {TARGET}")
print(f"  Features ({len(FEATURE_COLS)}):")
for i, f in enumerate(FEATURE_COLS, 1):
    print(f"    {i:2d}. {f}")


# ──────────────────────────────────────────────
# 3. Chronological train/test split
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("3. TRAIN / TEST SPLIT (chronological)")
print("=" * 70)

# Use last month (August) as test set
SPLIT_DATE = pd.Timestamp("2024-08-01")

train_mask = df["timestamp"] < SPLIT_DATE
test_mask  = df["timestamp"] >= SPLIT_DATE

X_train = df.loc[train_mask, FEATURE_COLS]
y_train = df.loc[train_mask, TARGET]
X_test  = df.loc[test_mask, FEATURE_COLS]
y_test  = df.loc[test_mask, TARGET]

print(f"  Split date      : {SPLIT_DATE.date()}")
print(f"  Train rows      : {len(X_train):,}  ({train_mask.sum() / len(df) * 100:.1f}%)")
print(f"  Test rows       : {len(X_test):,}  ({test_mask.sum() / len(df) * 100:.1f}%)")


# ──────────────────────────────────────────────
# 4. Train XGBoost model
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("4. TRAINING XGBoost REGRESSOR")
print("=" * 70)

try:
    import xgboost as xgb
except ImportError:
    print("  [!] xgboost not installed — installing now ...")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost", "-q"])
    import xgboost as xgb

model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
)

print("  Hyperparameters:")
for k, v in model.get_params().items():
    if k in ("n_estimators", "max_depth", "learning_rate", "subsample",
             "colsample_bytree", "min_child_weight", "reg_alpha", "reg_lambda"):
        print(f"    {k:25s}: {v}")

print("\n  Training ...")
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=100,
)
print("  Training complete.")


# ──────────────────────────────────────────────
# 5. Evaluate
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("5. EVALUATION ON TEST SET")
print("=" * 70)

y_pred = model.predict(X_test)

# Save test predictions for downstream analysis
test_predictions = df.loc[test_mask, ["zone_id", "timestamp"]].copy()
test_predictions["actual_demand_kw"] = y_test.values
test_predictions["predicted_demand_kw"] = y_pred
test_predictions.to_csv(os.path.join(OUT_DIR, "test_predictions.csv"), index=False)
print(f"  Test predictions saved to: {os.path.join(OUT_DIR, 'test_predictions.csv')}")

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

# MAPE (avoid division by zero)
nonzero_mask = y_test > 1.0
mape = np.mean(np.abs((y_test[nonzero_mask] - y_pred[nonzero_mask]) / y_test[nonzero_mask])) * 100

metrics = {
    "MAE": round(mae, 4),
    "RMSE": round(rmse, 4),
    "MAPE_%": round(mape, 2),
    "R2": round(r2, 4),
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "n_features": len(FEATURE_COLS),
}

print(f"  MAE             : {mae:.4f} kW")
print(f"  RMSE            : {rmse:.4f} kW")
print(f"  MAPE            : {mape:.2f} %")
print(f"  R²              : {r2:.4f}")

# Save metrics
metrics_path = os.path.join(OUT_DIR, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\n  Metrics saved to: {metrics_path}")


# ──────────────────────────────────────────────
# 6. Feature importances
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("6. FEATURE IMPORTANCES")
print("=" * 70)

importances = model.feature_importances_
fi_df = (
    pd.DataFrame({"feature": FEATURE_COLS, "importance": importances})
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)

fi_df.to_csv(os.path.join(OUT_DIR, "feature_importances.csv"), index=False)

for i, row in fi_df.head(15).iterrows():
    bar = "#" * int(row["importance"] * 200)
    print(f"  {row['feature']:30s}  {row['importance']:.4f}  {bar}")


# ──────────────────────────────────────────────
# 7. Plot — Actual vs Predicted scatter
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("7. GENERATING DIAGNOSTIC PLOTS")
print("=" * 70)

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_test, y_pred, alpha=0.08, s=8, color=COLORS["primary"], zorder=2)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax.plot(lims, lims, "--", color=COLORS["error"], linewidth=1.5, label="Perfect prediction")
ax.set_xlabel("Actual Charging Demand (kW)")
ax.set_ylabel("Predicted Charging Demand (kW)")
ax.set_title(f"Actual vs Predicted — R² = {r2:.4f}")
ax.legend(loc="upper left")
ax.grid(True)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "actual_vs_pred.png"))
print("  [OK] actual_vs_pred.png saved")
plt.close(fig)


# ──────────────────────────────────────────────
# 8. Plot — Residual distribution
# ──────────────────────────────────────────────
residuals = y_test.values - y_pred

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(residuals, bins=100, color=COLORS["primary"], edgecolor="white",
        linewidth=0.4, alpha=0.85, zorder=3)
ax.axvline(0, color=COLORS["error"], linewidth=1.5, linestyle="--", label="Zero error")
ax.set_xlabel("Residual (Actual - Predicted) kW")
ax.set_ylabel("Count")
ax.set_title(f"Residual Distribution — MAE = {mae:.2f} kW")
ax.legend()
ax.grid(True, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "residual_dist.png"))
print("  [OK] residual_dist.png saved")
plt.close(fig)


# ──────────────────────────────────────────────
# 9. Plot — Error by hour of day
# ──────────────────────────────────────────────
test_df = df.loc[test_mask].copy()
test_df["predicted"] = y_pred
test_df["abs_error"] = np.abs(test_df[TARGET] - test_df["predicted"])

hourly_error = test_df.groupby("hour")["abs_error"].mean()

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(hourly_error.index, hourly_error.values, color=COLORS["primary"],
       edgecolor="white", linewidth=0.6, zorder=3, width=0.7)

# Highlight peak hours
for h in [18, 19, 20, 21]:
    if h in hourly_error.index:
        ax.bar(h, hourly_error[h], color=COLORS["secondary"],
               edgecolor="white", linewidth=0.6, zorder=3, width=0.7)

ax.set_xlabel("Hour of Day")
ax.set_ylabel("Mean Absolute Error (kW)")
ax.set_title("Mean Absolute Error by Hour of Day (Test Set)")
ax.set_xticks(range(0, 24))
ax.grid(True, axis="y")

# Custom legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=COLORS["primary"], label="Regular hours"),
    Patch(facecolor=COLORS["secondary"], label="Peak hours (18-21)"),
]
ax.legend(handles=legend_elements, loc="upper left")

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "hourly_error.png"))
print("  [OK] hourly_error.png saved")
plt.close(fig)


# ──────────────────────────────────────────────
# 10. Save model
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("8. SAVING MODEL")
print("=" * 70)

model_path = os.path.join(OUT_DIR, "model_xgb.json")
model.save_model(model_path)
model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
print(f"  Model saved to  : {model_path}")
print(f"  Model size      : {model_size_mb:.1f} MB")


# ──────────────────────────────────────────────
# 11. Per-zone performance summary
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("9. PER-ZONE PERFORMANCE (Test Set)")
print("=" * 70)

zone_perf = (
    test_df.groupby("zone_id")
    .apply(lambda g: pd.Series({
        "mae": mean_absolute_error(g[TARGET], g["predicted"]),
        "rmse": np.sqrt(mean_squared_error(g[TARGET], g["predicted"])),
        "r2": r2_score(g[TARGET], g["predicted"]),
    }))
    .sort_values("mae")
)

print(f"\n  {'zone_id':>10}  {'MAE':>8}  {'RMSE':>8}  {'R²':>8}")
print(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}")
for zone_id, row in zone_perf.iterrows():
    print(f"  {zone_id:>10}  {row['mae']:8.2f}  {row['rmse']:8.2f}  {row['r2']:8.4f}")

zone_perf.to_csv(os.path.join(OUT_DIR, "zone_performance.csv"))


# ──────────────────────────────────────────────
# Done
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETE")
print("=" * 70)
print(f"  R²    = {r2:.4f}")
print(f"  MAE   = {mae:.2f} kW")
print(f"  RMSE  = {rmse:.2f} kW")
print(f"  MAPE  = {mape:.2f}%")
print(f"  All outputs saved to: {OUT_DIR}")
print("=" * 70)
