"""
SmartChargeAI — Peak Validation Plot (Marathahalli)
====================================================
Loads the saved XGBoost model, generates test predictions,
saves test_predictions.csv, and produces a line chart comparing
actual vs predicted demand for Marathahalli (Z006) during the
first full week of August 2024 (Aug 1–7).

Output: outputs/model/peak_validation_marathahalli.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xgboost as xgb

# ──────────────────────────────────────────────
# 0. Paths
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "featured_demand.csv")
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "model", "model_xgb.json")
OUT_DIR = os.path.join(BASE_DIR, "outputs", "model")
PRED_CSV = os.path.join(OUT_DIR, "test_predictions.csv")

# ──────────────────────────────────────────────
# 1. Generate test_predictions.csv if needed
# ──────────────────────────────────────────────
if os.path.exists(PRED_CSV):
    print(f"[OK] Loading existing predictions from {PRED_CSV}")
    preds = pd.read_csv(PRED_CSV, parse_dates=["timestamp"])
else:
    print("[..] test_predictions.csv not found — generating from saved model ...")

    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    FEATURE_COLS = [
        "hour", "day_of_week", "is_weekend", "month",
        "grid_total_load_kw", "temperature_celsius", "is_holiday",
        "ev_count_current", "feeder_capacity_kw", "ev_growth_rate_monthly",
        "land_use_encoded",
        "lag_1h", "lag_24h", "lag_168h",
        "rolling_mean_3h", "rolling_mean_24h", "rolling_std_24h",
        "is_peak_hour", "is_morning_ramp",
        "demand_to_capacity_ratio", "headroom_kw", "season",
    ]

    SPLIT_DATE = pd.Timestamp("2024-08-01")
    test_mask = df["timestamp"] >= SPLIT_DATE

    X_test = df.loc[test_mask, FEATURE_COLS]
    y_test = df.loc[test_mask, "charging_demand_kw"]

    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    y_pred = model.predict(X_test)

    preds = df.loc[test_mask, ["zone_id", "timestamp"]].copy()
    preds["actual_demand_kw"] = y_test.values
    preds["predicted_demand_kw"] = y_pred
    preds.to_csv(PRED_CSV, index=False)
    print(f"[OK] Saved {len(preds):,} rows to {PRED_CSV}")

# ──────────────────────────────────────────────
# 2. Filter: Marathahalli (Z006), Aug 1–7
# ──────────────────────────────────────────────
ZONE_ID = "Z006"
ZONE_NAME = "Marathahalli"
WEEK_START = pd.Timestamp("2024-08-01")
WEEK_END = pd.Timestamp("2024-08-07 23:59:59")

zone_df = preds[preds["zone_id"] == ZONE_ID].copy()
week_df = zone_df[
    (zone_df["timestamp"] >= WEEK_START) &
    (zone_df["timestamp"] <= WEEK_END)
].sort_values("timestamp").reset_index(drop=True)

print(f"\n  Zone            : {ZONE_NAME} ({ZONE_ID})")
print(f"  Week            : {WEEK_START.date()} to {WEEK_END.date()}")
print(f"  Data points     : {len(week_df)}")

# ──────────────────────────────────────────────
# 3. Plot
# ──────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "Helvetica"],
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "#f9f9f9",
    "axes.edgecolor": "#cccccc",
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
})

fig, ax = plt.subplots(figsize=(16, 5.5))

# Actual demand — solid blue
ax.plot(
    week_df["timestamp"], week_df["actual_demand_kw"],
    color="#1a73e8", linewidth=1.6, alpha=0.9,
    label="Actual Demand", zorder=3,
)

# Predicted demand — dashed orange
ax.plot(
    week_df["timestamp"], week_df["predicted_demand_kw"],
    color="#e8710a", linewidth=1.6, linestyle="--", alpha=0.9,
    label="Predicted Demand", zorder=3,
)

# Vertical gridlines at midnight
for day_offset in range(8):
    midnight = WEEK_START + pd.Timedelta(days=day_offset)
    ax.axvline(midnight, color="#bbb", linewidth=0.8, linestyle="-", zorder=1)

# Day labels at noon
day_names = ["Thu", "Fri", "Sat", "Sun", "Mon", "Tue", "Wed"]
for i, name in enumerate(day_names):
    noon = WEEK_START + pd.Timedelta(days=i, hours=12)
    ax.text(noon, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 0,
            f"{name}\nAug {1+i}",
            ha="center", va="bottom", fontsize=8, color="#666", fontweight="bold")

# Formatting
ax.set_xlabel("Timestamp (Hourly)", fontweight="bold")
ax.set_ylabel("Charging Demand (kW)", fontweight="bold")
ax.set_title(
    f"Peak Validation — {ZONE_NAME} ({ZONE_ID})  |  Aug 1–7, 2024",
    fontweight="bold", fontsize=15, pad=12,
)

ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
ax.set_xlim(WEEK_START, WEEK_END + pd.Timedelta(hours=1))

ax.legend(loc="upper left", framealpha=0.9, edgecolor="#ccc")
ax.grid(True, axis="y")
ax.grid(False, axis="x")  # we use manual midnight lines instead

fig.tight_layout()

# ──────────────────────────────────────────────
# 4. Save
# ──────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, "peak_validation_marathahalli.png")
fig.savefig(out_path, bbox_inches="tight")
plt.close(fig)

print(f"\n[OK] Saved: {out_path}")
