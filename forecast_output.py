"""
SmartCharge.AI - Module A, Step 5: 24-Hour Forecast Generation
===============================================================
Generates hourly demand forecasts for the next 24 hours for all 35 zones
using the trained XGBoost model. Outputs feed into the Streamlit dashboard.

Inputs:
    - outputs/model/model_xgb.json              (trained XGBoost model)
    - featured_demand.csv                       (full featured dataset)
    - zone_master.csv                           (zone metadata)
    - outputs/scheduling/zone_recommendations.csv (scheduling outputs)

Outputs:
    - outputs/forecasts/zone_forecasts_24h.csv   (hourly forecasts, all zones)
    - outputs/forecasts/zone_forecast_summary.csv (one row per zone summary)
"""

import os
from datetime import timedelta

import numpy as np
import pandas as pd
import xgboost as xgb


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
FORECAST_HOURS = 24
HISTORY_WINDOW = 168           # 7 days of hourly data for lag/rolling features
PEAK_RISK_THRESHOLD = 0.70     # demand >= 70% of feeder_capacity => peak risk

SEASON_MAP = {1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3}
LAND_USE_MAP = {
    "residential": 0,
    "commercial":  1,
    "mixed":       2,
    "industrial":  3,
    "logistics":   4,
}

MODEL_PATH   = os.path.join("outputs", "model", "model_xgb.json")
FEATURED_CSV = "featured_demand.csv"
ZONE_CSV     = "zone_master.csv"
SCHED_CSV    = os.path.join("outputs", "scheduling", "zone_recommendations.csv")
OUTPUT_DIR   = os.path.join("outputs", "forecasts")

# The exact feature order the model was trained on
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


def load_inputs():
    """Load model, featured data, zone master, and scheduling recommendations."""
    print("Loading inputs...")

    # Model
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    print(f"  Model loaded from {MODEL_PATH}")

    # Featured demand
    feat_df = pd.read_csv(FEATURED_CSV, parse_dates=["timestamp"])
    feat_df = feat_df.sort_values(["zone_id", "timestamp"]).reset_index(drop=True)
    print(f"  Featured demand: {len(feat_df):,} rows, {feat_df['zone_id'].nunique()} zones")

    # Zone master
    zones = pd.read_csv(ZONE_CSV)
    print(f"  Zone master: {len(zones)} zones")

    # Scheduling recommendations
    sched = pd.read_csv(SCHED_CSV)
    print(f"  Scheduling recommendations: {len(sched)} zones")

    return model, feat_df, zones, sched


def build_forecast_features(zone_history: pd.DataFrame, zones: pd.DataFrame,
                            zone_id: str) -> pd.DataFrame:
    """
    Construct feature rows for the next 24 forecast hours for a single zone.

    Uses the last 168 rows as the history window. For each forecast hour,
    features are built iteratively so that lag_1h from hour N uses the
    prediction from hour N-1.
    """
    zone_meta = zones.loc[zones["zone_id"] == zone_id].iloc[0]
    feeder_cap = zone_meta["feeder_capacity_kw"]
    ev_count = zone_meta["ev_count_current"]
    ev_growth = zone_meta["ev_growth_rate_monthly"]
    land_use = zone_meta["land_use_type"]
    land_use_enc = LAND_USE_MAP.get(land_use, 2)

    # Last known values as proxies for the forecast period
    last_temp = zone_history["temperature_celsius"].iloc[-1]
    last_grid_load = zone_history["grid_total_load_kw"].iloc[-1]

    # Get the demand series (will be extended with predictions)
    demand_series = zone_history["charging_demand_kw"].tolist()
    last_timestamp = zone_history["timestamp"].iloc[-1]

    forecast_rows = []

    for h_offset in range(1, FORECAST_HOURS + 1):
        ts = last_timestamp + timedelta(hours=h_offset)
        hour = ts.hour
        dow = ts.dayofweek
        is_weekend = int(dow >= 5)
        month = ts.month
        season = SEASON_MAP.get(month, 3)
        is_holiday = 0
        is_peak = int(18 <= hour <= 22)
        is_morning = int(7 <= hour <= 9)

        # Lag features — indices relative to end of demand_series
        n = len(demand_series)
        lag_1h   = demand_series[n - 1]       # previous hour
        lag_24h  = demand_series[n - 24] if n >= 24 else demand_series[0]
        lag_168h = demand_series[n - 168] if n >= 168 else demand_series[0]

        # Rolling features (shift(1) style: use indices [-1] through [-k-1])
        recent_3  = demand_series[max(0, n-3):n]
        recent_24 = demand_series[max(0, n-24):n]
        rolling_mean_3h  = np.mean(recent_3) if len(recent_3) >= 3 else np.mean(recent_3)
        rolling_mean_24h = np.mean(recent_24) if len(recent_24) >= 24 else np.mean(recent_24)
        rolling_std_24h  = np.std(recent_24, ddof=1) if len(recent_24) >= 24 else np.std(recent_24, ddof=1) if len(recent_24) > 1 else 0.0

        # Derived features using lag_1h as proxy for current demand
        demand_proxy = lag_1h
        d2c_ratio = demand_proxy / feeder_cap
        headroom = feeder_cap - last_grid_load

        row = {
            "hour": hour,
            "day_of_week": dow,
            "is_weekend": is_weekend,
            "month": month,
            "grid_total_load_kw": last_grid_load,
            "temperature_celsius": last_temp,
            "is_holiday": is_holiday,
            "ev_count_current": ev_count,
            "feeder_capacity_kw": feeder_cap,
            "ev_growth_rate_monthly": ev_growth,
            "land_use_encoded": land_use_enc,
            "lag_1h": lag_1h,
            "lag_24h": lag_24h,
            "lag_168h": lag_168h,
            "rolling_mean_3h": rolling_mean_3h,
            "rolling_mean_24h": rolling_mean_24h,
            "rolling_std_24h": rolling_std_24h,
            "is_peak_hour": is_peak,
            "is_morning_ramp": is_morning,
            "demand_to_capacity_ratio": d2c_ratio,
            "headroom_kw": headroom,
            "season": season,
            # Metadata (not model features)
            "_forecast_timestamp": ts,
            "_feeder_capacity_kw": feeder_cap,
        }
        forecast_rows.append(row)

        # Placeholder for the predicted demand (will be filled after prediction)
        # For now, use lag_1h as a rough proxy to keep the series going
        demand_series.append(lag_1h)

    return pd.DataFrame(forecast_rows)


def generate_forecasts(model, feat_df, zones):
    """Generate 24-hour forecasts for all zones."""
    print("\nGenerating 24-hour forecasts for all zones...")
    all_forecasts = []

    zone_ids = sorted(feat_df["zone_id"].unique())

    for zone_id in zone_ids:
        zone_data = feat_df[feat_df["zone_id"] == zone_id].copy()
        zone_data = zone_data.sort_values("timestamp")

        # Take last HISTORY_WINDOW rows
        history = zone_data.tail(HISTORY_WINDOW).copy()

        if len(history) < HISTORY_WINDOW:
            print(f"  [!] {zone_id}: only {len(history)} history rows, need {HISTORY_WINDOW}")

        # Build feature rows
        fc_df = build_forecast_features(history, zones, zone_id)

        # Predict using the model
        X_forecast = fc_df[FEATURE_COLS].values
        predictions = model.predict(X_forecast)
        predictions = np.maximum(predictions, 0.0)  # demand can't be negative

        # Now re-run with autoregressive feedback:
        # feed each prediction back into the demand series for subsequent hours
        demand_series = history["charging_demand_kw"].tolist()
        last_timestamp = history["timestamp"].iloc[-1]
        feeder_cap = zones.loc[zones["zone_id"] == zone_id, "feeder_capacity_kw"].iloc[0]

        # Initial prediction pass gives us rough estimates; now refine with feedback
        refined_rows = []
        for h_offset in range(FORECAST_HOURS):
            ts = fc_df.iloc[h_offset]["_forecast_timestamp"]

            if h_offset > 0:
                # Update lag/rolling features using previous predictions
                n = len(demand_series)
                fc_df.at[fc_df.index[h_offset], "lag_1h"] = demand_series[n - 1]
                recent_3 = demand_series[max(0, n-3):n]
                recent_24 = demand_series[max(0, n-24):n]
                fc_df.at[fc_df.index[h_offset], "rolling_mean_3h"] = np.mean(recent_3)
                fc_df.at[fc_df.index[h_offset], "rolling_mean_24h"] = np.mean(recent_24)
                fc_df.at[fc_df.index[h_offset], "rolling_std_24h"] = (
                    np.std(recent_24, ddof=1) if len(recent_24) > 1 else 0.0
                )
                fc_df.at[fc_df.index[h_offset], "demand_to_capacity_ratio"] = (
                    demand_series[n - 1] / feeder_cap
                )

                # Re-predict this hour with updated features
                X_row = fc_df.iloc[h_offset:h_offset+1][FEATURE_COLS].values
                pred = float(model.predict(X_row)[0])
                pred = max(pred, 0.0)
            else:
                pred = float(predictions[0])

            demand_series.append(pred)

            refined_rows.append({
                "zone_id": zone_id,
                "forecast_timestamp": ts,
                "hour": int(fc_df.iloc[h_offset]["hour"]),
                "predicted_demand_kw": round(pred, 2),
                "feeder_capacity_kw": feeder_cap,
                "grid_headroom_kw": round(feeder_cap - pred, 2),
                "is_peak_risk": int(pred >= PEAK_RISK_THRESHOLD * feeder_cap),
            })

        all_forecasts.extend(refined_rows)

    print(f"  Generated {len(all_forecasts)} forecast rows ({len(zone_ids)} zones x {FORECAST_HOURS} hours)")
    return pd.DataFrame(all_forecasts)


def merge_scheduling(forecasts, sched):
    """Merge scheduling recommendation columns into forecasts."""
    sched_cols = sched[["zone_id", "zone_name", "recommended_charging_window",
                        "typical_peak_window", "high_priority"]]
    merged = forecasts.merge(sched_cols, on="zone_id", how="left")
    return merged


def build_summary(forecasts):
    """Build one-row-per-zone summary."""
    summary = []
    for zone_id, grp in forecasts.groupby("zone_id"):
        max_row = grp.loc[grp["predicted_demand_kw"].idxmax()]
        summary.append({
            "zone_id": zone_id,
            "zone_name": grp["zone_name"].iloc[0],
            "max_predicted_demand_kw": round(grp["predicted_demand_kw"].max(), 2),
            "max_demand_hour": int(max_row["hour"]),
            "avg_headroom_kw": round(grp["grid_headroom_kw"].mean(), 2),
            "peak_risk_hours_count": int(grp["is_peak_risk"].sum()),
            "recommended_charging_window": grp["recommended_charging_window"].iloc[0],
            "typical_peak_window": grp["typical_peak_window"].iloc[0],
            "high_priority": grp["high_priority"].iloc[0],
        })
    return pd.DataFrame(summary)


def save_outputs(forecasts, summary):
    """Save forecast outputs."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Full 24h forecasts
    output_cols = [
        "zone_id", "zone_name", "forecast_timestamp", "hour",
        "predicted_demand_kw", "feeder_capacity_kw", "grid_headroom_kw",
        "is_peak_risk", "recommended_charging_window", "typical_peak_window",
        "high_priority",
    ]
    fc_path = os.path.join(OUTPUT_DIR, "zone_forecasts_24h.csv")
    forecasts[output_cols].to_csv(fc_path, index=False)
    print(f"\nSaved: {fc_path}  ({len(forecasts)} rows)")

    # Zone summary
    sum_path = os.path.join(OUTPUT_DIR, "zone_forecast_summary.csv")
    summary.to_csv(sum_path, index=False)
    print(f"Saved: {sum_path}  ({len(summary)} zones)")


def print_report(summary):
    """Print peak risk report."""
    print("\n" + "=" * 80)
    print("SmartCharge.AI  --  24-Hour Forecast Report")
    print("=" * 80)

    zones_with_risk = summary[summary["peak_risk_hours_count"] > 0]
    print(f"\nZones with at least one peak risk hour: {len(zones_with_risk)} / {len(summary)}")

    if len(zones_with_risk) > 0:
        print("\nPeak risk zones:")
        print(
            zones_with_risk[
                ["zone_id", "zone_name", "max_predicted_demand_kw",
                 "max_demand_hour", "peak_risk_hours_count", "high_priority"]
            ].to_string(index=False)
        )

    # Top 10 by max demand
    top10 = summary.nlargest(10, "max_predicted_demand_kw")
    print("\nTop 10 zones by max predicted demand:")
    print(
        top10[
            ["zone_id", "zone_name", "max_predicted_demand_kw",
             "max_demand_hour", "avg_headroom_kw", "high_priority"]
        ].to_string(index=False)
    )
    print("=" * 80)


def main():
    # 1. Load everything
    model, feat_df, zones, sched = load_inputs()

    # 2. Generate forecasts
    forecasts = generate_forecasts(model, feat_df, zones)

    # 3. Merge scheduling info
    forecasts = merge_scheduling(forecasts, sched)

    # 4. Build summary
    summary = build_summary(forecasts)

    # 5. Report
    print_report(summary)

    # 6. Save
    save_outputs(forecasts, summary)


if __name__ == "__main__":
    main()
