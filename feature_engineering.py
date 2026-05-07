"""
SmartChargeAI -- Feature Engineering
=====================================
Merges zone metadata into hourly demand, creates lag / rolling / derived
features per zone, drops incomplete rows, and saves featured_demand.csv.

No scaling or train/test splitting is done here.
"""

import os
import pandas as pd
import numpy as np

# ──────────────────────────────────────────────
# 0. Paths
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ZONE_PATH = os.path.join(BASE_DIR, "zone_master.csv")
DEMAND_PATH = os.path.join(BASE_DIR, "hourly_demand.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "featured_demand.csv")


# ──────────────────────────────────────────────
# 1. Load & merge
# ──────────────────────────────────────────────
print("=" * 70)
print("1. LOADING & MERGING DATA")
print("=" * 70)

zone = pd.read_csv(ZONE_PATH)
demand = pd.read_csv(DEMAND_PATH)

# Keep only the zone-level columns we need
zone_cols = ["zone_id", "ev_count_current", "feeder_capacity_kw",
             "land_use_type", "ev_growth_rate_monthly"]
df = demand.merge(zone[zone_cols], on="zone_id", how="left")

print(f"  hourly_demand rows  : {len(demand):,}")
print(f"  After merge         : {len(df):,}")


# ──────────────────────────────────────────────
# 2. Convert timestamp & sort  (critical for lag correctness)
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("2. TIMESTAMP CONVERSION & SORTING")
print("=" * 70)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(["zone_id", "timestamp"]).reset_index(drop=True)

print(f"  dtype of timestamp  : {df['timestamp'].dtype}")
print(f"  Sorted by           : zone_id, timestamp")
print(f"  Date range          : {df['timestamp'].min()} --> {df['timestamp'].max()}")


# ──────────────────────────────────────────────
# 3. Lag features  (per zone)
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("3. LAG FEATURES (per zone)")
print("=" * 70)

grouped = df.groupby("zone_id")["charging_demand_kw"]

df["lag_1h"]   = grouped.shift(1)
df["lag_24h"]  = grouped.shift(24)
df["lag_168h"] = grouped.shift(168)

print("  lag_1h   -- charging_demand_kw from 1 hour ago")
print("  lag_24h  -- charging_demand_kw from 24 hours ago")
print("  lag_168h -- charging_demand_kw from 168 hours (1 week) ago")


# ──────────────────────────────────────────────
# 4. Rolling features  (per zone)
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("4. ROLLING FEATURES (per zone)")
print("=" * 70)

# Use shift(1) inside the rolling window so we never leak the current row
rolling_3  = grouped.transform(lambda s: s.shift(1).rolling(window=3,  min_periods=3).mean())
rolling_24 = grouped.transform(lambda s: s.shift(1).rolling(window=24, min_periods=24).mean())
rolling_24_std = grouped.transform(lambda s: s.shift(1).rolling(window=24, min_periods=24).std())

df["rolling_mean_3h"]  = rolling_3
df["rolling_mean_24h"] = rolling_24
df["rolling_std_24h"]  = rolling_24_std

print("  rolling_mean_3h  -- mean of last 3 hours  (excludes current)")
print("  rolling_mean_24h -- mean of last 24 hours (excludes current)")
print("  rolling_std_24h  -- std  of last 24 hours (excludes current)")


# ──────────────────────────────────────────────
# 5. Derived features
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("5. DERIVED FEATURES")
print("=" * 70)

# Time-based flags
df["is_peak_hour"]    = df["hour"].between(18, 22).astype(int)
df["is_morning_ramp"] = df["hour"].between(7, 9).astype(int)

# Capacity metrics
df["demand_to_capacity_ratio"] = df["charging_demand_kw"] / df["feeder_capacity_kw"]
df["headroom_kw"]              = df["feeder_capacity_kw"] - df["grid_total_load_kw"]

# Season  (1 = Jan-Feb, 2 = Mar-May, 3 = Jun-Aug)
season_map = {1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3}
df["season"] = df["month"].map(season_map)

# Land-use encoding
land_use_map = {
    "residential": 0,
    "commercial":  1,
    "mixed":       2,
    "industrial":  3,
    "logistics":   4,
}
df["land_use_encoded"] = df["land_use_type"].map(land_use_map)

derived_cols = ["is_peak_hour", "is_morning_ramp", "demand_to_capacity_ratio",
                "headroom_kw", "season", "land_use_encoded"]
for c in derived_cols:
    print(f"  {c}")


# ──────────────────────────────────────────────
# 6. Drop rows with NaN in lag/rolling columns
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("6. DROPPING INCOMPLETE ROWS (NaN from lags/rolling)")
print("=" * 70)

lag_rolling_cols = ["lag_1h", "lag_24h", "lag_168h",
                    "rolling_mean_3h", "rolling_mean_24h", "rolling_std_24h"]

rows_before = len(df)
df = df.dropna(subset=lag_rolling_cols).reset_index(drop=True)
rows_after = len(df)
rows_dropped = rows_before - rows_after

print(f"  Rows before drop : {rows_before:,}")
print(f"  Rows after drop  : {rows_after:,}")
print(f"  Rows dropped     : {rows_dropped:,}")


# ──────────────────────────────────────────────
# 7. Final summary
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("7. FINAL DATAFRAME SUMMARY")
print("=" * 70)
print(f"  Shape  : {df.shape}")
print(f"\n  Columns ({len(df.columns)}):")
for i, col in enumerate(df.columns, 1):
    print(f"    {i:2d}. {col}")


# ──────────────────────────────────────────────
# 8. Save
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("8. SAVING")
print("=" * 70)

df.to_csv(OUTPUT_PATH, index=False)
print(f"  Saved to: {OUTPUT_PATH}")
print(f"  File size: {os.path.getsize(OUTPUT_PATH) / (1024 * 1024):.1f} MB")
print("=" * 70)
