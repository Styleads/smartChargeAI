"""
SmartChargeAI — Exploratory Data Analysis (EDA)
================================================
Loads zone_master.csv and hourly_demand.csv, validates data integrity,
computes summary statistics, generates diagnostic plots, and identifies
key zones of interest (top demand & grid stress).

All plots are saved to ./outputs/eda/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ──────────────────────────────────────────────
# 0. Configuration
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs", "eda")
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
    "commercial": "#e63946",
    "residential": "#457b9d",
}


# ──────────────────────────────────────────────
# 1. Load data
# ──────────────────────────────────────────────
print("=" * 70)
print("1. LOADING DATA")
print("=" * 70)

zone = pd.read_csv(os.path.join(BASE_DIR, "zone_master.csv"))
demand = pd.read_csv(os.path.join(BASE_DIR, "hourly_demand.csv"), parse_dates=["timestamp"])

print(f"\nzone_master.csv  — shape: {zone.shape}")
print(zone.head().to_string(index=False))

print(f"\nhourly_demand.csv — shape: {demand.shape}")
print(demand.head().to_string(index=False))


# ──────────────────────────────────────────────
# 2. Missing values
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("2. MISSING VALUES")
print("=" * 70)

zone_missing = zone.isnull().sum()
demand_missing = demand.isnull().sum()

print("\nzone_master.csv:")
print(zone_missing.to_string())
print(f"  Total missing cells: {zone_missing.sum()}")

print("\nhourly_demand.csv:")
print(demand_missing.to_string())
print(f"  Total missing cells: {demand_missing.sum()}")


# ──────────────────────────────────────────────
# 3. Zone-ID consistency check
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("3. ZONE-ID CONSISTENCY CHECK")
print("=" * 70)

zones_master = set(zone["zone_id"].unique())
zones_demand = set(demand["zone_id"].unique())
missing_in_master = zones_demand - zones_master

if missing_in_master:
    print(f"\n[!!] {len(missing_in_master)} zone_id(s) in hourly_demand NOT in zone_master:")
    for z in sorted(missing_in_master):
        print(f"   - {z}")
else:
    print("\n[OK] All zone_ids in hourly_demand.csv exist in zone_master.csv")


# ──────────────────────────────────────────────
# 4. Basic statistics
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("4. BASIC STATISTICS")
print("=" * 70)

for col in ["charging_demand_kw", "grid_total_load_kw"]:
    vals = demand[col]
    print(f"\n  {col}:")
    print(f"    Mean               : {vals.mean():.2f}")
    print(f"    Min                : {vals.min():.2f}")
    print(f"    Max                : {vals.max():.2f}")
    print(f"    Std Dev            : {vals.std():.2f}")
    print(f"    95th Percentile    : {np.percentile(vals, 95):.2f}")


# ──────────────────────────────────────────────
# 5. Plot — Average charging demand by hour
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("5. GENERATING PLOTS")
print("=" * 70)

hourly_avg = demand.groupby("hour")["charging_demand_kw"].mean()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(hourly_avg.index, hourly_avg.values, color=COLORS["primary"],
        linewidth=2.4, marker="o", markersize=5, zorder=3)
ax.fill_between(hourly_avg.index, hourly_avg.values, alpha=0.12,
                color=COLORS["primary"])

# Highlight the evening peak (hour 18-21)
peak_hours = hourly_avg.loc[18:21]
ax.fill_between(peak_hours.index, peak_hours.values, alpha=0.25,
                color=COLORS["secondary"], label="Evening peak (18–21h)")

ax.set_xlabel("Hour of Day")
ax.set_ylabel("Avg Charging Demand (kW)")
ax.set_title("Average Charging Demand by Hour of Day (All Zones, All Days)")
ax.set_xticks(range(0, 24))
ax.set_xlim(-0.5, 23.5)
ax.legend(loc="upper left")
ax.grid(True)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "avg_demand_by_hour.png"))
print("  [OK] avg_demand_by_hour.png saved")
plt.close(fig)


# ──────────────────────────────────────────────
# 6. Plot — Average charging demand by month
# ──────────────────────────────────────────────
monthly_avg = demand.groupby("month")["charging_demand_kw"].mean()

month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug"]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(monthly_avg.index, monthly_avg.values, color=COLORS["primary"],
              edgecolor="white", linewidth=0.8, zorder=3, width=0.6)

# Add value labels on bars
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
            f"{h:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold",
            color=COLORS["primary"])

ax.set_xlabel("Month (2024)")
ax.set_ylabel("Avg Charging Demand (kW)")
ax.set_title("Average Charging Demand by Month — Upward Growth Trend")
ax.set_xticks(monthly_avg.index)
ax.set_xticklabels(month_labels[:len(monthly_avg)])
ax.grid(True, axis="y")

# Trendline
z = np.polyfit(monthly_avg.index, monthly_avg.values, 1)
p = np.poly1d(z)
ax.plot(monthly_avg.index, p(monthly_avg.index), "--",
        color=COLORS["secondary"], linewidth=2, label=f"Trend (+{z[0]:.2f} kW/month)")
ax.legend(loc="upper left")

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "avg_demand_by_month.png"))
print("  [OK] avg_demand_by_month.png saved")
plt.close(fig)


# ──────────────────────────────────────────────
# 7. Plot — Demand by day of week: commercial vs residential
# ──────────────────────────────────────────────
merged = demand.merge(zone[["zone_id", "land_use_type"]], on="zone_id", how="left")

dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

commercial = merged[merged["land_use_type"] == "commercial"]
residential = merged[merged["land_use_type"] == "residential"]

comm_dow = commercial.groupby("day_of_week")["charging_demand_kw"].mean()
resi_dow = residential.groupby("day_of_week")["charging_demand_kw"].mean()

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(comm_dow.index, comm_dow.values, color=COLORS["commercial"],
        linewidth=2.4, marker="s", markersize=6, label="Commercial zones", zorder=3)
ax.plot(resi_dow.index, resi_dow.values, color=COLORS["residential"],
        linewidth=2.4, marker="o", markersize=6, label="Residential zones", zorder=3)

ax.set_xlabel("Day of Week")
ax.set_ylabel("Avg Charging Demand (kW)")
ax.set_title("Avg Charging Demand by Day of Week — Commercial vs Residential")
ax.set_xticks(range(7))
ax.set_xticklabels(dow_labels)
ax.legend()
ax.grid(True)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "demand_by_dow_land_use.png"))
print("  [OK] demand_by_dow_land_use.png saved")
plt.close(fig)


# ──────────────────────────────────────────────
# 8. Top 5 zones by average charging demand
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("6. TOP 5 ZONES BY AVERAGE CHARGING DEMAND")
print("=" * 70)

zone_avg = (
    demand.groupby("zone_id")["charging_demand_kw"]
    .mean()
    .reset_index()
    .rename(columns={"charging_demand_kw": "avg_demand_kw"})
    .merge(zone[["zone_id", "zone_name", "land_use_type"]], on="zone_id")
    .sort_values("avg_demand_kw", ascending=False)
)

top5 = zone_avg.head(5)
print()
print(top5[["zone_id", "zone_name", "land_use_type", "avg_demand_kw"]].to_string(index=False))


# ──────────────────────────────────────────────
# 9. Grid stress zones (load > 90% feeder capacity)
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("7. GRID STRESS ZONES  (grid_total_load_kw > 90% feeder_capacity_kw)")
print("=" * 70)

stress_merged = demand.merge(zone[["zone_id", "zone_name", "feeder_capacity_kw"]], on="zone_id")
stress_merged["threshold_kw"] = stress_merged["feeder_capacity_kw"] * 0.90
stress_merged["is_stressed"] = stress_merged["grid_total_load_kw"] > stress_merged["threshold_kw"]

stress_zones = (
    stress_merged[stress_merged["is_stressed"]]
    .groupby(["zone_id", "zone_name", "feeder_capacity_kw"])
    .agg(
        breach_count=("is_stressed", "sum"),
        max_load_kw=("grid_total_load_kw", "max"),
    )
    .reset_index()
    .sort_values("breach_count", ascending=False)
)

stress_zones["max_load_pct"] = (stress_zones["max_load_kw"] / stress_zones["feeder_capacity_kw"] * 100).round(1)

if stress_zones.empty:
    print("\n[OK] No zones exceeded 90% of feeder capacity.")
else:
    print(f"\n[WARN] {len(stress_zones)} zone(s) flagged as STRESS ZONES:\n")
    print(
        stress_zones[["zone_id", "zone_name", "feeder_capacity_kw",
                       "breach_count", "max_load_kw", "max_load_pct"]]
        .to_string(index=False)
    )


# ──────────────────────────────────────────────
# Done
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"All plots saved to: {OUT_DIR}")
print("=" * 70)
