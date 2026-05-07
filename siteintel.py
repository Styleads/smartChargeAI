"""Module B — SiteIntel pipeline.

Loads zone_master.csv, hourly_demand.csv, candidate_sites.csv and produces
zone_scored.csv, demand_surface.csv, siteintel_recommendations.csv,
recommended_sites.json.
"""

import json
import math
import os
import warnings

os.makedirs("outputs/siteintel", exist_ok=True)

import numpy as np
import pandas as pd
from scipy.interpolate import Rbf

warnings.filterwarnings("ignore")

try:
    from scipy.interpolate import RBFInterpolator
    _HAS_RBFI = True
except ImportError:
    _HAS_RBFI = False


def percentile_rank(series: pd.Series) -> pd.Series:
    return series.rank(pct=True) * 100


# ============================================================
# STEP 1 — ZONE DEMAND SCORING
# ============================================================
zm = pd.read_csv("zone_master.csv")
hd = pd.read_csv("hourly_demand.csv")

peak_mask = hd["hour"].isin([18, 19, 20, 21])
avg_peak = (
    hd.loc[peak_mask]
    .groupby("zone_id")["charging_demand_kw"]
    .mean()
    .rename("avg_peak_demand_kw")
)

hd_cap = hd.merge(zm[["zone_id", "feeder_capacity_kw"]], on="zone_id")
stress = (
    hd_cap.loc[hd_cap["grid_total_load_kw"] > 0.90 * hd_cap["feeder_capacity_kw"]]
    .groupby("zone_id")
    .size()
    .rename("stress_event_count")
)

monthly_means = (
    hd.groupby(["zone_id", "month"])["charging_demand_kw"].mean().reset_index()
)


def _slope(g: pd.DataFrame) -> float:
    if len(g) < 2:
        return 0.0
    return float(np.polyfit(g["month"].to_numpy(float),
                            g["charging_demand_kw"].to_numpy(), 1)[0])


slopes = (
    monthly_means.groupby("zone_id").apply(_slope).rename("monthly_growth_slope")
)

zone_scored = (
    zm.merge(avg_peak, on="zone_id", how="left")
      .merge(stress, on="zone_id", how="left")
      .merge(slopes, on="zone_id", how="left")
)
zone_scored["stress_event_count"] = zone_scored["stress_event_count"].fillna(0).astype(int)
zone_scored["peak_to_capacity_ratio"] = (
    zone_scored["avg_peak_demand_kw"] / zone_scored["feeder_capacity_kw"]
)
zone_scored["demand_percentile"] = percentile_rank(zone_scored["avg_peak_demand_kw"])

zone_scored["demand_pressure_score"] = (
    0.35 * percentile_rank(zone_scored["avg_peak_demand_kw"])
    + 0.30 * percentile_rank(zone_scored["peak_to_capacity_ratio"])
    + 0.20 * percentile_rank(zone_scored["stress_event_count"])
    + 0.15 * percentile_rank(zone_scored["monthly_growth_slope"])
)

zone_scored["monthly_growth_rate"] = (
    zone_scored["monthly_growth_slope"] / zone_scored["avg_peak_demand_kw"]
)
zone_scored["projected_6m_kw"] = (
    zone_scored["avg_peak_demand_kw"] * (1 + zone_scored["monthly_growth_rate"]) ** 6
)
zone_scored["projected_12m_kw"] = (
    zone_scored["avg_peak_demand_kw"] * (1 + zone_scored["monthly_growth_rate"]) ** 12
)

print("\nTop 10 zones by demand_pressure_score:")
print(
    zone_scored.nlargest(10, "demand_pressure_score")[
        [
            "zone_id", "zone_name", "avg_peak_demand_kw", "peak_to_capacity_ratio",
            "stress_event_count", "monthly_growth_slope", "demand_pressure_score",
        ]
    ].to_string(index=False)
)
zone_scored.to_csv("outputs/siteintel/zone_scored.csv", index=False)
print("✅ Step 1 complete — zone scoring done")


# ============================================================
# STEP 2 — SPATIAL DEMAND SURFACE
# ============================================================
lat_grid = np.linspace(12.70, 13.25, 100)
lon_grid = np.linspace(77.38, 77.85, 100)
grid_lon, grid_lat = np.meshgrid(lon_grid, lat_grid)

try:
    rbf = Rbf(
        zone_scored["longitude"], zone_scored["latitude"],
        zone_scored["demand_pressure_score"],
        function="multiquadric", epsilon=0.1,
    )
    interpolated = rbf(grid_lon, grid_lat)
except Exception:
    if not _HAS_RBFI:
        raise
    pts = zone_scored[["longitude", "latitude"]].to_numpy()
    rbfi = RBFInterpolator(
        pts, zone_scored["demand_pressure_score"].to_numpy(),
        kernel="multiquadric", epsilon=0.1,
    )
    flat = np.column_stack([grid_lon.ravel(), grid_lat.ravel()])
    interpolated = rbfi(flat).reshape(grid_lon.shape)

interpolated = np.clip(interpolated, 0, 100)

surface = pd.DataFrame({
    "grid_lat": grid_lat.ravel(),
    "grid_lon": grid_lon.ravel(),
    "interpolated_demand_score": interpolated.ravel(),
})
surface.to_csv("outputs/siteintel/demand_surface.csv", index=False)
print(f"✅ Step 2 complete — demand surface built ({len(surface)} cells)")


# ============================================================
# STEP 3 — CANDIDATE SITE SCORING
# ============================================================
cs = pd.read_csv("candidate_sites.csv")
cs = cs.merge(
    zone_scored[
        ["zone_id", "zone_name", "demand_pressure_score",
         "projected_6m_kw", "projected_12m_kw"]
    ],
    on="zone_id", how="left",
)

cs["demand_score"] = cs["demand_pressure_score"]
cs["grid_score"] = (cs["available_grid_capacity_kw"] / 500 * 100).clip(upper=100)
cs["accessibility_score"] = cs["road_accessibility_score"] * 100
cs["coverage_gap_score"] = (cs["distance_to_nearest_station_km"] / 5.0).clip(upper=1.0) * 100

cs["opportunity_score"] = (
    0.35 * cs["demand_score"]
    + 0.25 * cs["grid_score"]
    + 0.15 * cs["accessibility_score"]
    + 0.25 * cs["coverage_gap_score"]
)

cs["demand_contribution"] = 0.35 * cs["demand_score"]
cs["grid_contribution"] = 0.25 * cs["grid_score"]
cs["accessibility_contribution"] = 0.15 * cs["accessibility_score"]
cs["coverage_contribution"] = 0.25 * cs["coverage_gap_score"]

conditions = [
    (cs["opportunity_score"] >= 72) & (cs["land_availability"] == "available"),
    (cs["land_availability"] == "restricted") | (cs["available_grid_capacity_kw"] < 100),
]
cs["confidence_tier"] = np.select(conditions, ["High", "Needs Field Survey"], default="Medium")

print("\nTop 10 candidates by opportunity_score:")
print(
    cs.nlargest(10, "opportunity_score")[
        ["site_id", "site_name", "zone_name", "opportunity_score", "confidence_tier"]
    ].to_string(index=False)
)
print("✅ Step 3 complete — candidate scoring done")


# ============================================================
# STEP 4 — FACILITY LOCATION OPTIMIZATION
# ============================================================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.asin(math.sqrt(a))


eligible = (
    cs[cs["land_availability"] != "restricted"]
    .sort_values("opportunity_score", ascending=False)
    .reset_index(drop=True)
)

selected_rows = []
for _, cand in eligible.iterrows():
    if len(selected_rows) == 10:
        break
    too_close = False
    for sel in selected_rows:
        if haversine(cand["latitude"], cand["longitude"],
                     sel["latitude"], sel["longitude"]) < 1.5:
            too_close = True
            break
    if not too_close:
        selected_rows.append(cand)

top10_sites = pd.DataFrame(selected_rows).reset_index(drop=True)

print(f"\n✅ Step 4 complete — optimization done, {len(top10_sites)} sites selected")
print("\nSelected sites:")
print(
    top10_sites[
        ["site_id", "site_name", "zone_name", "opportunity_score", "confidence_tier"]
    ].to_string(index=False)
)


# ============================================================
# STEP 5 — BASELINE COMPARISON
# ============================================================
lat_centers = [12.78, 12.97, 13.16]
lon_centers = [77.42, 77.57, 77.70, 77.82]

eligible_pool = cs[cs["land_availability"] != "restricted"].reset_index(drop=True)

baseline_idx = []
for la in lat_centers:
    for lo in lon_centers:
        dists = eligible_pool.apply(
            lambda r: haversine(la, lo, r["latitude"], r["longitude"]), axis=1
        )
        baseline_idx.append(int(dists.idxmin()))

# Dedupe preserving order
seen = set()
unique_idx = []
for i in baseline_idx:
    if i not in seen:
        unique_idx.append(i)
        seen.add(i)

baseline_pool = eligible_pool.loc[unique_idx]
baseline_top = baseline_pool.nlargest(min(10, len(baseline_pool)), "opportunity_score")


def metrics(df):
    zones = df["zone_id"].unique()
    return {
        "mean_opportunity_score": df["opportunity_score"].mean(),
        "total_demand_covered": (
            zone_scored.set_index("zone_id").loc[zones, "demand_pressure_score"].sum()
        ),
        "mean_coverage_gap": df["distance_to_nearest_station_km"].mean(),
    }


opt_m = metrics(top10_sites)
base_m = metrics(baseline_top)
improvement = (
    (opt_m["mean_opportunity_score"] - base_m["mean_opportunity_score"])
    / base_m["mean_opportunity_score"] * 100
)

print("\nComparison:")
print(f"  {'Metric':<32}| {'Optimized':>10} | {'Baseline':>10}")
print(f"  {'-'*32}+{'-'*12}+{'-'*12}")
print(f"  {'Mean opportunity score':<32}| {opt_m['mean_opportunity_score']:>10.1f} | {base_m['mean_opportunity_score']:>10.1f}")
print(f"  {'Total demand covered':<32}| {opt_m['total_demand_covered']:>10.1f} | {base_m['total_demand_covered']:>10.1f}")
print(f"  {'Mean coverage gap (km)':<32}| {opt_m['mean_coverage_gap']:>10.2f} | {base_m['mean_coverage_gap']:>10.2f}")
imp_str = f"+{improvement:.1f}%"
print(f"  {'Improvement in score':<32}| {imp_str:>10} | {'-':>10}")
print("✅ Step 5 complete — baseline comparison done")


# ============================================================
# STEP 6 — SAVE ALL OUTPUTS
# ============================================================

# Save all 80 scored sites for the dashboard
cs.to_csv("outputs/siteintel/siteintel_recommendations.csv", index=False)

# Save top 10 optimized sites separately
top10_sites.to_csv("outputs/siteintel/siteintel_top10_sites.csv", index=False)

records = []
for _, r in top10_sites.iterrows():
    records.append({
        "site_id": r["site_id"],
        "site_name": r["site_name"],
        "zone_id": r["zone_id"],
        "zone_name": r["zone_name"],
        "latitude": float(r["latitude"]),
        "longitude": float(r["longitude"]),
        "site_type": r["site_type"],
        "opportunity_score": float(r["opportunity_score"]),
        "confidence_tier": r["confidence_tier"],
        "demand_contribution": float(r["demand_contribution"]),
        "grid_contribution": float(r["grid_contribution"]),
        "accessibility_contribution": float(r["accessibility_contribution"]),
        "coverage_contribution": float(r["coverage_contribution"]),
        "available_grid_capacity_kw": float(r["available_grid_capacity_kw"]),
        "road_accessibility_score": float(r["road_accessibility_score"]),
        "distance_to_nearest_station_km": float(r["distance_to_nearest_station_km"]),
        "land_availability": r["land_availability"],
        "zone_demand_pressure_score": float(r["demand_pressure_score"]),
        "projected_6m_kw": float(r["projected_6m_kw"]),
        "projected_12m_kw": float(r["projected_12m_kw"]),
    })

with open("outputs/siteintel/recommended_sites.json", "w") as f:
    json.dump(records, f, indent=2)

top = top10_sites.iloc[0]
print("=" * 55)
print("SITEINTEL COMPLETE")
print(f"Top site    : {top['site_name']} ({top['zone_name']})")
print(f"Score       : {top['opportunity_score']:.1f} / 100 — Tier: {top['confidence_tier']}")
print(f"Optimized mean score : {opt_m['mean_opportunity_score']:.1f}")
print(f"Baseline mean score  : {base_m['mean_opportunity_score']:.1f}")
print(f"Improvement          : +{improvement:.1f}%")
print("=" * 55)
