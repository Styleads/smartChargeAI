"""
SmartCharge.AI - Module A, Step 4: Scheduling Recommendation Logic
==================================================================
Generates per-zone EV charging schedule recommendations for BESCOM, Bengaluru.

Inputs:
    - outputs/model/test_predictions.csv  (zone_id, timestamp, actual_demand_kw, predicted_demand_kw)
    - zone_master.csv                     (zone_id, zone_name, feeder_capacity_kw, land_use_type, ...)

Outputs:
    - outputs/scheduling/zone_recommendations.csv
    - outputs/scheduling/high_priority_zones.csv
"""

import os
from collections import Counter

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
PEAK_RISK_THRESHOLD = 0.12       # demand >= 12% of feeder_capacity triggers peak risk
HIGH_PRIORITY_TOP_N = 8          # flag the top N zones by avg_peak_demand_kw as high priority
SHIFT_FRACTION = 0.50            # assume 50% of peak demand can be shifted
NIGHT_WINDOW_START = 22          # recommended window searched between 22:00 ...
NIGHT_WINDOW_END = 6             # ... and 06:00 (next day)
RECOMMENDED_BLOCK_HOURS = 4      # contiguous block length for recommended window
ADOPTION_LEVELS = [0.20, 0.50, 0.80]

OUTPUT_DIR = os.path.join("outputs", "scheduling")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load test predictions and zone master, merge on zone_id."""
    preds = pd.read_csv(os.path.join("outputs", "model", "test_predictions.csv"),
                        parse_dates=["timestamp"])
    zones = pd.read_csv("zone_master.csv")

    df = preds.merge(
        zones[["zone_id", "zone_name", "feeder_capacity_kw", "land_use_type"]],
        on="zone_id",
        how="left",
    )
    return df, zones


def compute_grid_headroom(df: pd.DataFrame) -> pd.DataFrame:
    """Add grid_headroom_kw = feeder_capacity_kw - predicted_demand_kw."""
    df["grid_headroom_kw"] = df["feeder_capacity_kw"] - df["predicted_demand_kw"]
    return df


def find_typical_peak_window(zone_df: pd.DataFrame, feeder_cap: float) -> str:
    """
    For a single zone, identify the most common contiguous peak risk window
    across all days in August.

    Primary threshold: hours where predicted_demand_kw >= 70% of feeder_capacity_kw.
    Fallback: if no hours exceed the feeder-based threshold (common when demand is
    well below feeder capacity), use the zone's own 75th-percentile demand as the
    threshold to identify relative peak hours.

    Returns a string like '19:00-22:00'.
    """
    zone_df = zone_df.copy()
    zone_df["date"] = zone_df["timestamp"].dt.date
    zone_df["hour"] = zone_df["timestamp"].dt.hour

    # Try feeder-based threshold first
    feeder_threshold = PEAK_RISK_THRESHOLD * feeder_cap
    has_feeder_peaks = (zone_df["predicted_demand_kw"] >= feeder_threshold).any()

    if has_feeder_peaks:
        threshold = feeder_threshold
    else:
        # Fallback: zone-relative 75th percentile
        threshold = zone_df["predicted_demand_kw"].quantile(0.75)

    zone_df["is_peak"] = zone_df["predicted_demand_kw"] >= threshold

    window_counter: Counter = Counter()

    for _, day_group in zone_df.groupby("date"):
        day_sorted = day_group.sort_values("hour")
        peak_hours = day_sorted.loc[day_sorted["is_peak"], "hour"].tolist()

        if not peak_hours:
            continue

        # Find contiguous blocks
        blocks: list[list[int]] = []
        current_block = [peak_hours[0]]
        for i in range(1, len(peak_hours)):
            if peak_hours[i] == peak_hours[i - 1] + 1:
                current_block.append(peak_hours[i])
            else:
                blocks.append(current_block)
                current_block = [peak_hours[i]]
        blocks.append(current_block)

        # Use the longest contiguous block for this day
        longest = max(blocks, key=len)
        start_h = longest[0]
        end_h = longest[-1] + 1  # end is exclusive
        window_str = f"{start_h:02d}:00-{end_h:02d}:00"
        window_counter[window_str] += 1

    if not window_counter:
        return "no_peak"

    return window_counter.most_common(1)[0][0]


def find_recommended_charging_window(zone_df: pd.DataFrame) -> str:
    """
    Find the best 4-hour contiguous block between 22:00 and 06:00
    where average grid_headroom_kw is highest across the test period.

    Night hours (mapped): 22, 23, 0, 1, 2, 3, 4, 5
    Possible 4-hour windows: 22-02, 23-03, 00-04, 01-05, 02-06
    """
    zone_df = zone_df.copy()
    zone_df["hour"] = zone_df["timestamp"].dt.hour

    # Define the night hours in order
    night_hours = [22, 23, 0, 1, 2, 3, 4, 5]

    # Average headroom per hour across all days
    hourly_headroom = zone_df.groupby("hour")["grid_headroom_kw"].mean()

    best_window = None
    best_avg_headroom = -np.inf

    # Slide a 4-hour window across the night hours
    for start_idx in range(len(night_hours) - RECOMMENDED_BLOCK_HOURS + 1):
        window_hours = night_hours[start_idx:start_idx + RECOMMENDED_BLOCK_HOURS]
        avg_headroom = hourly_headroom.reindex(window_hours).mean()
        if avg_headroom > best_avg_headroom:
            best_avg_headroom = avg_headroom
            start_h = window_hours[0]
            end_h = (window_hours[-1] + 1) % 24
            best_window = f"{start_h:02d}:00-{end_h:02d}:00"

    return best_window


def compute_shift_impact(zone_df: pd.DataFrame, feeder_cap: float) -> tuple[float, float, float]:
    """
    Compute load-shift impact if 50% of predicted peak demand is shifted.

    Returns:
        avg_peak_demand_kw: average predicted demand during peak-risk hours
        shift_impact_kw:    50% of avg_peak_demand_kw
        shift_impact_pct:   shift_impact_kw as % of feeder_capacity_kw
    """
    threshold = PEAK_RISK_THRESHOLD * feeder_cap
    zone_df = zone_df.copy()
    peak_mask = zone_df["predicted_demand_kw"] >= threshold
    peak_rows = zone_df.loc[peak_mask]

    if peak_rows.empty:
        # Fall back to the top-quartile hours as "peak" if nothing exceeds 70%
        q75 = zone_df["predicted_demand_kw"].quantile(0.75)
        peak_rows = zone_df.loc[zone_df["predicted_demand_kw"] >= q75]

    avg_peak = peak_rows["predicted_demand_kw"].mean()
    shift_kw = round(avg_peak * SHIFT_FRACTION, 2)
    shift_pct = round((shift_kw / feeder_cap) * 100, 2)

    return round(avg_peak, 2), shift_kw, shift_pct


def build_zone_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build the per-zone summary dataframe."""
    records = []

    for zone_id, zone_df in df.groupby("zone_id"):
        zone_name = zone_df["zone_name"].iloc[0]
        feeder_cap = zone_df["feeder_capacity_kw"].iloc[0]
        land_use = zone_df["land_use_type"].iloc[0]

        # 1. Typical peak window
        typical_peak = find_typical_peak_window(zone_df, feeder_cap)

        # 2. Recommended charging window
        rec_window = find_recommended_charging_window(zone_df)

        # 3. Shift impact
        avg_peak, shift_kw, shift_pct = compute_shift_impact(zone_df, feeder_cap)

        # 4. High priority flag (set after all zones are processed)
        high_priority = False  # placeholder

        # 5. Adoption scenario impacts
        adoption_impacts = {}
        for level in ADOPTION_LEVELS:
            key = f"impact_{int(level * 100)}pct"
            adoption_impacts[key] = round(avg_peak * level, 2)

        records.append({
            "zone_id": zone_id,
            "zone_name": zone_name,
            "typical_peak_window": typical_peak,
            "recommended_charging_window": rec_window,
            "avg_peak_demand_kw": avg_peak,
            "shift_impact_kw": shift_kw,
            "shift_impact_pct": shift_pct,
            "feeder_capacity_kw": feeder_cap,
            "land_use_type": land_use,
            "high_priority": high_priority,
            **adoption_impacts,
        })

    summary = pd.DataFrame(records)

    # Flag the top N zones by avg_peak_demand_kw as high priority
    top_n_idx = summary.nlargest(HIGH_PRIORITY_TOP_N, "avg_peak_demand_kw").index
    summary["high_priority"] = False
    summary.loc[top_n_idx, "high_priority"] = True

    return summary


def print_report(summary: pd.DataFrame) -> None:
    """Print the top-10 zones by shift_impact_pct and high-priority count."""
    print("=" * 80)
    print("SmartCharge.AI  --  Scheduling Recommendations Report")
    print("=" * 80)

    top10 = summary.nlargest(10, "shift_impact_pct")
    print("\nTop 10 zones by shift impact (% of feeder capacity):\n")
    print(
        top10[
            [
                "zone_id",
                "zone_name",
                "typical_peak_window",
                "recommended_charging_window",
                "avg_peak_demand_kw",
                "shift_impact_kw",
                "shift_impact_pct",
                "feeder_capacity_kw",
                "high_priority",
            ]
        ].to_string(index=False)
    )

    hp_count = summary["high_priority"].sum()
    print(f"\nTotal high-priority zones (top {HIGH_PRIORITY_TOP_N} by avg peak demand): {hp_count}")

    if hp_count > 0:
        print("\nHigh-priority zones:")
        hp = summary.loc[summary["high_priority"]]
        print(
            hp[
                [
                    "zone_id",
                    "zone_name",
                    "avg_peak_demand_kw",
                    "feeder_capacity_kw",
                    "typical_peak_window",
                    "recommended_charging_window",
                ]
            ].to_string(index=False)
        )
    print("=" * 80)


def save_outputs(summary: pd.DataFrame) -> None:
    """Save zone_recommendations.csv and high_priority_zones.csv."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rec_path = os.path.join(OUTPUT_DIR, "zone_recommendations.csv")
    summary.to_csv(rec_path, index=False)
    print(f"\nSaved: {rec_path}  ({len(summary)} zones)")

    hp = summary.loc[summary["high_priority"]]
    hp_path = os.path.join(OUTPUT_DIR, "high_priority_zones.csv")
    hp.to_csv(hp_path, index=False)
    print(f"Saved: {hp_path}  ({len(hp)} high-priority zones)")


def main() -> None:
    # 1. Load & merge
    print("Loading data...")
    df, zones = load_data()
    print(f"  Predictions: {len(df):,} rows  |  Zones: {df['zone_id'].nunique()}")

    # 2. Compute headroom
    df = compute_grid_headroom(df)

    # 3. Build zone-level summary
    print("Computing per-zone scheduling recommendations...")
    summary = build_zone_summary(df)

    # 4. Report
    print_report(summary)

    # 5. Save
    save_outputs(summary)


if __name__ == "__main__":
    main()
