"""
Converts candidate_sites.pdf into candidate_sites.csv
Uses pdfplumber to extract tabular data from all pages.

Requirements: pip install pdfplumber
"""

import csv
import pdfplumber


def pdf_to_csv(pdf_path: str, csv_path: str) -> None:
    """Extract table data from a PDF and write it to a CSV file."""
    all_rows: list[list[str]] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            tables = page.extract_tables()

            if not tables:
                print(f"  ⚠ No tables found on page {page_num + 1}, skipping.")
                continue

            # Use the largest table on each page
            table = max(tables, key=len)

            for i, row in enumerate(table):
                # First row of the first page is the header
                if page_num == 0 and i == 0:
                    all_rows.append(row)  # header
                else:
                    # Skip rows that look like repeated headers
                    if row and row[0] == "site_id":
                        continue
                    all_rows.append(row)

    # Write to CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(all_rows)

    data_rows = len(all_rows) - 1  # subtract header
    cols = len(all_rows[0]) if all_rows else 0
    print(f"✅ Converted '{pdf_path}' → '{csv_path}'")
    print(f"   {data_rows} data rows, {cols} columns")
    print(f"   Columns: {', '.join(all_rows[0])}")


if __name__ == "__main__":
    pdf_to_csv("candidate_sites.pdf", "candidate_sites.csv")
import pandas as pd
import numpy as np
import math

np.random.seed(42)

# ============================================================
# DATASET 1 — zone_master.csv
# ============================================================

zones_raw = [
    ("Whitefield", 12.9698, 77.7499), ("Koramangala", 12.9352, 77.6245),
    ("HSR Layout", 12.9116, 77.6389), ("Indiranagar", 12.9784, 77.6408),
    ("Electronic City", 12.8399, 77.6770), ("Marathahalli", 12.9591, 77.6972),
    ("Bellandur", 12.9261, 77.6762), ("Sarjapur Road", 12.9010, 77.6849),
    ("BTM Layout", 12.9166, 77.6101), ("Jayanagar", 12.9308, 77.5836),
    ("Banashankari", 12.9255, 77.5468), ("JP Nagar", 12.9063, 77.5857),
    ("Hebbal", 13.0354, 77.5970), ("Yelahanka", 13.1007, 77.5963),
    ("Bagalur", 13.1490, 77.7810), ("KR Puram", 13.0050, 77.6940),
    ("Mahadevapura", 12.9940, 77.7100), ("Hoskote", 13.0704, 77.7980),
    ("Rajajinagar", 12.9902, 77.5560), ("Malleswaram", 13.0035, 77.5665),
    ("Yeshwanthpur", 13.0220, 77.5510), ("Tumkur Road", 13.0600, 77.5100),
    ("Peenya", 13.0290, 77.5190), ("Dasarahalli", 13.0440, 77.5090),
    ("Kengeri", 12.9074, 77.4822), ("Uttarahalli", 12.8937, 77.5412),
    ("Mysore Road", 12.9500, 77.5000), ("Bannerghatta Road", 12.8800, 77.5970),
    ("Electronic City Phase 2", 12.8270, 77.6760), ("Bommanahalli", 12.8960, 77.6360),
    ("Devanahalli", 13.2460, 77.7120), ("Nelamangala", 13.0980, 77.3920),
    ("Anekal", 12.7100, 77.6960), ("Varthur", 12.9400, 77.7460),
    ("Domlur", 12.9602, 77.6391),
]

land_use_map = {
    "Whitefield": "mixed", "Koramangala": "commercial", "HSR Layout": "mixed",
    "Indiranagar": "commercial", "Electronic City": "industrial", "Marathahalli": "mixed",
    "Bellandur": "mixed", "Sarjapur Road": "residential", "BTM Layout": "residential",
    "Jayanagar": "residential", "Banashankari": "residential", "JP Nagar": "residential",
    "Hebbal": "mixed", "Yelahanka": "residential", "Bagalur": "logistics",
    "KR Puram": "mixed", "Mahadevapura": "industrial", "Hoskote": "logistics",
    "Rajajinagar": "residential", "Malleswaram": "commercial", "Yeshwanthpur": "commercial",
    "Tumkur Road": "logistics", "Peenya": "industrial", "Dasarahalli": "industrial",
    "Kengeri": "residential", "Uttarahalli": "residential", "Mysore Road": "logistics",
    "Bannerghatta Road": "mixed", "Electronic City Phase 2": "industrial",
    "Bommanahalli": "residential", "Devanahalli": "logistics", "Nelamangala": "logistics",
    "Anekal": "industrial", "Varthur": "residential", "Domlur": "commercial",
}

existing_stations = {
    "Koramangala": 6, "Indiranagar": 5, "Whitefield": 8, "Electronic City": 7,
    "HSR Layout": 4, "Marathahalli": 5, "Domlur": 3, "Malleswaram": 3,
    "BTM Layout": 4, "JP Nagar": 3, "Hebbal": 4, "Yelahanka": 2,
    "Jayanagar": 3, "Banashankari": 2,
}

tech_corridor = {"Whitefield", "Electronic City", "Marathahalli", "Bellandur",
                 "Sarjapur Road", "Mahadevapura", "Bagalur"}
commercial_hubs = {"Koramangala", "Indiranagar", "Domlur", "Malleswaram"}
rapidly_developing = {"Bagalur", "Sarjapur Road", "Varthur", "Devanahalli", "Hoskote"}
established_tech = tech_corridor - rapidly_developing

zone_records = []
for i, (name, lat, lon) in enumerate(zones_raw):
    lu = land_use_map[name]

    # ev_count_current
    if name in tech_corridor:
        ev = np.random.randint(2000, 4001)
    elif name in commercial_hubs:
        ev = np.random.randint(1200, 2501)
    elif lu == "residential":
        ev = np.random.randint(600, 1801)
    else:
        ev = np.random.randint(200, 901)

    # ev_growth_rate_monthly
    if name in rapidly_developing:
        gr = np.random.uniform(0.045, 0.06)
    elif name in established_tech:
        gr = np.random.uniform(0.03, 0.045)
    elif lu in ("residential", "commercial"):
        gr = np.random.uniform(0.02, 0.035)
    else:
        gr = np.random.uniform(0.015, 0.025)

    # feeder_capacity_kw
    if lu == "industrial":
        fc = np.random.randint(1800, 3001)
    elif lu == "mixed":
        fc = np.random.randint(1200, 2201)
    elif lu == "commercial":
        fc = np.random.randint(1000, 1801)
    elif lu == "residential":
        fc = np.random.randint(500, 1201)
    else:  # logistics/peripheral
        fc = np.random.randint(600, 1501)

    # existing_station_count
    esc = existing_stations.get(name, np.random.randint(0, 3))

    # population_density
    if lu == "residential":
        pd_ = "high"
    elif lu in ("commercial", "mixed"):
        pd_ = "medium"
    else:
        pd_ = "low"

    zone_records.append({
        "zone_id": f"Z{i+1:03d}",
        "zone_name": name,
        "latitude": lat,
        "longitude": lon,
        "land_use_type": lu,
        "ev_count_current": ev,
        "ev_growth_rate_monthly": round(gr, 4),
        "feeder_capacity_kw": fc,
        "existing_station_count": esc,
        "population_density": pd_,
    })

zone_df = pd.DataFrame(zone_records)

print(f"zone_master.csv — {len(zone_df)} rows")
print(zone_df.head(3))
print()

# ============================================================
# DATASET 2 — hourly_demand.csv
# ============================================================

# Build time axis
start = pd.Timestamp("2024-01-01 00:00")
end = pd.Timestamp("2024-08-31 23:00")
timestamps = pd.date_range(start, end, freq="h")
n_hours = len(timestamps)  # 5856

# Hour multiplier lookup
hour_mult = np.array([
    0.15, 0.15, 0.15, 0.15, 0.15, 0.15,  # 0-5
    0.55, 0.55, 0.55,                      # 6-8
    0.70, 0.70, 0.70,                      # 9-11
    0.65, 0.65, 0.65,                      # 12-14
    0.75, 0.75, 0.75,                      # 15-17
    1.00, 1.30, 1.50, 1.40,               # 18-21
    0.80, 0.40,                            # 22-23
])

# Temperature params by month (1-indexed → index 0 unused)
temp_mean = {1: 22, 2: 25, 3: 28, 4: 30, 5: 29, 6: 25, 7: 23, 8: 23}
temp_std = {1: 2, 2: 2, 3: 2.5, 4: 2, 5: 2, 6: 1.5, 7: 1.5, 8: 1.5}

# Holiday dates
holidays = {
    pd.Timestamp("2024-01-26"), pd.Timestamp("2024-03-25"),
    pd.Timestamp("2024-04-14"), pd.Timestamp("2024-04-17"),
    pd.Timestamp("2024-05-01"), pd.Timestamp("2024-06-17"),
    pd.Timestamp("2024-08-15"), pd.Timestamp("2024-08-26"),
}

# Pre-compute time features
hours = timestamps.hour.values
dow = timestamps.dayofweek.values
is_weekend = (dow >= 5).astype(int)
months = timestamps.month.values
dates = timestamps.normalize()
is_holiday = np.array([1 if d in holidays else 0 for d in dates])

# Hour-based temperature offset
temp_hour_offset = np.where((hours >= 4) & (hours <= 6), -3,
                   np.where((hours >= 13) & (hours <= 15), 2, 0)).astype(float)

# Non-EV peak factor: gaussian peaks at hour 12 and 19
non_ev_peak = np.where(
    (hours >= 8) & (hours <= 22),
    0.60 + 0.25 * (np.exp(-0.5 * ((hours - 12) / 3.0) ** 2) +
                    np.exp(-0.5 * ((hours - 19) / 2.5) ** 2)),
    0.35
)

# Stress zones and conditions
stress_zones = {"Whitefield", "Koramangala", "Electronic City", "Marathahalli",
                "HSR Layout", "Bellandur", "Indiranagar", "Sarjapur Road"}

# Build per-zone data vectorised and concatenate
all_chunks = []

for _, zrow in zone_df.iterrows():
    zid = zrow["zone_id"]
    zname = zrow["zone_name"]
    lu = zrow["land_use_type"]
    ev = zrow["ev_count_current"]
    gr = zrow["ev_growth_rate_monthly"]
    fc = zrow["feeder_capacity_kw"]

    base_demand = ev * 0.018

    # Day type multiplier
    if lu == "commercial":
        day_mult = np.where(is_weekend, 0.65, 1.0)
    elif lu == "residential":
        day_mult = np.where(is_weekend, 1.10, 0.85)
    elif lu == "mixed":
        day_mult = np.where(is_weekend, 0.85, 1.0)
    else:  # industrial / logistics
        day_mult = np.where(is_weekend, 0.45, 1.0)

    # Month growth
    month_growth = (1 + gr) ** (months - 1)

    # Noise
    noise = np.random.normal(1.0, 0.08, size=n_hours)
    noise = np.clip(noise, 0.05, None)

    # Charging demand
    charging = base_demand * hour_mult[hours] * day_mult * month_growth * noise

    # Grid total load
    non_ev_base = fc * 0.42
    day_factor = np.where(is_weekend, 0.90, 1.0)
    non_ev_load = non_ev_base * non_ev_peak * day_factor
    grid_total = non_ev_load + charging

    # Stress multiplier for specific zones during peak weekday summer
    if zname in stress_zones:
        stress_mask = (
            (hours >= 18) & (hours <= 21) &
            (is_weekend == 0) &
            (months >= 5) & (months <= 8)
        )
        grid_total = np.where(stress_mask, grid_total * 1.15, grid_total)

    # Clip at 105% feeder capacity
    grid_total = np.clip(grid_total, 0, fc * 1.05)

    # Temperature
    month_means = np.array([temp_mean[m] for m in months])
    month_stds = np.array([temp_std[m] for m in months])
    temperature = np.random.normal(month_means, month_stds) + temp_hour_offset
    temperature = np.round(temperature, 1)

    chunk = pd.DataFrame({
        "zone_id": zid,
        "timestamp": timestamps,
        "hour": hours,
        "day_of_week": dow,
        "is_weekend": is_weekend,
        "month": months,
        "charging_demand_kw": np.round(charging, 2),
        "grid_total_load_kw": np.round(grid_total, 2),
        "temperature_celsius": temperature,
        "is_holiday": is_holiday,
    })
    all_chunks.append(chunk)

hourly_df = pd.concat(all_chunks, ignore_index=True)

print(f"hourly_demand.csv — {len(hourly_df)} rows")
print(hourly_df.head(3))
print()

# ============================================================
# DATASET 3 — candidate_sites.csv
# ============================================================

high_priority = {"Whitefield", "Koramangala", "Electronic City", "Marathahalli",
                 "HSR Layout", "Bellandur", "Sarjapur Road", "Indiranagar",
                 "Hebbal", "Yelahanka"}

site_type_pool = (
    ["parking_lot"] * 25 + ["commercial_complex"] * 20 +
    ["metro_station"] * 15 + ["fuel_station"] * 15 +
    ["residential_complex"] * 15 + ["highway_node"] * 10
)

# Descriptive name parts per site_type
name_parts = {
    "parking_lot": ["Metro Parking Lot", "Mall Parking Deck", "Public Parking Facility", "IT Park Lot"],
    "commercial_complex": ["Forum Mall Basement", "Brigade Gateway", "Phoenix Marketcity Lot", "Mantri Square Bay"],
    "metro_station": ["Metro Station Forecourt", "Namma Metro Hub", "Metro East Exit Plaza", "Metro Park-and-Ride"],
    "fuel_station": ["HP Fuel Station", "Indian Oil Pump", "Bharat Petroleum Station", "Shell Fuel Point"],
    "residential_complex": ["Prestige Apartments Gate", "Sobha Dream Acres Lot", "Brigade Meadows Entry", "Purva Skydale Bay"],
    "highway_node": ["NH Junction Node", "Ring Road Service Area", "Toll Plaza EV Bay", "Highway Rest Stop"],
}

# Assign site counts per zone
zone_site_counts = {}
for _, zrow in zone_df.iterrows():
    zname = zrow["zone_name"]
    if zname in high_priority:
        zone_site_counts[zname] = np.random.choice([3, 4])
    else:
        lu = zrow["land_use_type"]
        if lu in ("mixed", "commercial"):
            zone_site_counts[zname] = 2
        else:
            zone_site_counts[zname] = np.random.choice([1, 2])

# Adjust to hit ~80 total
total = sum(zone_site_counts.values())
# Fine-tune: add or remove from medium zones
while total < 80:
    for zn in zone_site_counts:
        if zn not in high_priority and zone_site_counts[zn] < 3:
            zone_site_counts[zn] += 1
            total += 1
            if total >= 80:
                break
while total > 80:
    for zn in reversed(list(zone_site_counts.keys())):
        if zn not in high_priority and zone_site_counts[zn] > 1:
            zone_site_counts[zn] -= 1
            total -= 1
            if total <= 80:
                break

site_records = []
site_counter = 0
np.random.shuffle(site_type_pool)  # shuffle for variety
pool_idx = 0

for _, zrow in zone_df.iterrows():
    zid = zrow["zone_id"]
    zname = zrow["zone_name"]
    zlat = zrow["latitude"]
    zlon = zrow["longitude"]
    esc = zrow["existing_station_count"]
    count = zone_site_counts[zname]

    for j in range(count):
        site_counter += 1
        st = site_type_pool[pool_idx % len(site_type_pool)]
        pool_idx += 1

        # Site name
        descriptor = np.random.choice(name_parts[st])
        sname = f"{zname} {descriptor}"

        # Coordinates
        slat = round(zlat + np.random.uniform(-0.015, 0.015), 4)
        slon = round(zlon + np.random.uniform(-0.015, 0.015), 4)

        # available_grid_capacity_kw
        cap_ranges = {
            "metro_station": (200, 500), "commercial_complex": (150, 400),
            "parking_lot": (100, 350), "fuel_station": (100, 300),
            "residential_complex": (50, 200), "highway_node": (200, 500),
        }
        lo, hi = cap_ranges[st]
        agc = np.random.randint(lo, hi + 1)

        # road_accessibility_score
        acc_ranges = {
            "highway_node": (0.80, 1.0), "fuel_station": (0.80, 1.0),
            "metro_station": (0.70, 0.95), "commercial_complex": (0.70, 0.95),
            "parking_lot": (0.55, 0.85), "residential_complex": (0.40, 0.75),
        }
        alo, ahi = acc_ranges[st]
        ras = round(np.random.uniform(alo, ahi), 2)

        # distance_to_nearest_station_km
        if esc >= 4:
            dist = round(np.random.uniform(0.3, 1.5), 1)
        elif esc >= 1:
            dist = round(np.random.uniform(1.0, 4.0), 1)
        else:
            dist = round(np.random.uniform(3.0, 8.0), 1)

        # land_availability
        if st in ("metro_station", "fuel_station"):
            la = np.random.choice(["available", "pending_approval", "restricted"],
                                  p=[0.75, 0.15, 0.10])
        else:
            la = np.random.choice(["available", "pending_approval", "restricted"],
                                  p=[0.55, 0.30, 0.15])

        site_records.append({
            "site_id": f"S{site_counter:03d}",
            "zone_id": zid,
            "site_name": sname,
            "latitude": slat,
            "longitude": slon,
            "site_type": st,
            "available_grid_capacity_kw": agc,
            "road_accessibility_score": ras,
            "distance_to_nearest_station_km": dist,
            "land_availability": la,
        })

sites_df = pd.DataFrame(site_records)

print(f"candidate_sites.csv — {len(sites_df)} rows")
print(sites_df.head(3))
print()

# ============================================================
# SAVE
# ============================================================

zone_df.to_csv("zone_master.csv", index=False)
hourly_df.to_csv("hourly_demand.csv", index=False)
sites_df.to_csv("candidate_sites.csv", index=False)

print("All datasets generated successfully.")
