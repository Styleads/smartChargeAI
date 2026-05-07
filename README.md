# SmartCharge.AI

SmartCharge.AI is a two-module AI decision support platform designed as a read-only layer for BESCOM planners and grid operators to optimize EV charging infrastructure and schedule management. Using purely synthetic data, it provides actionable, explainable intelligence without modifying existing grid systems.

---

## Architecture Overview

SmartCharge.AI follows a modular data pipeline architecture, splitting the intelligence into two core modules:

1. **Module A — ChargeCast**: Spatiotemporal EV charging demand forecasting using an XGBoost regressor, coupled with a rule-based scheduling optimizer. It predicts grid stress events and recommends optimal off-peak charging windows per zone.
2. **Module B — SiteIntel**: Geospatial demand zone scoring and candidate site optimization using multi-criteria scoring and clustering logic (haversine distancing) to recommend optimal new charging station locations.

**Data Flow**:
Raw synthetic datasets (`zone_master.csv`, `hourly_demand.csv`, `candidate_sites.csv`) are ingested by discrete pipeline scripts. These scripts output processed metrics, models, and JSON/CSV artifacts into the `outputs/` directory. The Streamlit dashboard (`app.py`) then acts as the presentation layer, dynamically loading these static output files into a multi-page interactive UI for grid operators.

---

## File Structure

```text
smartChargeAI/
├── outputs/                         # All script outputs
│   ├── eda/                         # Exploratory data plots
│   ├── forecasts/                   # 24h predictions and summaries
│   ├── model/                       # Trained model, metrics, and validation plots
│   ├── scheduling/                  # Recommended shift windows and zone priorities
│   └── siteintel/                   # Spatial surfaces, site scores, and top 10 sites
├── app.js                           # Custom frontend JS
├── app.py                           # Streamlit application entry point
├── candidate_sites.csv              # Synthetic candidate EV charging sites
├── candidate_sites.pdf              # Source PDF for candidate EV charging sites
├── eda.py                           # Exploratory Data Analysis and plotting
├── feature_engineering.py           # Creates lag, rolling, and derived features
├── featured_demand.csv              # Output of feature engineering
├── forecast_output.py               # Generates the 24-hour demand lookahead
├── hourly_demand.csv                # Synthetic hourly load profile per zone
├── index.html                       # Custom frontend HTML
├── model_training.py                # Trains XGBoost regressor, evaluates performance
├── pages_config.py                  # UI components, layout configs, and map generation
├── plot_peak_validation.py          # Generates specific validation plots
├── run_pipeline.py                  # Master script to execute the full pipeline sequentially
├── scheduling.py                    # Recommends shift windows and evaluates impact
├── siteintel.py                     # Scores zones & candidate sites, runs optimization
├── style.css                        # Custom frontend styling
└── zone_master.csv                  # Zone metadata (capacity, EV count, etc.)
```

---

## Setup Guide

### Prerequisites
- **Python**: 3.9+ recommended.
- **pip**: Standard Python package manager.

### Installation
Install all required dependencies:
```bash
pip install pandas numpy scikit-learn xgboost streamlit plotly scipy matplotlib fpdf
```

### Execution Steps
To run the full project from scratch:

1. **Run the Full Pipeline**:
   This executes Module A and Module B in the correct sequential order.
   ```bash
   python run_pipeline.py
   ```
   *(Alternatively, run scripts individually: `eda.py` → `feature_engineering.py` → `model_training.py` → `scheduling.py` → `forecast_output.py` → `siteintel.py`)*
2. **Launch the Dashboard**:
   ```bash
   streamlit run app.py
   ```

*Note: No special environment variables or API keys are required as all data is synthetic and processed locally.*

---

## Module A — ChargeCast Documentation

ChargeCast handles predictive forecasting and grid load scheduling.

### Scripts
- **`eda.py`**: Validates data integrity, computes basic stats, and generates diagnostic plots (e.g., avg demand by hour/month, demand by land-use).
  - *Inputs*: `zone_master.csv`, `hourly_demand.csv`
  - *Outputs*: Plots in `outputs/eda/`
- **`feature_engineering.py`**: Merges zone data with demand. Generates critical lag features (1h, 24h, 168h), 24h rolling means/stds, and categorical encodings.
  - *Inputs*: `zone_master.csv`, `hourly_demand.csv`
  - *Outputs*: `featured_demand.csv`
- **`model_training.py`**: Splits data chronologically (Aug 2024 as test). Trains the XGBoost Regressor (`hist` tree method). Evaluates metrics and extracts SHAP/feature importance.
  - *Inputs*: `featured_demand.csv`
  - *Outputs*: `outputs/model/model_xgb.json`, `metrics.json`, `feature_importances.csv`, plots.
- **`scheduling.py`**: Analyzes test predictions to identify peak risk windows. Recommends a 4-hour off-peak block (between 22:00 and 06:00) with maximum grid headroom to shift loads.
  - *Inputs*: `outputs/model/test_predictions.csv`, `zone_master.csv`
  - *Outputs*: `zone_recommendations.csv`, `high_priority_zones.csv`
- **`forecast_output.py`**: Uses the trained XGBoost model to generate autoregressive 24-hour demand forecasts for all 35 zones.
  - *Inputs*: `model_xgb.json`, `featured_demand.csv`, `zone_master.csv`
  - *Outputs*: `zone_forecasts_24h.csv`, `zone_forecast_summary.csv`

### Model Performance
Tested on August 2024 data:
- **R² Score**: 0.9989
- **MAE**: 0.25 kW
- **RMSE**: 0.66 kW
- **MAPE**: 1.74%

### Key EDA Findings
- `lag_168h` (1 week lag) is the dominant feature, showing a massive weekly cycle.
- Evening peaks occur predictably between 18:00 and 21:00.
- Commercial zones peak earlier in the day, while residential zones heavily drive the evening peak.

---

## Module B — SiteIntel Documentation

SiteIntel determines the best locations for deploying new charging infrastructure.

### Pipeline (`siteintel.py`)
1. **Zone Scoring**: Zones are scored on demand pressure using peak-to-capacity ratios, stress events, and growth slopes.
2. **Spatial Surface**: A multiquadric RBF interpolator generates a spatial demand surface across Bengaluru.
3. **Candidate Scoring**: 80 candidate sites are evaluated using a multi-criteria approach.
4. **Optimization**: Haversine distance rules (>1.5 km apart) ensure the top 10 selections don't cannibalize each other's coverage.
5. **Baseline Comparison**: Compares the optimized selection against a uniform, distance-based baseline grid.

### Inputs & Outputs
- *Inputs*: `zone_master.csv`, `hourly_demand.csv`, `candidate_sites.csv`
- *Outputs*: `zone_scored.csv`, `demand_surface.csv`, `siteintel_recommendations.csv`, `siteintel_top10_sites.csv`, `recommended_sites.json`

### Scoring Methodology
Opportunity Score Breakdown:
- **Demand Pressure** (35%): Based on the underlying zone's EV load pressure.
- **Grid Capacity** (25%): Availability of raw feeder capacity at the site.
- **Coverage Gap** (25%): Distance to existing EV infrastructure.
- **Accessibility** (15%): Quality of road access (highways, arterial roads).

### Baseline Comparison
The optimized SiteIntel selection outperformed a naive baseline (uniform distribution) by **+32.8%** in mean opportunity score, maximizing demand coverage while ensuring grid feasibility.

---

## Dashboard Documentation

The Streamlit dashboard (`app.py`) is a responsive, multi-page web application.

### Pages / Tabs
- **Overview**: High-level KPIs, Top 5 High-Risk Zones, and a live Leaflet.js interactive map plotting all 35 zones and the top 10 recommended sites.
- **Demand Forecast**: Select a zone to view its 72-hour projected load vs. feeder limit. Displays peak risk windows, feature importance, and model metrics.
- **Scheduling**: Interactive slider simulating 20%-80% EV adoption for load shifting. Displays calculated load reduction potential per zone.
- **Infrastructure**: Analyzes the 80 candidate sites, breaking down the Opportunity Score (Demand, Grid, Access, Coverage) for the top 10 optimized sites.
- **Zone Explorer**: Deep-dive into a specific zone. Shows live demand monitoring gauges (simulated), stress events, and specific scheduling recommendations.
- **Export & Reports**: Download pre-computed CSV artifacts and view an internal audit log of pipeline runs.

### Navigation
Use the top horizontal tabs to switch between context views. The UI leverages `pages_config.py` to render dark-mode UI elements, Plotly charts, and raw HTML/CSS injection. The dashboard reads directly from the `outputs/` directory and does not trigger model retraining.

---

## Output Files Reference

| File | Location | Content |
|---|---|---|
| `zone_forecasts_24h.csv` | `outputs/forecasts/` | Hourly 24h predictions for all 35 zones. |
| `zone_forecast_summary.csv` | `outputs/forecasts/` | One row per zone summarizing max demand and peak risk. |
| `test_predictions.csv` | `outputs/model/` | Actual vs Predicted data for the test set (Aug 2024). |
| `feature_importances.csv` | `outputs/model/` | XGBoost feature weights. |
| `metrics.json` | `outputs/model/` | Performance scores (R², MAE, RMSE). |
| `zone_recommendations.csv` | `outputs/scheduling/` | Off-peak scheduling shift recommendations. |
| `high_priority_zones.csv` | `outputs/scheduling/` | Top 8 zones needing immediate grid management. |
| `zone_scored.csv` | `outputs/siteintel/` | 35 zones scored with demand pressure metrics. |
| `demand_surface.csv` | `outputs/siteintel/` | Interpolated RBF demand grid data for maps. |
| `siteintel_recommendations.csv` | `outputs/siteintel/` | All 80 candidate sites scored and tiered. |
| `siteintel_top10_sites.csv` | `outputs/siteintel/` | The final 10 optimized sites. |
| `recommended_sites.json` | `outputs/siteintel/` | Top 10 sites converted to JSON for web maps. |

---

## Data

The platform operates on three primary **synthetic** datasets:
1. **`zone_master.csv`**: Metadata for 35 distinct zones (Whitefield, Indiranagar, etc.) covering land use, existing stations, and EV counts.
2. **`hourly_demand.csv`**: Granular time-series data covering 8 months (Jan 1, 2024 – Aug 31, 2024), totaling ~205k rows. Includes base load, charging demand, temperature, and day/time features.
3. **`candidate_sites.csv`**: 80 prospective charging locations with variables for grid capacity, land availability, and road accessibility.

---

## Known Limitations

- **Synthetic Data**: All data is synthetically generated and does not reflect actual BESCOM proprietary grid loads.
- **Stress Thresholds**: Feeder capacities in the synthetic data are relatively large compared to current demand, meaning critical stress thresholds (>90% capacity) are rarely triggered organically.
- **Priority Logic**: High-priority zones are defined dynamically as the top-8 absolute demand zones rather than by strict percentage thresholds.
- **Underprediction**: Saturday demand is slightly underpredicted in mixed-use zones due to complex weekend transition behaviors.
- **Static Telemetry**: The "Live Demand Monitoring" gauge in the dashboard simulates real-time data using localized randomization; it is not hooked into a live IoT stream.
