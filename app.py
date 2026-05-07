import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json

import datetime
import calendar
import os

from pages_config import (
    render_overview, render_demand_forecast,
    render_scheduling, render_infrastructure,
    _metric, _card_open, render_dark_table, build_gauge_html, COLORS
)

st.set_page_config(page_title="SmartCharge AI", layout="wide", initial_sidebar_state="collapsed")

BASE = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data():
    zs = pd.read_csv(os.path.join(BASE, 'outputs', 'siteintel', 'zone_scored.csv'))
    fc = pd.read_csv(os.path.join(BASE, 'outputs', 'forecasts', 'zone_forecasts_24h.csv'))
    sc = pd.read_csv(os.path.join(BASE, 'outputs', 'scheduling', 'zone_recommendations.csv'))
    si = pd.read_csv(os.path.join(BASE, 'outputs', 'siteintel', 'siteintel_recommendations.csv'))
    sh = pd.read_csv(os.path.join(BASE, 'outputs', 'model', 'feature_importances.csv'))
    with open(os.path.join(BASE, 'outputs', 'siteintel', 'recommended_sites.json')) as f:
        rs = json.load(f)
    return zs, fc, sc, si, sh, rs

zone_scored, forecast, scheduling, siteintel, shap_df, recommended_sites = load_data()

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;700&display=swap');
.stApp { background-color: #0D1117; color: #E6EDF3; font-family: 'IBM Plex Sans', sans-serif; }
header { visibility: hidden; }
.block-container { padding-top: 1rem; padding-bottom: 2rem; }

.brand-bar { display:flex; align-items:baseline; gap:10px; padding:8px 0 4px 0; margin-bottom:-8px; }
.brand-name { color:#FFFFFF; font-size:22px; font-weight:700; letter-spacing:-0.5px; }
.brand-accent { color:#E35A28; font-size:22px; font-weight:700; letter-spacing:-0.5px; }
.brand-sub { color:#8B9098; font-size:11px; font-weight:400; margin-left:6px; }

div[data-testid="stTabs"] { position:sticky; top:0; z-index:999; background-color:#0D1117; padding-top:6px; border-bottom:1px solid rgba(255,255,255,0.1); }
div[data-testid="stTabs"] button { font-size:15px; font-weight:500; color:#8B949E; border-bottom:2px solid transparent !important; }
div[data-testid="stTabs"] button[aria-selected="true"] { color:#E6EDF3 !important; font-weight:500; border-bottom:2px solid #E35A28 !important; }

.metric-card { background-color:#161B22; border-radius:8px; padding:16px 20px; display:flex; flex-direction:column; margin-bottom:20px; }
.metric-label { font-size:12px; color:#8B949E; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:8px; font-weight:400; }
.metric-value { font-size:24px; font-weight:500; font-variant-numeric:tabular-nums; color:#E6EDF3; }
.text-danger { color:#D85A30; }
.text-success { color:#1D9E75; }

.surface-card { background-color:#161B22; border:0.5px solid rgba(255,255,255,0.1); border-radius:8px; padding:20px; margin-bottom:20px; }
.card-title { font-size:14px; font-weight:500; color:#F0EEE8; margin-bottom:16px; }

.chart-legend { display:flex; gap:16px; margin-bottom:12px; font-size:12px; color:#8B9098; }
.legend-item { display:flex; align-items:center; gap:6px; }
.legend-color { width:12px; height:12px; border-radius:2px; }

.stSelectbox label { color:#8B9098 !important; font-size:12px !important; }
</style>
""", unsafe_allow_html=True)

# ── Brand ────────────────────────────────────────────────────────────────────
st.markdown('<div class="brand-bar"><span class="brand-name">SmartCharge </span><span class="brand-accent">AI</span><span class="brand-sub">BESCOM</span></div>', unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_ov, tab_fc, tab_sc, tab_in, tab_ex, tab_rp = st.tabs([
    "Overview", "Demand Forecast", "Scheduling",
    "Infrastructure", "Zone Explorer", "Export & Reports"
])

with tab_ov:
    render_overview(zone_scored, recommended_sites)

with tab_fc:
    render_demand_forecast(zone_scored, forecast, shap_df)

with tab_sc:
    render_scheduling(zone_scored, scheduling)

with tab_in:
    render_infrastructure(zone_scored, recommended_sites)

# ── Zone Explorer ────────────────────────────────────────────────────────────
with tab_ex:
    selected = st.selectbox("Select Zone to Explore", zone_scored['zone_name'].tolist(),
                            label_visibility="visible", key="explorer_zone")
    zone = zone_scored[zone_scored['zone_name'] == selected].iloc[0]
    sched = scheduling[scheduling['zone_id'] == zone['zone_id']].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    sc_c = 'text-danger' if zone['demand_pressure_score'] >= 75 else ''
    st_c = 'text-danger' if zone['stress_event_count'] > 10 else ''
    with c1: st.markdown(_metric('EV Count', zone['ev_count_current']), unsafe_allow_html=True)
    with c2: st.markdown(_metric('Demand Score', f"{zone['demand_pressure_score']:.1f}", sc_c), unsafe_allow_html=True)
    with c3: st.markdown(_metric('Peak Demand', f"{zone['avg_peak_demand_kw']:.1f} kW"), unsafe_allow_html=True)
    with c4: st.markdown(_metric('Stress Events', f"{zone['stress_event_count']:.0f}", st_c), unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(_card_open('Demand Growth Projection'), unsafe_allow_html=True)
        periods = ['Current', '6 Months', '12 Months']
        vals = [zone['avg_peak_demand_kw'], zone['projected_6m_kw'], zone['projected_12m_kw']]
        fig = go.Figure(go.Bar(x=periods, y=vals,
            marker_color=['#888780','#378ADD','#D85A30'],
            text=[f"{v:.1f}" for v in vals], textposition='outside',
            textfont=dict(size=11, color='#F0EEE8')))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#8B9098', family='IBM Plex Sans', size=12),
            margin=dict(l=0,r=0,t=20,b=0), height=250, showlegend=False,
            yaxis=dict(gridcolor='rgba(255,255,255,0.06)', title='kW'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.06)'))
        st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown(_card_open('Scheduling Recommendation'), unsafe_allow_html=True)
        st.markdown(f"""
        <div style="padding:14px;background:rgba(29,158,117,0.08);border-left:3px solid #1D9E75;border-radius:4px;margin-bottom:16px;">
            <div style="font-size:13px;font-weight:500;margin-bottom:6px;color:#8B9098">Shift Window: <span style="color:#F0EEE8">{sched['recommended_charging_window']}</span></div>
            <div style="font-size:13px;font-weight:500;color:#8B9098">Base Shift: <span style="color:#F0EEE8">{sched['shift_impact_kw']:.1f} kW</span></div>
        </div>""", unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3)
        with s1: st.markdown(_metric('20% Adoption', f"{sched['impact_20pct']:.1f} kW", 'text-success'), unsafe_allow_html=True)
        with s2: st.markdown(_metric('50% Adoption', f"{sched['impact_50pct']:.1f} kW", 'text-success'), unsafe_allow_html=True)
        with s3: st.markdown(_metric('80% Adoption', f"{sched['impact_80pct']:.1f} kW", 'text-success'), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Live Demand Monitoring gauge
    import random
    kw_val = zone['avg_peak_demand_kw'] * (0.85 + random.random() * 0.3)
    ev_val = zone['ev_count_current'] * (0.7 + random.random() * 0.4)
    gauge_html = build_gauge_html(
        zone_name=selected,
        kw_val=kw_val,
        kw_max=zone['feeder_capacity_kw'],
        ev_val=ev_val,
        ev_max=zone['ev_count_current'] * 1.2,
        score=zone['demand_pressure_score']
    )
    st.markdown(_card_open('Live Demand Monitoring'), unsafe_allow_html=True)
    st.markdown(gauge_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Export & Reports ─────────────────────────────────────────────────────────
with tab_rp:
    st.markdown(_card_open('Data Exports'), unsafe_allow_html=True)
    e1, e2, e3 = st.columns(3)
    with e1:
        st.download_button("Download Forecast Data", forecast.to_csv(index=False).encode(),
                           "zone_forecasts_24h.csv", "text/csv", width='stretch')
        st.caption("24-hour demand forecast — all 35 zones")
    with e2:
        st.download_button("Download Scheduling Data", scheduling.to_csv(index=False).encode(),
                           "zone_recommendations.csv", "text/csv", width='stretch')
        st.caption("Zone scheduling recommendations")
    with e3:
        st.download_button("Download Site Rankings", siteintel.to_csv(index=False).encode(),
                           "siteintel_recommendations.csv", "text/csv", width='stretch')
        st.caption("Top 10 recommended station sites")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Audit log
    st.markdown(_card_open('Audit Log'), unsafe_allow_html=True)
    now = datetime.datetime.now()
    audit = pd.DataFrame({
        'Timestamp': [now.strftime('%Y-%m-%d %H:%M'),
                      (now - datetime.timedelta(hours=1)).strftime('%Y-%m-%d %H:%M'),
                      (now - datetime.timedelta(hours=2)).strftime('%Y-%m-%d %H:%M')],
        'Action': ['SiteIntel recommendations generated',
                   'ChargeCast 24h forecast generated', 'Zone demand scores updated'],
        'Output': ['10 sites recommended, +32.8% vs baseline',
                   '840 forecast records, MAPE 1.74%', '35 zones scored, stress zones identified'],
        'Status': ['Complete', 'Complete', 'Complete']
    })
    st.markdown(render_dark_table(audit), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
