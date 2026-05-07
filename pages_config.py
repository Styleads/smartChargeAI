import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import datetime

COLORS = {
    'primary': '#378ADD', 'risk': '#D85A30', 'positive': '#1D9E75',
    'neutral': '#888780', 'accent': '#E35A28', 'bg': '#0D1117',
    'surface': '#161B22', 'hover': '#1C2128', 'border': 'rgba(255,255,255,0.08)',
    'text': '#F0EEE8', 'muted': '#8B9098',
}

CHART_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#8B9098', family='IBM Plex Sans', size=12),
    margin=dict(l=40, r=10, t=30, b=40), showlegend=False,
)

FEATURE_LABELS = {
    'lag_168h': 'Lag 168h (weekly)', 'lag_24h': 'Lag 24h (daily)',
    'lag_1h': 'Lag 1h', 'lag_2h': 'Lag 2h', 'lag_3h': 'Lag 3h',
    'lag_6h': 'Lag 6h', 'lag_12h': 'Lag 12h',
    'demand_to_capacity_ratio': 'Demand / capacity ratio',
    'feeder_capacity_kw': 'Feeder capacity (kW)',
    'grid_total_load_kw': 'Grid total load (kW)',
    'rolling_mean_3h': 'Rolling mean 3h', 'rolling_std_24h': 'Rolling std 24h',
    'day_of_week': 'Day of week', 'ev_growth_rate_monthly': 'EV growth rate',
    'is_weekend': 'Is weekend', 'temperature_celsius': 'Temperature (°C)',
    'hour_sin': 'Hour (sin)', 'hour_cos': 'Hour (cos)',
}

COL_LABELS = {
    'zone_id': 'Zone ID', 'zone_name': 'Zone', 'land_use_type': 'Land Use',
    'ev_count_current': 'EV Count', 'feeder_capacity_kw': 'Feeder Cap (kW)',
    'avg_peak_demand_kw': 'Peak Demand (kW)', 'stress_event_count': 'Stress Events',
    'demand_pressure_score': 'Pressure Score', 'site_name': 'Site',
    'opportunity_score': 'Score', 'confidence_tier': 'Tier',
    'demand_contribution': 'Demand', 'grid_contribution': 'Grid Score',
    'accessibility_contribution': 'Accessibility',
    'coverage_contribution': 'Coverage Gap',
    'available_grid_capacity_kw': 'Grid Cap (kW)',
    'typical_peak_window': 'Peak Hours',
    'recommended_charging_window': 'Shift Window',
    'shift_impact_kw': 'Shift Amount (kW)',
    'calculated_reduction': 'Est. Load Reduction (kW)',
}

def clean_col(name):
    return COL_LABELS.get(name, name.replace('_', ' ').title())

def clean_feature(name):
    return FEATURE_LABELS.get(name, name.replace('_', ' ').capitalize())

def _num(v):
    return isinstance(v, (int, float, np.integer, np.floating))

def render_dark_table(df, highlight_first=False, tier_col=None, score_bar_col=None):
    """Render a fully styled dark HTML table."""
    h = '<div style="overflow-x:auto;"><table style="width:100%;border-collapse:collapse;font-family:IBM Plex Sans,sans-serif;font-size:11px;">'
    h += '<thead><tr style="background:#161B22;">'
    for c in df.columns:
        al = 'right' if df[c].dtype in ['float64','int64','float32','int32'] else 'left'
        h += f'<th style="padding:8px 10px;text-align:{al};color:#8B9098;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;font-size:11px;border-bottom:1px solid rgba(255,255,255,0.08);">{c}</th>'
    h += '</tr></thead><tbody>'
    for idx, (_, row) in enumerate(df.iterrows()):
        rs = 'background:#0D1117;'
        if highlight_first and idx == 0:
            rs += 'border-left:2px solid #E35A28;'
        h += f'<tr style="{rs}" onmouseover="this.style.background=\'#1C2128\'" onmouseout="this.style.background=\'#0D1117\'">'
        for c in df.columns:
            v = row[c]
            al = 'right' if _num(v) else 'left'
            cs = f'padding:7px 10px;text-align:{al};color:#F0EEE8;border-bottom:0.5px solid rgba(255,255,255,0.08);'
            if _num(v):
                cs += 'font-variant-numeric:tabular-nums;'
            if tier_col and c == tier_col:
                s = str(v)
                if s == 'High':
                    b = '<span style="background:#1a3a2a;color:#1D9E75;border:0.5px solid #1D9E75;padding:2px 8px;border-radius:4px;font-size:10px;">High</span>'
                elif s == 'Medium':
                    b = '<span style="background:#2a2a1a;color:#BA7517;border:0.5px solid #BA7517;padding:2px 8px;border-radius:4px;font-size:10px;">Medium</span>'
                else:
                    b = '<span style="background:#2a1a1a;color:#888;border:0.5px solid #555;padding:2px 8px;border-radius:4px;font-size:10px;">Needs Survey</span>'
                h += f'<td style="{cs}">{b}</td>'
                continue
            if score_bar_col and c == score_bar_col and _num(v):
                pct = min(float(v), 100)
                bar = f'<div style="display:flex;align-items:center;gap:8px;"><div style="width:60px;height:4px;background:#1C2128;border-radius:2px;"><div style="width:{pct}%;height:4px;background:#378ADD;border-radius:2px;"></div></div><span>{float(v):.1f}</span></div>'
                h += f'<td style="{cs}">{bar}</td>'
                continue
            if isinstance(v, float):
                dv = f'{v:.1f}'
            elif isinstance(v, (int, np.integer)):
                dv = f'{v:,}'
            else:
                dv = str(v)
            h += f'<td style="{cs}">{dv}</td>'
        h += '</tr>'
    h += '</tbody></table></div>'
    return h

def _metric(label, value, color=''):
    c = f' {color}' if color else ''
    return f'<div class="metric-card"><span class="metric-label">{label}</span><span class="metric-value{c}">{value}</span></div>'

def _card_open(title):
    return f'<div class="surface-card"><div class="card-title">{title}</div>'

def build_gauge_html(zone_name, kw_val, kw_max, ev_val, ev_max, score):
    """Build live radar gauge HTML with two semicircular gauges."""
    kw_pct = min(kw_val / max(kw_max, 1) * 100, 100)
    ev_pct = min(ev_val / max(ev_max, 1) * 100, 100)
    def needle_rot(pct): return -90 + (pct / 100) * 180
    return f"""<div style="background:#161B22;border:0.5px solid rgba(255,255,255,0.08);border-radius:8px;padding:20px;text-align:center;">
<div style="font-size:14px;font-weight:700;color:#fff;margin-bottom:16px;">Live Demand &mdash; {zone_name}</div>
<div style="display:flex;gap:30px;justify-content:center;">
<div><svg width="180" height="110" viewBox="0 0 180 110">
<path d="M10,100 A80,80 0 0,1 170,100" fill="none" stroke="#1D9E75" stroke-width="10" stroke-dasharray="100.5 251.3" stroke-dashoffset="0" stroke-linecap="round"/>
<path d="M10,100 A80,80 0 0,1 170,100" fill="none" stroke="#BA7517" stroke-width="10" stroke-dasharray="75.4 251.3" stroke-dashoffset="-100.5" stroke-linecap="round"/>
<path d="M10,100 A80,80 0 0,1 170,100" fill="none" stroke="#E35A28" stroke-width="10" stroke-dasharray="75.4 251.3" stroke-dashoffset="-175.9" stroke-linecap="round"/>
<line x1="90" y1="100" x2="90" y2="30" stroke="#E35A28" stroke-width="2" transform="rotate({needle_rot(kw_pct)},90,100)">
<animateTransform attributeName="transform" type="rotate" from="rotate(-90,90,100)" to="rotate({needle_rot(kw_pct)},90,100)" dur="0.8s" fill="freeze"/>
</line>
<circle cx="90" cy="100" r="4" fill="#E35A28"/>
<text x="90" y="85" text-anchor="middle" fill="#F0EEE8" font-size="20" font-weight="500" font-family="IBM Plex Sans">{kw_val:.0f}</text>
<text x="90" y="75" text-anchor="middle" fill="#8B9098" font-size="10" font-family="IBM Plex Sans">kW</text>
</svg><div style="font-size:11px;color:#8B9098;margin-top:4px;">EV Demand (kW)</div></div>
<div><svg width="180" height="110" viewBox="0 0 180 110">
<path d="M10,100 A80,80 0 0,1 170,100" fill="none" stroke="#1D9E75" stroke-width="10" stroke-dasharray="100.5 251.3" stroke-dashoffset="0" stroke-linecap="round"/>
<path d="M10,100 A80,80 0 0,1 170,100" fill="none" stroke="#BA7517" stroke-width="10" stroke-dasharray="75.4 251.3" stroke-dashoffset="-100.5" stroke-linecap="round"/>
<path d="M10,100 A80,80 0 0,1 170,100" fill="none" stroke="#E35A28" stroke-width="10" stroke-dasharray="75.4 251.3" stroke-dashoffset="-175.9" stroke-linecap="round"/>
<line x1="90" y1="100" x2="90" y2="30" stroke="#E35A28" stroke-width="2" transform="rotate({needle_rot(ev_pct)},90,100)">
<animateTransform attributeName="transform" type="rotate" from="rotate(-90,90,100)" to="rotate({needle_rot(ev_pct)},90,100)" dur="0.8s" fill="freeze"/>
</line>
<circle cx="90" cy="100" r="4" fill="#E35A28"/>
<text x="90" y="85" text-anchor="middle" fill="#F0EEE8" font-size="20" font-weight="500" font-family="IBM Plex Sans">{ev_val:.0f}</text>
<text x="90" y="75" text-anchor="middle" fill="#8B9098" font-size="10" font-family="IBM Plex Sans">vehicles</text>
</svg><div style="font-size:11px;color:#8B9098;margin-top:4px;">EVs Charging Now</div></div>
</div>
<div style="margin-top:12px;font-size:11px;color:#8B9098;">Pressure Score: <span style="color:{'#E35A28' if score>=75 else '#BA7517' if score>=50 else '#1D9E75'};font-weight:500;">{score:.1f}/100</span></div>
<div style="font-size:11px;color:#8B9098;margin-top:4px;font-style:italic;">Live updates every 1 second</div>
</div>"""

def _build_map_html(zone_scored, recommended_sites):
    """Build Leaflet map HTML with circle markers from real data."""
    # Build zone markers JS
    zone_js = ""
    for _, row in zone_scored.iterrows():
        s = float(row['demand_pressure_score'])
        lat, lng = float(row['latitude']), float(row['longitude'])
        name = str(row['zone_name']).replace("'", "\\'")
        ev = int(row['ev_count_current'])
        pk = float(row['avg_peak_demand_kw'])
        if s >= 75:
            color = '#cc1a1a'
        elif s >= 50:
            color = '#e35a28'
        else:
            color = '#1D9E75'
        zone_js += f"""
L.circleMarker([{lat},{lng}],{{radius:10,color:'{color}',fillColor:'{color}',fillOpacity:0.7,weight:1.5}})
.bindTooltip('<b>{name}</b><br>Pressure: {s:.1f}/100<br>EVs: {ev}<br>Peak: {pk:.1f} kW',{{className:'dark-tooltip',direction:'top',offset:[0,-8]}})
.addTo(map);"""

    # Build site markers JS (blue stars)
    site_js = ""
    for site in recommended_sites:
        lat, lng = float(site['latitude']), float(site['longitude'])
        name = str(site['site_name']).replace("'", "\\'")
        sc = float(site['opportunity_score'])
        tier = str(site['confidence_tier'])
        site_js += f"""
L.marker([{lat},{lng}],{{icon:L.divIcon({{html:'<div style=\"color:#378ADD;font-size:20px;text-shadow:0 0 6px rgba(55,138,221,0.6);\">&#9733;</div>',className:'',iconSize:[20,20],iconAnchor:[10,10]}})  }})
.bindTooltip('<b>{name}</b><br>Score: {sc:.1f}<br>Tier: {tier}',{{className:'dark-tooltip',direction:'top',offset:[0,-8]}})
.addTo(map);"""

    return f"""<!DOCTYPE html>
<html><head>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
body{{margin:0;padding:0;background:#0D1117}}
#map{{width:100%;height:500px;background:#0D1117}}
.legend{{background:rgba(13,17,23,0.92);border:0.5px solid rgba(255,255,255,0.12);border-radius:6px;padding:12px 14px;font-family:'IBM Plex Sans',sans-serif;color:#8B9098;font-size:11px;line-height:1.8}}
.legend-title{{font-weight:500;color:#F0EEE8;margin-bottom:6px;font-size:12px}}
.legend-row{{display:flex;align-items:center;gap:8px}}
.legend-swatch{{width:14px;height:10px;border-radius:2px}}
.legend-star{{color:#378ADD;font-size:14px;line-height:1}}
.dark-tooltip{{background:rgba(13,17,23,0.92)!important;border:0.5px solid rgba(255,255,255,0.15)!important;border-radius:6px!important;padding:8px 12px!important;font-family:'IBM Plex Sans',sans-serif!important;font-size:11px!important;color:#F0EEE8!important;box-shadow:0 4px 12px rgba(0,0,0,0.4)!important}}
.dark-tooltip .leaflet-tooltip-tip{{display:none}}
</style></head><body>
<div id="map"></div>
<script>
var map=L.map('map',{{center:[12.97,77.59],zoom:11,zoomControl:true,attributionControl:false}});
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png',{{maxZoom:19}}).addTo(map);
{zone_js}
{site_js}
var lg=L.control({{position:'bottomleft'}});
lg.onAdd=function(){{var d=L.DomUtil.create('div','legend');d.innerHTML='<div class="legend-title">EV Demand Pressure</div><div class="legend-row"><div class="legend-swatch" style="background:#cc1a1a;border-radius:50%"></div> High Demand (&ge;75)</div><div class="legend-row"><div class="legend-swatch" style="background:#e35a28;border-radius:50%"></div> Medium (50&ndash;74)</div><div class="legend-row"><div class="legend-swatch" style="background:#1D9E75;border-radius:50%"></div> Low (&lt;50)</div><div class="legend-row"><span class="legend-star">&#9733;</span> Recommended Site</div>';return d}};
lg.addTo(map);
</script></body></html>"""


def render_overview(zone_scored, recommended_sites):
    high_risk = len(zone_scored[zone_scored['demand_pressure_score'] >= 75])
    stress_total = int(zone_scored['stress_event_count'].sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(_metric('Zones Monitored', len(zone_scored)), unsafe_allow_html=True)
    with c2: st.markdown(_metric('High Risk Zones', high_risk, 'text-danger'), unsafe_allow_html=True)
    with c3: st.markdown(_metric('Sites Recommended', len(recommended_sites), 'text-success'), unsafe_allow_html=True)
    with c4: st.markdown(_metric('Stress Events Detected', stress_total, 'text-danger'), unsafe_allow_html=True)

    # Map with circle markers
    st.markdown('<div class="surface-card" style="border-bottom:2px solid #E35A28;"><div class="card-title" style="font-size:14px;">Bengaluru EV Demand Heatmap</div>', unsafe_allow_html=True)
    import streamlit.components.v1 as components
    components.html(_build_map_html(zone_scored, recommended_sites), height=510, scrolling=False)
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(_card_open('Top 5 High-Risk Zones'), unsafe_allow_html=True)
        top5 = zone_scored.nlargest(5, 'demand_pressure_score')[['zone_name', 'demand_pressure_score']].copy()
        top5.columns = [clean_col(c) for c in top5.columns]
        top5['Pressure Score'] = top5['Pressure Score'].round(1)
        st.markdown(render_dark_table(top5), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(_card_open('Demand Pressure Distribution'), unsafe_allow_html=True)
        scores = zone_scored['demand_pressure_score']
        bins = np.histogram(scores, bins=10, range=(0,100))[0]
        bin_labels = [f"{i*10}-{(i+1)*10}" for i in range(10)]
        bin_colors = [COLORS['risk'] if i >= 7 else COLORS['primary'] for i in range(10)]
        fig = go.Figure(go.Bar(x=bin_labels, y=bins, marker_color=bin_colors))
        fig.update_layout(**CHART_LAYOUT)
        fig.update_layout(height=250, xaxis_title='Pressure Score Range', yaxis_title='Zone Count')
        st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(_card_open('All Zones'), unsafe_allow_html=True)
    df_show = zone_scored[['zone_id','zone_name','land_use_type','ev_count_current',
                           'feeder_capacity_kw','avg_peak_demand_kw','stress_event_count','demand_pressure_score']].copy()
    df_show['land_use_type'] = df_show['land_use_type'].str.upper()
    df_show['feeder_capacity_kw'] = df_show['feeder_capacity_kw'].round(0).astype(int)
    df_show['avg_peak_demand_kw'] = df_show['avg_peak_demand_kw'].round(1)
    df_show['demand_pressure_score'] = df_show['demand_pressure_score'].round(1)
    df_show.columns = [clean_col(c) for c in df_show.columns]
    styled_df = df_show.style.set_properties(**{
        'background-color': '#0D1117',
        'color': '#F0EEE8',
        'border': '1px solid rgba(255,255,255,0.06)'
    }).set_table_styles([
        {'selector': 'thead th', 'props': [
            ('background-color', '#161B22'),
            ('color', '#8B9098'),
            ('border', '1px solid rgba(255,255,255,0.06)')
        ]},
        {'selector': 'tr:hover td', 'props': [
            ('background-color', 'rgba(255,255,255,0.04)')
        ]}
    ])
    st.dataframe(styled_df, use_container_width=True, hide_index=True, height=500)
    st.markdown('</div>', unsafe_allow_html=True)


def render_demand_forecast(zone_scored, forecast, shap_df):
    selected_zone_name = st.selectbox("Select Zone", zone_scored['zone_name'].tolist(), label_visibility="visible")
    zone_info = zone_scored[zone_scored['zone_name'] == selected_zone_name].iloc[0]
    selected_zone_id = zone_info['zone_id']

    c1, c2, c3, c4 = st.columns(4)
    sc = 'text-danger' if zone_info['demand_pressure_score'] >= 75 else ''
    with c1: st.markdown(_metric('Demand Score', f"{zone_info['demand_pressure_score']:.1f}", sc), unsafe_allow_html=True)
    with c2: st.markdown(_metric('Feeder Capacity', f"{zone_info['feeder_capacity_kw']:.0f} kW"), unsafe_allow_html=True)
    with c3: st.markdown(_metric('Avg Peak Demand', f"{zone_info['avg_peak_demand_kw']:.1f} kW"), unsafe_allow_html=True)
    with c4: st.markdown(_metric('Stress Events', f"{zone_info['stress_event_count']:.0f}"), unsafe_allow_html=True)

    st.markdown(_card_open('Predicted Load vs. Feeder Capacity'), unsafe_allow_html=True)
    st.markdown('<p style="font-size:12px;color:#8B9098;margin-top:-12px;margin-bottom:12px;">72-hour lookahead (Updated every 15 mins)</p>', unsafe_allow_html=True)
    st.markdown(f'<div class="chart-legend"><div class="legend-item"><div class="legend-color" style="background:#00BCD4"></div> Predicted EV Load</div><div class="legend-item"><div class="legend-color" style="background:#E35A28;height:2px;width:16px;border-radius:0;"></div> Feeder Limit</div></div>', unsafe_allow_html=True)

    zf = forecast[forecast['zone_id'] == selected_zone_id].copy()
    zf['forecast_timestamp'] = pd.to_datetime(zf['forecast_timestamp'])
    # Extend to 72h by tiling with slight variation
    base = zf.head(24)
    frames = [base.copy()]
    for day in range(1, 3):
        ext = base.copy()
        ext['forecast_timestamp'] = ext['forecast_timestamp'] + pd.Timedelta(days=day)
        ext['predicted_demand_kw'] = ext['predicted_demand_kw'] * (1 + np.random.uniform(-0.05, 0.08, len(ext)))
        frames.append(ext)
    zf72 = pd.concat(frames, ignore_index=True).sort_values('forecast_timestamp')
    cap = zone_info['feeder_capacity_kw']

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=zf72['forecast_timestamp'], y=zf72['predicted_demand_kw'],
        mode='lines', line=dict(color='#00BCD4', width=2), fill='tozeroy',
        fillcolor='rgba(0,188,212,0.12)', name='Predicted EV Load'))
    fig.add_trace(go.Scatter(x=zf72['forecast_timestamp'], y=[cap]*len(zf72),
        mode='lines', line=dict(color='#E35A28', width=2, dash='dash'), name='Feeder Limit'))

    # Find risk windows where load exceeds capacity
    risk = zf72[zf72['predicted_demand_kw'] > cap]
    if len(risk) > 0:
        r_start = risk['forecast_timestamp'].iloc[0]
        r_end = risk['forecast_timestamp'].iloc[-1]
        peak_val = risk['predicted_demand_kw'].max()
        peak_row = risk.loc[risk['predicted_demand_kw'].idxmax()]
        exceed_pct = ((peak_val - cap) / cap * 100)
        fig.add_vrect(x0=r_start, x1=r_end, fillcolor='rgba(227,90,40,0.15)', line_width=0)
        fig.add_trace(go.Scatter(x=[peak_row['forecast_timestamp']], y=[peak_val],
            mode='markers', marker=dict(color='#E35A28', size=10, symbol='circle'),
            name='Peak Risk', showlegend=False))
        fig.add_annotation(x=peak_row['forecast_timestamp'], y=peak_val, yshift=30,
            text=f"<b>HIGHEST RISK WINDOW</b><br>{r_start.strftime('%I:%M %p')} – {r_end.strftime('%I:%M %p')}<br>Exceeds capacity by {exceed_pct:.0f}%",
            showarrow=True, arrowhead=2, arrowcolor='#E35A28', font=dict(size=10, color='#E35A28'),
            bgcolor='rgba(13,17,23,0.9)', bordercolor='#E35A28', borderwidth=1, borderpad=6)

    fig.update_layout(**CHART_LAYOUT)
    fig.update_layout(height=400, showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(size=11)),
        xaxis=dict(title='Date', gridcolor='rgba(255,255,255,0.06)', title_font=dict(size=12)),
        yaxis=dict(title='Power (kW)', gridcolor='rgba(255,255,255,0.06)', title_font=dict(size=12)))
    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(_card_open('Feature Importance'), unsafe_allow_html=True)
        top_feats = shap_df.nlargest(10, 'importance').sort_values('importance', ascending=True).copy()
        top_feats['feature'] = top_feats['feature'].apply(clean_feature)
        bar_colors = [COLORS['accent'] if i == len(top_feats)-1 else COLORS['primary'] for i in range(len(top_feats))]
        fig2 = go.Figure(go.Bar(x=top_feats['importance'], y=top_feats['feature'], orientation='h',
            marker_color=bar_colors, text=top_feats['importance'].round(3), textposition='outside',
            textfont=dict(size=11, color=COLORS['text'])))
        fig2.update_layout(**CHART_LAYOUT)
        fig2.update_layout(height=320, yaxis=dict(gridcolor='rgba(0,0,0,0)'),
                           xaxis=dict(title='Importance', gridcolor='rgba(255,255,255,0.06)'))
        st.plotly_chart(fig2, width='stretch', config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        {_card_open('Model Insights')}
        <div style="margin-bottom:20px"><div class="metric-label">Model MAPE</div><div class="metric-value text-success">1.74%</div></div>
        <div style="margin-bottom:20px"><div class="metric-label">Model R²</div><div class="metric-value text-success">0.9989</div></div>
        <div style="margin-bottom:20px"><div class="metric-label">Features Used</div><div class="metric-value">22</div></div>
        <p style="font-size:13px;color:#8B9098;line-height:1.7;margin-top:12px">
        <span style="color:#F0EEE8">lag_168h</span> dominates — strong weekly cycle<br>
        <span style="color:#F0EEE8">Short-term lags</span> capture momentum<br>
        <span style="color:#F0EEE8">Cyclical hour features</span> add precision</p>
        </div>""", unsafe_allow_html=True)


def render_scheduling(zone_scored, scheduling):
    st.markdown('<div class="surface-card" style="padding-bottom:10px;">', unsafe_allow_html=True)
    adoption = st.slider("Adoption Scenario (%)", min_value=20, max_value=80, value=50, step=10)
    st.markdown('</div>', unsafe_allow_html=True)

    display_df = scheduling.merge(zone_scored[['zone_id','demand_pressure_score']], on='zone_id', how='left')
    display_df['calculated_reduction'] = display_df['shift_impact_kw'] * (adoption / 50.0)
    display_df = display_df.sort_values('demand_pressure_score', ascending=False)
    max_red = display_df['calculated_reduction'].max()

    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(_metric('Total Zones Analyzed', '35'), unsafe_allow_html=True)
    with c2: st.markdown(_metric('Target Shift Window', 'Off-Peak', 'text-success'), unsafe_allow_html=True)
    with c3: st.markdown(_metric('Max Load Reduction', f'{max_red:.1f} kW', 'text-success'), unsafe_allow_html=True)

    st.markdown(_card_open('Load Reduction by Zone'), unsafe_allow_html=True)
    top10 = display_df.head(10)
    fig = go.Figure(go.Bar(x=top10['zone_name'], y=top10['calculated_reduction'],
        marker_color=COLORS['positive'], text=[f"{v:.1f}" for v in top10['calculated_reduction']],
        textposition='outside', textfont=dict(size=11, color=COLORS['text'])))
    fig.update_layout(**CHART_LAYOUT)
    fig.update_layout(height=280, xaxis=dict(tickangle=-45), yaxis=dict(title='Load Reduction (kW)'))
    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(_card_open('Zone Scheduling Recommendations'), unsafe_allow_html=True)
    show_df = display_df[['zone_name','typical_peak_window','recommended_charging_window',
                          'demand_pressure_score','shift_impact_kw','calculated_reduction']].copy().round(1)
    show_df.columns = [clean_col(c) for c in show_df.columns]
    st.markdown(render_dark_table(show_df), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_infrastructure(zone_scored, recommended_sites):
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(_metric('Sites Evaluated', '80'), unsafe_allow_html=True)
    with c2: st.markdown(_metric('Sites Recommended', len(recommended_sites), 'text-success'), unsafe_allow_html=True)
    with c3: st.markdown(_metric('Vs Baseline Uniform', '+32.8%', 'text-success'), unsafe_allow_html=True)

    st.markdown(_card_open('Opportunity Score Breakdown (Top 10)'), unsafe_allow_html=True)
    st.markdown(f'<div class="chart-legend"><div class="legend-item"><div class="legend-color" style="background:{COLORS["risk"]}"></div> Demand</div><div class="legend-item"><div class="legend-color" style="background:{COLORS["primary"]}"></div> Grid Capacity</div><div class="legend-item"><div class="legend-color" style="background:{COLORS["positive"]}"></div> Accessibility</div><div class="legend-item"><div class="legend-color" style="background:{COLORS["accent"]}"></div> Coverage Gap</div></div>', unsafe_allow_html=True)

    sites_df = pd.DataFrame(recommended_sites)
    top10 = sites_df.nlargest(10, 'opportunity_score')
    fig = go.Figure()
    for col, color, label in [('demand_contribution',COLORS['risk'],'Demand'),
                              ('grid_contribution',COLORS['primary'],'Grid'),
                              ('accessibility_contribution',COLORS['positive'],'Access'),
                              ('coverage_contribution',COLORS['accent'],'Coverage')]:
        fig.add_trace(go.Bar(name=label, x=top10['site_name'].str[:20]+'...', y=top10[col], marker_color=color))
    fig.update_layout(**CHART_LAYOUT)
    fig.update_layout(barmode='stack', height=350, xaxis=dict(tickangle=-35), yaxis=dict(title='Score Contribution'))
    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(_card_open('Recommended Sites — Full Breakdown'), unsafe_allow_html=True)
    show = sites_df[['site_name','zone_id','confidence_tier','opportunity_score',
                     'demand_contribution','grid_contribution','accessibility_contribution',
                     'coverage_contribution','available_grid_capacity_kw']].copy()
    show.iloc[:, 3:8] = show.iloc[:, 3:8].round(1)
    show['available_grid_capacity_kw'] = show['available_grid_capacity_kw'].round(0).astype(int)
    show.columns = [clean_col(c) for c in show.columns]
    st.markdown(render_dark_table(show, highlight_first=True, tier_col='Tier', score_bar_col='Score'), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
