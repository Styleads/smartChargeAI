// Data store
let data = {
    zones: [],
    forecast: [],
    scheduling: [],
    features: [],
    sites: []
};

// Colors matching CSS
const colors = {
    primary: '#378ADD',
    risk: '#D85A30',
    positive: '#1D9E75',
    neutral: '#888780',
    accent: '#E35A28',
    bg: '#0D1117',
    text: '#E6EDF3',
    textMuted: '#8B949E',
    gridLine: 'rgba(255,255,255,0.05)'
};

// Standard Chart.js configuration
Chart.defaults.color = colors.textMuted;
Chart.defaults.font.family = "'IBM Plex Sans', 'DM Sans', sans-serif";
Chart.defaults.font.size = 11;
Chart.defaults.plugins.legend.display = false;
Chart.defaults.maintainAspectRatio = false;
Chart.defaults.scale.grid.color = colors.gridLine;

// Simple CSV Parser
async function loadCSV(url) {
    const res = await fetch(url);
    const text = await res.text();
    const lines = text.trim().split('\n');
    const headers = lines[0].split(',').map(h => h.trim());
    return lines.slice(1).filter(l => l.trim().length > 0).map(line => {
        const values = line.split(',');
        let obj = {};
        headers.forEach((h, i) => {
            obj[h] = values[i] ? values[i].trim() : '';
        });
        return obj;
    });
}

async function loadJSON(url) {
    const res = await fetch(url);
    return res.json();
}

async function init() {
    // Load all data
    data.zones = await loadCSV('outputs/siteintel/zone_scored.csv');
    data.forecast = await loadCSV('outputs/forecasts/zone_forecasts_24h.csv');
    data.scheduling = await loadCSV('outputs/scheduling/zone_recommendations.csv');
    data.features = await loadCSV('outputs/model/feature_importances.csv');
    data.sites = await loadJSON('outputs/siteintel/recommended_sites.json');

    setupNavigation();
    
    renderOverview();
    renderForecast();
    renderScheduling();
    renderInfrastructure();
    renderExplorer();
    
    setupTableSorting();
    
    // Set up event listeners
    document.getElementById('fc-zone-select').addEventListener('change', updateForecast);
    document.getElementById('sc-adoption').addEventListener('input', updateScheduling);
    document.getElementById('ex-zone-select').addEventListener('change', updateExplorer);
}

// Navigation
function setupNavigation() {
    const tabs = document.querySelectorAll('.nav-tab');
    const pages = document.querySelectorAll('.page');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            pages.forEach(p => p.classList.remove('active'));
            tab.classList.add('active');
            document.getElementById(tab.dataset.target).classList.add('active');
        });
    });
}

// Table Sorting Setup
function setupTableSorting() {
    document.querySelectorAll('table.sortable').forEach(table => {
        const headers = table.querySelectorAll('th[data-col]');
        let currentSort = { col: null, asc: true };
        
        headers.forEach(th => {
            th.addEventListener('click', () => {
                const col = th.dataset.col;
                if (currentSort.col === col) {
                    currentSort.asc = !currentSort.asc;
                } else {
                    currentSort.col = col;
                    currentSort.asc = true;
                }
                
                // Update header icons
                headers.forEach(h => h.innerHTML = h.dataset.origText || h.innerHTML);
                if (!th.dataset.origText) th.dataset.origText = th.innerHTML;
                th.innerHTML = th.dataset.origText + (currentSort.asc ? ' ↑' : ' ↓');
                
                // Sort tbody
                const tbody = table.querySelector('tbody');
                const rows = Array.from(tbody.querySelectorAll('tr'));
                rows.sort((a, b) => {
                    let valA = a.dataset[col] || '';
                    let valB = b.dataset[col] || '';
                    
                    const numA = parseFloat(valA);
                    const numB = parseFloat(valB);
                    if (!isNaN(numA) && !isNaN(numB)) {
                        return currentSort.asc ? numA - numB : numB - numA;
                    }
                    return currentSort.asc ? valA.localeCompare(valB) : valB.localeCompare(valA);
                });
                tbody.innerHTML = '';
                rows.forEach(r => tbody.appendChild(r));
            });
        });
    });
}

// Row creation helper
function createRow(rowObj, colDefs) {
    const tr = document.createElement('tr');
    colDefs.forEach(def => {
        tr.dataset[def.col] = rowObj[def.col] || '';
        const td = document.createElement('td');
        td.className = def.class || '';
        td.innerHTML = def.fmt ? def.fmt(rowObj[def.col]) : rowObj[def.col];
        tr.appendChild(td);
    });
    return tr;
}

// --- Page Renderers ---

function renderOverview() {
    const highRisk = data.zones.filter(z => parseFloat(z.demand_pressure_score) >= 75).length;
    let stressTotal = 0;
    data.zones.forEach(z => stressTotal += parseInt(z.stress_event_count || 0));
    
    document.getElementById('ov-monitored').textContent = data.zones.length;
    document.getElementById('ov-highrisk').textContent = highRisk;
    document.getElementById('ov-sites').textContent = data.sites.length;
    document.getElementById('ov-stress').textContent = stressTotal;
    
    const topZones = [...data.zones].sort((a,b) => parseFloat(b.demand_pressure_score) - parseFloat(a.demand_pressure_score)).slice(0,5);
    const topTbody = document.querySelector('#ov-top-zones tbody');
    topZones.forEach(z => {
        topTbody.appendChild(createRow(z, [
            {col: 'zone_name'},
            {col: 'demand_pressure_score', class: 'num', fmt: v => parseFloat(v).toFixed(1)}
        ]));
    });
    
    const allTbody = document.querySelector('#ov-all-zones tbody');
    data.zones.forEach(z => {
        allTbody.appendChild(createRow(z, [
            {col: 'zone_id'},
            {col: 'zone_name'},
            {col: 'land_use_type', fmt: v => v.toUpperCase()},
            {col: 'ev_count_current', class: 'num'},
            {col: 'feeder_capacity_kw', class: 'num', fmt: v => parseFloat(v).toFixed(0)},
            {col: 'avg_peak_demand_kw', class: 'num', fmt: v => parseFloat(v).toFixed(1)},
            {col: 'stress_event_count', class: 'num'},
            {col: 'demand_pressure_score', class: 'num', fmt: v => parseFloat(v).toFixed(1)}
        ]));
    });

    const scores = data.zones.map(z => parseFloat(z.demand_pressure_score));
    const bins = [0,0,0,0,0,0,0,0,0,0];
    scores.forEach(s => {
        let idx = Math.min(Math.floor(s/10), 9);
        bins[idx]++;
    });
    const binLabels = bins.map((_, i) => `${i*10}-${(i+1)*10}`);
    const binColors = bins.map((_, i) => i >= 7 ? colors.accent : colors.primary);
    
    new Chart(document.getElementById('ov-dist-chart'), {
        type: 'bar',
        data: {
            labels: binLabels,
            datasets: [{
                data: bins,
                backgroundColor: binColors,
                borderRadius: 4
            }]
        },
        options: {
            scales: {
                y: { beginAtZero: true, grid: { color: colors.gridLine } },
                x: { grid: { display: false } }
            }
        }
    });
}

let fcLineChart, fcFeatChart;

function renderForecast() {
    const sel = document.getElementById('fc-zone-select');
    data.zones.forEach(z => {
        const opt = document.createElement('option');
        opt.value = z.zone_id;
        opt.textContent = z.zone_name;
        sel.appendChild(opt);
    });
    
    const topFeats = [...data.features].sort((a,b) => parseFloat(b.importance) - parseFloat(a.importance)).slice(0, 10);
    topFeats.reverse();
    
    fcFeatChart = new Chart(document.getElementById('fc-feat-chart'), {
        type: 'bar',
        data: {
            labels: topFeats.map(f => f.feature),
            datasets: [{
                data: topFeats.map(f => parseFloat(f.importance)),
                backgroundColor: topFeats.map((f, i) => i === topFeats.length-1 ? colors.accent : colors.primary),
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            scales: {
                x: { grid: { color: colors.gridLine } },
                y: { grid: { display: false } }
            }
        }
    });
    
    updateForecast();
}

function updateForecast() {
    const zoneId = document.getElementById('fc-zone-select').value;
    const zone = data.zones.find(z => z.zone_id === zoneId);
    
    document.getElementById('fc-score').textContent = parseFloat(zone.demand_pressure_score).toFixed(1);
    document.getElementById('fc-cap').textContent = Math.round(zone.feeder_capacity_kw);
    document.getElementById('fc-peak').textContent = parseFloat(zone.avg_peak_demand_kw).toFixed(1);
    document.getElementById('fc-stress').textContent = zone.stress_event_count;
    
    const zFc = data.forecast.filter(f => f.zone_id === zoneId).slice(0, 24);
    
    if (fcLineChart) fcLineChart.destroy();
    fcLineChart = new Chart(document.getElementById('fc-line-chart'), {
        type: 'line',
        data: {
            labels: zFc.map(f => {
                let d = new Date(f.forecast_timestamp);
                return d.getHours() + ':00';
            }),
            datasets: [
                {
                    label: 'Predicted Demand',
                    data: zFc.map(f => parseFloat(f.predicted_demand_kw)),
                    borderColor: colors.primary,
                    backgroundColor: 'rgba(55,138,221,0.1)',
                    fill: true,
                    tension: 0.4,
                    pointBackgroundColor: zFc.map(f => parseInt(f.hour) >= 18 && parseInt(f.hour) <= 22 ? colors.risk : colors.primary),
                    pointRadius: zFc.map(f => parseInt(f.hour) >= 18 && parseInt(f.hour) <= 22 ? 4 : 0)
                },
                {
                    label: 'Capacity',
                    data: zFc.map(f => parseFloat(f.feeder_capacity_kw)),
                    borderColor: colors.risk,
                    borderDash: [5,5],
                    pointRadius: 0,
                    fill: false
                }
            ]
        },
        options: {
            interaction: { intersect: false, mode: 'index' },
            scales: {
                y: { beginAtZero: true, grid: { color: colors.gridLine } },
                x: { grid: { display: false } }
            }
        }
    });
}

let scBarChart;

function renderScheduling() {
    updateScheduling();
}

function updateScheduling() {
    const adoption = parseInt(document.getElementById('sc-adoption').value);
    document.getElementById('sc-adoption-val').textContent = adoption;
    
    let tableData = data.scheduling.map(s => {
        let z = data.zones.find(zone => zone.zone_id === s.zone_id) || {};
        let shiftBase = parseFloat(s.shift_impact_kw);
        let calculated = shiftBase * (adoption / 50.0);
        return {
            ...s,
            demand_pressure_score: z.demand_pressure_score || 0,
            calculated_reduction: calculated
        };
    }).sort((a,b) => b.demand_pressure_score - a.demand_pressure_score);
    
    let maxRed = Math.max(...tableData.map(d => d.calculated_reduction));
    document.getElementById('sc-max-red').textContent = maxRed.toFixed(1) + ' kW';
    
    const tbody = document.querySelector('#sc-table tbody');
    tbody.innerHTML = '';
    tableData.forEach(d => {
        tbody.appendChild(createRow(d, [
            {col: 'zone_name'},
            {col: 'typical_peak_window'},
            {col: 'recommended_charging_window'},
            {col: 'demand_pressure_score', class: 'num', fmt: v => parseFloat(v).toFixed(1)},
            {col: 'shift_impact_kw', class: 'num', fmt: v => parseFloat(v).toFixed(1)},
            {col: 'calculated_reduction', class: 'num text-success', fmt: v => parseFloat(v).toFixed(1)}
        ]));
    });
    
    // Sort logic requires re-adding sort listeners or just initial sort. 
    // Data is added dynamically so clicking header will resort correctly.
    
    const top10 = tableData.slice(0, 10);
    if(scBarChart) scBarChart.destroy();
    
    scBarChart = new Chart(document.getElementById('sc-bar-chart'), {
        type: 'bar',
        data: {
            labels: top10.map(d => d.zone_name),
            datasets: [{
                data: top10.map(d => d.calculated_reduction),
                backgroundColor: colors.positive,
                borderRadius: 4
            }]
        },
        options: {
            scales: {
                y: { beginAtZero: true, grid: { color: colors.gridLine } },
                x: { grid: { display: false } }
            }
        }
    });
}

function renderInfrastructure() {
    document.getElementById('in-sites').textContent = data.sites.length;
    
    const tbody = document.querySelector('#in-table tbody');
    data.sites.forEach(s => {
        tbody.appendChild(createRow(s, [
            {col: 'site_name'},
            {col: 'zone_id'},
            {col: 'confidence_tier'},
            {col: 'opportunity_score', class:'num', fmt: v => parseFloat(v).toFixed(1)},
            {col: 'demand_contribution', class:'num', fmt: v => parseFloat(v).toFixed(1)},
            {col: 'grid_contribution', class:'num', fmt: v => parseFloat(v).toFixed(1)},
            {col: 'accessibility_contribution', class:'num', fmt: v => parseFloat(v).toFixed(1)},
            {col: 'coverage_contribution', class:'num', fmt: v => parseFloat(v).toFixed(1)},
            {col: 'available_grid_capacity_kw', class:'num', fmt: v => parseFloat(v).toFixed(0)}
        ]));
    });
    
    const top10 = [...data.sites].sort((a,b) => b.opportunity_score - a.opportunity_score).slice(0, 10);
    
    new Chart(document.getElementById('in-stack-chart'), {
        type: 'bar',
        data: {
            labels: top10.map(s => s.site_name.substring(0,20) + '...'),
            datasets: [
                { label: 'Demand', data: top10.map(s => s.demand_contribution), backgroundColor: colors.risk },
                { label: 'Grid', data: top10.map(s => s.grid_contribution), backgroundColor: colors.primary },
                { label: 'Access', data: top10.map(s => s.accessibility_contribution), backgroundColor: colors.positive },
                { label: 'Coverage', data: top10.map(s => s.coverage_contribution), backgroundColor: colors.accent }
            ]
        },
        options: {
            scales: {
                x: { stacked: true, grid: { display: false } },
                y: { stacked: true, grid: { color: colors.gridLine } }
            }
        }
    });
}

let exBarChart;

function renderExplorer() {
    const sel = document.getElementById('ex-zone-select');
    data.zones.forEach(z => {
        const opt = document.createElement('option');
        opt.value = z.zone_id;
        opt.textContent = z.zone_name;
        sel.appendChild(opt);
    });
    
    updateExplorer();
}

function updateExplorer() {
    const zoneId = document.getElementById('ex-zone-select').value;
    const zone = data.zones.find(z => z.zone_id === zoneId);
    const sched = data.scheduling.find(s => s.zone_id === zoneId);
    
    document.getElementById('ex-ev').textContent = zone.ev_count_current;
    document.getElementById('ex-score').textContent = parseFloat(zone.demand_pressure_score).toFixed(1);
    document.getElementById('ex-peak').textContent = parseFloat(zone.avg_peak_demand_kw).toFixed(1) + ' kW';
    document.getElementById('ex-stress').textContent = zone.stress_event_count;
    
    document.getElementById('ex-shift-win').textContent = sched.recommended_charging_window;
    document.getElementById('ex-shift-amt').textContent = parseFloat(sched.shift_impact_kw).toFixed(1) + ' kW';
    
    document.getElementById('ex-ad-20').textContent = parseFloat(sched.impact_20pct).toFixed(1) + ' kW';
    document.getElementById('ex-ad-50').textContent = parseFloat(sched.impact_50pct).toFixed(1) + ' kW';
    document.getElementById('ex-ad-80').textContent = parseFloat(sched.impact_80pct).toFixed(1) + ' kW';
    
    if(exBarChart) exBarChart.destroy();
    exBarChart = new Chart(document.getElementById('ex-bar-chart'), {
        type: 'bar',
        data: {
            labels: ['Current', '6 Months', '12 Months'],
            datasets: [{
                data: [parseFloat(zone.avg_peak_demand_kw), parseFloat(zone.projected_6m_kw), parseFloat(zone.projected_12m_kw)],
                backgroundColor: [colors.neutral, colors.primary, colors.risk],
                borderRadius: 4
            }]
        },
        options: {
            scales: {
                y: { beginAtZero: true, grid: { color: colors.gridLine } },
                x: { grid: { display: false } }
            }
        }
    });
}

// Boot
window.onload = init;
