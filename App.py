import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px
# ── Load the real dataset once — used as exact simulation candidates ──────────
@st.cache_data
def load_dataset():
    df = pd.read_csv("retrofit_dataset_final_solution1.csv")
    return df

DATASET = load_dataset()


# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Retrofit AI",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding: 1rem 1.5rem;}

    [data-testid="metric-container"] {
        background: #f0f4f8;
        border: 1px solid #c9d4e0;
        border-radius: 10px;
        padding: 12px 16px;
    }
    [data-testid="metric-container"] label {
        font-size: 13px !important;
        font-weight: 600 !important;
        color: #1a1a2e !important;
    }
    [data-testid="metric-container"] [data-testid="metric-value"] {
        font-size: 22px !important;
        font-weight: 800 !important;
        color: #0a0a0a !important;
    }

    .city-selected {
        background: #1a1a2e;
        border: 2px solid #1a1a2e;
        border-radius: 10px;
        padding: 10px 16px;
        font-size: 18px;
        font-weight: 800;
        color: #ffffff;
        text-align: center;
        margin-bottom: 8px;
        letter-spacing: 0.01em;
    }
    .city-prompt {
        background: #f0f4f8;
        border: 2px dashed #9aabb8;
        border-radius: 10px;
        padding: 10px 16px;
        font-size: 14px;
        font-weight: 600;
        color: #3a3a4a;
        text-align: center;
        margin-bottom: 8px;
    }
    .section-label {
        font-size: 11px;
        font-weight: 800;
        color: #3a3a4a;
        text-transform: uppercase;
        letter-spacing: 0.10em;
        margin-bottom: 8px;
    }
    .warn {
        color: #b91c1c;
        font-size: 13px;
        font-weight: 700;
    }
    .result-group-title {
        font-size: 15px;
        font-weight: 800;
        color: #0a0a0a;
        margin: 18px 0 12px;
        padding-bottom: 8px;
        border-bottom: 3px solid #0a0a0a;
        letter-spacing: -0.01em;
    }
    .var-card {
        border: 1.5px solid #c9d4e0;
        border-radius: 12px;
        padding: 12px 14px;
        background: #ffffff;
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 9px;
    }
    .var-icon {
        font-size: 20px;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 10px;
        flex-shrink: 0;
    }
    .var-icon-building { background: #dce8f8; }
    .var-icon-economic { background: #fef3c7; }
    .var-info { flex: 1; min-width: 0; }
    .var-symbol {
        display: inline-block;
        font-size: 11px;
        font-weight: 800;
        font-family: 'Courier New', monospace;
        background: #1a1a2e;
        color: #ffffff;
        padding: 1px 6px;
        border-radius: 4px;
        letter-spacing: 0.03em;
        margin-right: 5px;
    }
    .var-symbol-econ { background: #78350f; color: #ffffff; }
    .var-label {
        font-size: 12px;
        font-weight: 600;
        color: #3a3a4a;
        margin-bottom: 3px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .var-value {
        font-size: 17px;
        font-weight: 800;
        color: #0a0a0a;
        letter-spacing: -0.01em;
        line-height: 1.2;
    }
    .var-range { font-size: 11px; font-weight: 600; color: #64748b; margin-top: 2px; }
    .bar-wrap  { width: 72px; flex-shrink: 0; text-align: right; }
    .bar-track { height: 6px; background: #e2e8f0; border-radius: 99px; overflow: hidden; margin-bottom: 3px; }
    .bar-fill  { height: 6px; border-radius: 99px; }
    .bar-pct   { font-size: 11px; font-weight: 700; color: #3a3a4a; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# VARIABLE DEFINITIONS
# ─────────────────────────────────────────
CITIES = {
    "Toronto":     (43.6532, -79.3832),
    "Vancouver":   (49.2827, -123.1207),
    "Montreal":    (45.5017, -73.5673),
    "Calgary":     (51.0447, -114.0719),
    "StJohns":     (47.5615, -52.7126),
    "Halifax":     (44.6488, -63.5752),
    "Winnipeg":    (49.8951, -97.1384),
    "Saskatoon":   (52.1332, -106.6700),
    "Whitehorse":  (60.7212, -135.0568),
    "Yellowknife": (62.4540, -114.3718),
}

# ── FIXED: Loan/Rebate ranges corrected to match the training dataset ────────
RANGES = {
    "V_bites":      (0.05, 0.25),
    "Albedo_roof":  (0.10, 0.70),
    "A_ST":         (0.10, 0.60),
    "Rvalue_roof":  (5.46, 11.0),
    "Loan":         (0,    10000),   # was (0, 50000) — swapped with Rebate
    "Rebate":       (20000, 50000),  # was (0, 10000) — swapped with Loan
    "Rvalue_wall":  (3.60,  8.00),
    "Glazing":      (0.10,  0.40),
    "IntRate":      (0.25,  1.50),   # was (0.75, 5.00) — corrected to dataset range
    "Infiltration": (0.50,  1.50),
    "Electax":      (0.00, 10.00),   # was (0, 4) — corrected to dataset range
    "SHGC":         (0.10,  0.70),
    "Fueltax":      (0.00, 10.00),   # was (0, 8) — corrected to dataset range
    "A_PV":         (0.10,  0.60),
}

BUILDING_VARS = {
    "Rvalue_roof":  {"label": "Roof R-value",             "unit": "m²K/W", "symbol": "V_roof",  "icon": "🏠"},
    "Rvalue_wall":  {"label": "Wall R-value",             "unit": "m²K/W", "symbol": "V_wall",  "icon": "🧱"},
    "Glazing":      {"label": "Glazing ratio",            "unit": "—",     "symbol": "G",       "icon": "🪟"},
    "SHGC":         {"label": "Solar Heat Gain Coeff.",   "unit": "—",     "symbol": "SHGC",    "icon": "🌤️"},
    "Infiltration": {"label": "Infiltration rate",        "unit": "ACH",   "symbol": "ṁ_inf",   "icon": "💨"},
    "Albedo_roof":  {"label": "Roof albedo",              "unit": "—",     "symbol": "α",       "icon": "☀️"},
    "A_PV":         {"label": "PV area ratio",            "unit": "—",     "symbol": "A_PV",    "icon": "⚡"},
    "A_ST":         {"label": "Solar thermal area",       "unit": "—",     "symbol": "A_ST",    "icon": "🌡️"},
    "V_bites":      {"label": "BITES system",             "unit": "—",     "symbol": "V_BITES", "icon": "🧊"},
}

ECONOMIC_VARS = {
    "Loan":         {"label": "Loan amount",              "unit": "$",     "symbol": "L",       "icon": "🏦"},
    "Rebate":       {"label": "Rebate amount",            "unit": "$",     "symbol": "R",       "icon": "💰"},
    "IntRate":      {"label": "Interest rate",            "unit": "%",     "symbol": "i",       "icon": "📈"},
    "Electax":      {"label": "Electricity tax",          "unit": "¢/kWh", "symbol": "τ_e",     "icon": "⚡"},
    "Fueltax":      {"label": "Fuel tax",                 "unit": "$/GJ",  "symbol": "τ_f",     "icon": "⛽"},
}

ALL_VARS = {**BUILDING_VARS, **ECONOMIC_VARS}

# ─────────────────────────────────────────
# ── CHANGED: Load one model per city ─────
# Place all city_models/ files next to app.py
# ─────────────────────────────────────────
MODEL_DIR = "city_models"  # folder containing <City>_model.pkl and <City>_columns.pkl

@st.cache_resource
def load_city_models():
    """Load all per-city RBF bundles: {city: {rbf, scaler, columns}}."""
    models = {}
    missing = []
    for city in CITIES:
        try:
            bundle = joblib.load(f"{MODEL_DIR}/{city}_rbf.pkl")
            models[city] = bundle   # keys: rbf, scaler, columns
        except FileNotFoundError:
            missing.append(city)
    return models, missing

city_models, missing_cities = load_city_models()
model_ok = len(city_models) > 0

if "selected_city" not in st.session_state:
    st.session_state.selected_city = None

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown("## 🏗️ AI Retrofit Decision Tool")
st.caption("Click a city on the map, configure parameters, and run the optimizer.")

if not model_ok:
    st.error(
        f"⚠️ No model files found — place the `city_models/` folder next to app.py. "
        f"Run `train_city_models.py` to generate them."
    )
elif missing_cities:
    st.warning(f"⚠️ Missing models for: {', '.join(missing_cities)}")

st.markdown("---")

# ─────────────────────────────────────────
# MAP | CONTROLS
# ─────────────────────────────────────────
map_col, ctrl_col = st.columns([1.6, 1], gap="large")

with map_col:
    st.markdown('<div class="section-label">Select city — click a marker</div>',
                unsafe_allow_html=True)
    selected = st.session_state.selected_city

    m = folium.Map(location=[56, -96], zoom_start=3.5,
                   tiles="CartoDB positron",
                   zoom_control=True, scrollWheelZoom=True, dragging=True)

    for city_name, (lat, lon) in CITIES.items():
        is_sel = (city_name == selected)
        has_model = city_name in city_models
        if is_sel:
            folium.CircleMarker(location=[lat, lon], radius=20,
                                color="#0a0a0a", fill=True,
                                fill_color="#c9d4e0", fill_opacity=0.5,
                                weight=2).add_to(m)
        folium.Marker(
            location=[lat, lon],
            tooltip=city_name,
            popup=folium.Popup(city_name, max_width=120),
            icon=folium.DivIcon(
                html=f"""<div style="
                    background:{'#0a0a0a' if is_sel else ('#ffffff' if has_model else '#fee2e2')};
                    color:{'#ffffff' if is_sel else '#0a0a0a'};
                    border:2px solid {'#0a0a0a' if is_sel else ('#64748b' if has_model else '#b91c1c')};
                    border-radius:50%;width:32px;height:32px;
                    display:flex;align-items:center;justify-content:center;
                    font-size:11px;font-weight:800;font-family:sans-serif;
                    box-shadow:0 2px 6px rgba(0,0,0,0.18);cursor:pointer;
                ">{city_name[:2]}</div>""",
                icon_size=(32, 32), icon_anchor=(16, 16),
            ),
        ).add_to(m)

    map_data = st_folium(m, height=420, use_container_width=True,
                         returned_objects=["last_object_clicked_popup"])
    clicked = (map_data or {}).get("last_object_clicked_popup")
    if clicked and clicked in CITIES:
        st.session_state.selected_city = clicked
        st.rerun()

    if selected:
        lat, lon = CITIES[selected]
        # ── CHANGED: show per-city model R² in the city banner ───────────────
        r2_badge = ""
        if selected in city_models:
            import json, os
            stats_path = f"{MODEL_DIR}/city_stats.json"
            if os.path.exists(stats_path):
                with open(stats_path) as f:
                    city_stats = json.load(f)
                r2 = city_stats.get(selected, {}).get("model_r2", None)
                if r2 is not None:
                    r2_badge = f'&nbsp;<span style="font-size:12px;font-weight:600;color:#9aabb8;">Model R²={r2:.2f}</span>'
        st.markdown(
            f'<div class="city-selected">📍 {selected} &nbsp;&nbsp;'
            f'<span style="font-size:13px;font-weight:600;color:#c9d4e0;">'
            f'{lat:.2f}°N, {abs(lon):.2f}°W</span>{r2_badge}</div>',
            unsafe_allow_html=True)
    else:
        st.markdown('<div class="city-prompt">👆 Click a city marker on the map</div>',
                    unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:10px;">Or click a name</div>',
                unsafe_allow_html=True)
    btn_cols = st.columns(5)
    for i, city_name in enumerate(CITIES):
        with btn_cols[i % 5]:
            if st.button(city_name, key=f"btn_{city_name}",
                         type="primary" if city_name == selected else "secondary",
                         use_container_width=True):
                st.session_state.selected_city = city_name
                st.rerun()

with ctrl_col:
    st.markdown('<div class="section-label">Building parameters</div>', unsafe_allow_html=True)
    footprint = st.number_input("Building footprint (m²)", value=130, min_value=30, max_value=2000, step=10)
    ssp       = st.selectbox("SSP Scenario", ["SSP126","SSP245","SSP585"], index=1)

    st.markdown('<div class="section-label" style="margin-top:12px;">Inflation rates</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    elec_inf = c1.number_input("Electricity", value=0.01, min_value=0.0, max_value=0.5, step=0.005, format="%.3f")
    fuel_inf = c2.number_input("Fuel",        value=0.05, min_value=0.0, max_value=0.5, step=0.005, format="%.3f")

    st.markdown('<div class="section-label" style="margin-top:12px;">Objective weights</div>', unsafe_allow_html=True)
    w_owner = st.slider("🏠 Owner savings", 0.0, 1.0, 0.60, 0.05)
    w_gov   = st.slider("🏛️ Gov savings",   0.0, 1.0, 0.20, 0.05)
    w_ghg   = st.slider("🌿 GHG reduction", 0.0, 1.0, 0.20, 0.05)

    wsum = round(w_owner + w_gov + w_ghg, 2)
    if abs(wsum - 1.0) > 0.01:
        st.markdown(f'<p class="warn">⚠️ Weights sum to {wsum:.2f} — must equal 1.0</p>',
                    unsafe_allow_html=True)
        weights_ok = False
    else:
        st.success(f"Weights ✓ ({wsum:.2f})")
        weights_ok = True

    # ── CHANGED: check city has a model loaded ────────────────────────────────
    city_has_model = selected in city_models
    ready = selected and weights_ok and model_ok and city_has_model
    run = st.button(
        f"▶ Find best retrofit{' for ' + selected if selected else ''}",
        type="primary", use_container_width=True, disabled=not ready,
    )
    if not selected:
        st.caption("← Select a city on the map first")
    elif not city_has_model:
        st.caption(f"⚠️ No model file found for {selected}")

# ─────────────────────────────────────────
# OPTIMIZATION
# ─────────────────────────────────────────
if run and selected:
    city = selected

    # ── Use real dataset rows for this city — exact simulation values, no prediction error ──
    OUTPUT_COLS_ALL = [
        "TotalOperationalCO2Save_kgCO2", "TotalEmbodiedCO2_kgCO2",
        "CostAnnualSysSave_CAD", "TotalCO2Sav", "AnnSCCSav_CAD",
        "BaseCostAnnual_CAD", "PercentCostSysSav_percent", "AnnGovtCostSav_CAD"
    ]

    with st.spinner(f"Filtering dataset for {city} / {ssp}…"):
        df = DATASET[
            (DATASET["City"] == city) &
            (DATASET["SSP"]  == ssp)
        ].copy().reset_index(drop=True)

        if df.empty:
            st.error(f"No data rows found for {city} + {ssp}. Try a different SSP scenario.")
            st.stop()

        df["GHG"]   = df["TotalCO2Sav"]
        df["Owner"] = df["CostAnnualSysSave_CAD"]
        df["Gov"]   = df["AnnGovtCostSav_CAD"]

        def norm(s): return (s - s.min()) / (s.max() - s.min() + 1e-9)
        df["GHG_n"]   = 1 - norm(df["GHG"])   # lower CO2 = better
        df["Owner_n"] = norm(df["Owner"])
        df["Gov_n"]   = norm(df["Gov"])
        df["Score"]   = w_owner*df["Owner_n"] + w_gov*df["Gov_n"] + w_ghg*df["GHG_n"]

        df_sorted = df.sort_values("Score", ascending=False).reset_index(drop=True)
        best = df_sorted.iloc[0]

    # ── Metric cards ──────────────────────────────
    st.markdown("---")
    st.success(f"✅ Best retrofit found for **{city}** under **{ssp}** from {len(df):,} real simulation scenarios")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Composite score", f"{best['Score']:.4f}")
    m2.metric("GHG reduction",   f"{best['GHG']:.1f} tCO₂e/yr")
    m3.metric("Owner savings",   f"${best['Owner']:,.0f}")
    m4.metric("Gov savings",     f"${best['Gov']:,.0f}")

    st.markdown("---")

    # ─────────────────────────────────────────────
    # VARIABLE CARD RENDERER
    # ─────────────────────────────────────────────
    def var_card(k, meta, best_val, is_economic=False):
        lo_v, hi_v = RANGES[k]
        pct = round((best_val - lo_v) / (hi_v - lo_v + 1e-9) * 100)
        pct = max(0, min(100, pct))

        if k in ("Loan", "Rebate") or meta["unit"] == "$":
            val_str = f"${best_val:,.0f}"
        elif meta["unit"] == "%":
            val_str = f"{best_val:.2f}%"
        else:
            val_str = f"{best_val:.3f} {meta['unit'] if meta['unit'] != '—' else ''}"

        range_str = f"{lo_v} – {hi_v}"
        if meta["unit"] not in ("—", "$", "%"):
            range_str += f" {meta['unit']}"

        icon_class   = "var-icon-economic" if is_economic else "var-icon-building"
        symbol_class = "var-symbol var-symbol-econ" if is_economic else "var-symbol"
        bar_color    = "#78350f" if is_economic else "#1a1a2e"

        return f"""
        <div class="var-card">
          <div class="var-icon {icon_class}">{meta['icon']}</div>
          <div class="var-info">
            <div class="var-label">
              <span class="{symbol_class}">{meta['symbol']}</span>{meta['label']}
            </div>
            <div class="var-value">{val_str}</div>
            <div class="var-range">Range: {range_str}</div>
          </div>
          <div class="bar-wrap">
            <div class="bar-track">
              <div class="bar-fill" style="width:{pct}%;background:{bar_color};"></div>
            </div>
            <div class="bar-pct">{pct}%</div>
          </div>
        </div>"""

    # ── Two result columns ─────────────────────────
    col_build, col_econ = st.columns(2, gap="large")

    with col_build:
        st.markdown('<div class="result-group-title">🏗️ Building features</div>',
                    unsafe_allow_html=True)
        html = "".join(
            var_card(k, meta, float(best[k]), is_economic=False)
            for k, meta in BUILDING_VARS.items()
        )
        st.markdown(html, unsafe_allow_html=True)

    with col_econ:
        st.markdown('<div class="result-group-title">💰 Economic parameters</div>',
                    unsafe_allow_html=True)
        html = "".join(
            var_card(k, meta, float(best[k]), is_economic=True)
            for k, meta in ECONOMIC_VARS.items()
        )
        st.markdown(html, unsafe_allow_html=True)

    st.markdown("---")

    # ── Charts ────────────────────────────────────
    ch1, ch2, ch3 = st.columns(3, gap="small")

    with ch1:
        fig_hist = px.histogram(df, x="Score", nbins=40,
                                color_discrete_sequence=["#1a1a2e"],
                                title="Score distribution")
        fig_hist.add_vline(x=best["Score"], line_dash="dash",
                           line_color="#b91c1c",
                           annotation_text="Best",
                           annotation_font_size=12,
                           annotation_font_color="#b91c1c")
        fig_hist.update_layout(
            margin=dict(t=40,b=10,l=10,r=10), height=230,
            showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
            font=dict(color="#0a0a0a", size=12),
            title=dict(font=dict(size=14, color="#0a0a0a")),
            xaxis=dict(showgrid=False, title="", tickfont=dict(size=11, color="#0a0a0a")),
            yaxis=dict(showgrid=False, title="", tickfont=dict(size=11, color="#0a0a0a")),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with ch2:
        fig_radar = go.Figure(go.Scatterpolar(
            r=[best["Owner_n"], best["Gov_n"], best["GHG_n"], best["Owner_n"]],
            theta=["Owner savings","Gov savings","GHG reduction","Owner savings"],
            fill="toself",
            fillcolor="rgba(26,26,46,0.12)",
            line=dict(color="#1a1a2e", width=2.5),
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(range=[0,1], showticklabels=False,
                                gridcolor="#c9d4e0", linecolor="#c9d4e0"),
                angularaxis=dict(tickfont=dict(size=12, color="#0a0a0a"),
                                 gridcolor="#c9d4e0"),
            ),
            margin=dict(t=40,b=20,l=50,r=50), height=230,
            paper_bgcolor="white",
            font=dict(color="#0a0a0a"),
            title=dict(text="Objective profile", font=dict(size=14, color="#0a0a0a")),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with ch3:
        sample_plot = df.sample(min(400, len(df)), random_state=1)
        fig_par = px.scatter(
            sample_plot, x="Owner", y="GHG",
            color="Score", color_continuous_scale=["#e2e8f0","#1a1a2e"],
            opacity=0.6, title="Pareto space",
            labels={"Owner":"Owner savings ($)","GHG":"GHG (tCO₂e/yr)"},
        )
        fig_par.add_scatter(
            x=[best["Owner"]], y=[best["GHG"]], mode="markers",
            marker=dict(size=14, color="#b91c1c", symbol="star"),
            name="Best", showlegend=False,
        )
        fig_par.update_layout(
            margin=dict(t=40,b=10,l=10,r=10), height=230,
            plot_bgcolor="white", paper_bgcolor="white",
            coloraxis_showscale=False,
            font=dict(color="#0a0a0a", size=12),
            title=dict(font=dict(size=14, color="#0a0a0a")),
            xaxis=dict(showgrid=False, tickfont=dict(size=11, color="#0a0a0a")),
            yaxis=dict(showgrid=False, tickfont=dict(size=11, color="#0a0a0a")),
        )
        st.plotly_chart(fig_par, use_container_width=True)

    # ── Sensitivity ───────────────────────────────
    with st.expander("📊 Sensitivity — which variables drive the score?"):

        # Compute correlation for every parameter
        corrs = {}
        for k in list(RANGES.keys()):
            xs, ys = df[k].values, df["Score"].values
            mx, my = xs.mean(), ys.mean()
            num = ((xs - mx) * (ys - my)).sum()
            den = np.sqrt(((xs - mx) ** 2).sum() * ((ys - my) ** 2).sum()) + 1e-9
            corrs[k] = abs(num / den)

        # Split into two ordered dicts — building then economic
        def make_chart(var_dict, color, title):
            subset = {k: corrs[k] for k in var_dict if k in corrs}
            subset = dict(sorted(subset.items(), key=lambda x: x[1]))  # ascending → bars left-to-right
            y_labels = [
                f"{var_dict[k]['icon']}  {var_dict[k]['symbol']}  —  {var_dict[k]['label']}"
                for k in subset
            ]
            fig = go.Figure(go.Bar(
                x=list(subset.values()),
                y=y_labels,
                orientation="h",
                marker_color=color,
                marker_line_width=0,
            ))
            fig.update_layout(
                title=dict(text=title, font=dict(size=14, color="#0a0a0a")),
                margin=dict(t=40, b=10, l=10, r=20),
                height=320,
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(color="#0a0a0a", size=12),
                xaxis=dict(
                    showgrid=False,
                    title="|Correlation with score|",
                    title_font=dict(size=12, color="#0a0a0a"),
                    tickfont=dict(size=11, color="#0a0a0a"),
                ),
                yaxis=dict(showgrid=False, tickfont=dict(size=12, color="#0a0a0a")),
            )
            return fig

        sens_col1, sens_col2 = st.columns(2, gap="large")

        with sens_col1:
            st.markdown('<div class="result-group-title">🏗️ Building parameters</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(
                make_chart(BUILDING_VARS, "#1a1a2e", ""),
                use_container_width=True,
            )

        with sens_col2:
            st.markdown('<div class="result-group-title">💰 Economic parameters</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(
                make_chart(ECONOMIC_VARS, "#78350f", ""),
                use_container_width=True,
            )

    # ── Top 10 ────────────────────────────────────
    with st.expander("🏆 Top 10 candidates"):
        top10 = df_sorted.head(10)[["Score","GHG","Owner","Gov"]].copy()
        top10.index = range(1, 11)
        top10.columns = ["Score","GHG (tCO₂e/yr)","Owner ($)","Gov ($)"]
        st.dataframe(
            top10.style.format({"Score":"{:.4f}","GHG (tCO₂e/yr)":"{:.2f}",
                                "Owner ($)":"${:,.0f}","Gov ($)":"${:,.0f}"}),
            use_container_width=True,
        )

    # ── Download ──────────────────────────────────
    csv = df_sorted.head(50).to_csv(index=False).encode()
    st.download_button("⬇️ Download top 50 results (CSV)", data=csv,
                       file_name=f"retrofit_{city}_{ssp}.csv", mime="text/csv")