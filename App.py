import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import qmc

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
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px 16px;
    }
    .city-selected {
        background: #eff6ff;
        border: 2px solid #3b82f6;
        border-radius: 10px;
        padding: 10px 16px;
        font-size: 18px;
        font-weight: 700;
        color: #1d4ed8;
        text-align: center;
        margin-bottom: 8px;
    }
    .city-prompt {
        background: #f8fafc;
        border: 1px dashed #cbd5e1;
        border-radius: 10px;
        padding: 10px 16px;
        font-size: 13px;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 8px;
    }
    .section-label {
        font-size: 10px;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }
    .warn { color: #dc2626; font-size: 12px; }

    /* ── Result cards ── */
    .result-group-title {
        font-size: 13px;
        font-weight: 700;
        color: #1e293b;
        margin: 18px 0 10px;
        padding-bottom: 6px;
        border-bottom: 2px solid #e2e8f0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .var-card {
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 12px 14px;
        background: #ffffff;
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 8px;
        transition: box-shadow 0.15s;
    }
    .var-card:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.07); }
    .var-icon {
        font-size: 22px;
        width: 36px;
        height: 36px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 8px;
        flex-shrink: 0;
    }
    .var-icon-building { background: #eff6ff; }
    .var-icon-economic { background: #fefce8; }
    .var-info { flex: 1; min-width: 0; }
    .var-label { font-size: 12px; color: #64748b; margin-bottom: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .var-value { font-size: 16px; font-weight: 700; color: #1e293b; }
    .var-range { font-size: 10px; color: #94a3b8; margin-top: 2px; }
    .bar-wrap  { width: 80px; flex-shrink: 0; }
    .bar-track { height: 5px; background: #f1f5f9; border-radius: 99px; overflow: hidden; }
    .bar-fill  { height: 5px; border-radius: 99px; }
    .bar-pct   { font-size: 10px; color: #94a3b8; text-align: right; margin-top: 2px; }
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

RANGES = {
    "V_bites":      (0.05, 0.25),
    "Albedo_roof":  (0.10, 0.70),
    "A_ST":         (0.10, 0.60),
    "Rvalue_roof":  (5.46, 11.0),
    "Loan":         (0,    50000),
    "Rebate":       (0,    10000),
    "Rvalue_wall":  (3.60, 8.00),
    "Glazing":      (0.20, 0.40),
    "IntRate":      (0.75, 5.00),
    "Infiltration": (0.50, 1.50),
    "Electax":      (0.00, 4.00),
    "SHGC":         (0.10, 0.70),
    "Fueltax":      (0.00, 8.00),
    "A_PV":         (0.10, 0.60),
}

# ── Building feature variables ──────────────────────
BUILDING_VARS = {
    "Rvalue_roof":  {"label": "Roof R-value",            "unit": "m²K/W",  "symbol": "V_roof",   "icon": "🏠"},
    "Rvalue_wall":  {"label": "Wall R-value",            "unit": "m²K/W",  "symbol": "V_wall",   "icon": "🧱"},
    "Glazing":      {"label": "Glazing ratio",           "unit": "—",      "symbol": "G",        "icon": "🪟"},
    "SHGC":         {"label": "Solar Heat Gain Coeff.",  "unit": "—",      "symbol": "SHGC",     "icon": "🌤️"},
    "Infiltration": {"label": "Infiltration rate",       "unit": "ACH",    "symbol": "ṁ_inf",    "icon": "💨"},
    "Albedo_roof":  {"label": "Roof albedo",             "unit": "—",      "symbol": "α",        "icon": "☀️"},
    "A_PV":         {"label": "PV area ratio",           "unit": "—",      "symbol": "A_PV",     "icon": "⚡"},
    "A_ST":         {"label": "Solar thermal area",      "unit": "—",      "symbol": "A_ST",     "icon": "🌡️"},
    "V_bites":      {"label": "BITES system",            "unit": "—",      "symbol": "V_BITES",  "icon": "🧊"},
}

# ── Economic / policy variables ─────────────────────
ECONOMIC_VARS = {
    "Loan":         {"label": "Loan amount",             "unit": "$",      "symbol": "L",        "icon": "🏦"},
    "Rebate":       {"label": "Rebate amount",           "unit": "$",      "symbol": "R",        "icon": "💰"},
    "IntRate":      {"label": "Interest rate",           "unit": "%",      "symbol": "i",        "icon": "📈"},
    "Electax":      {"label": "Electricity tax",         "unit": "¢/kWh",  "symbol": "τ_e",      "icon": "⚡"},
    "Fueltax":      {"label": "Fuel tax",                "unit": "$/GJ",   "symbol": "τ_f",      "icon": "⛽"},
}

ALL_VARS = {**BUILDING_VARS, **ECONOMIC_VARS}

# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    m = joblib.load("retrofit_forward_model.pkl")
    c = joblib.load("model_columns.pkl")
    return m, c

try:
    model, model_columns = load_model()
    model_ok = True
except FileNotFoundError:
    model_ok = False

# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
if "selected_city" not in st.session_state:
    st.session_state.selected_city = None

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown("## 🏗️ AI Retrofit Decision Tool")
st.caption("Click a city on the map, configure parameters, and run the optimizer.")
if not model_ok:
    st.error("⚠️ Model files not found — place `retrofit_forward_model.pkl` and `model_columns.pkl` next to app.py.")
st.markdown("---")

# ─────────────────────────────────────────
# LAYOUT: MAP | CONTROLS
# ─────────────────────────────────────────
map_col, ctrl_col = st.columns([1.6, 1], gap="large")

# ══════════════════════════════════════════
# MAP
# ══════════════════════════════════════════
with map_col:
    st.markdown('<div class="section-label">Select city — click a marker</div>',
                unsafe_allow_html=True)
    selected = st.session_state.selected_city

    m = folium.Map(location=[56, -96], zoom_start=3.5,
                   tiles="CartoDB positron",
                   zoom_control=True, scrollWheelZoom=True, dragging=True)

    for city_name, (lat, lon) in CITIES.items():
        is_sel = (city_name == selected)
        if is_sel:
            folium.CircleMarker(location=[lat, lon], radius=18,
                                color="#1d4ed8", fill=True,
                                fill_color="#bfdbfe", fill_opacity=0.4,
                                weight=2).add_to(m)
        folium.Marker(
            location=[lat, lon],
            tooltip=city_name,
            popup=folium.Popup(city_name, max_width=120),
            icon=folium.DivIcon(
                html=f"""<div style="
                    background:{'#1d4ed8' if is_sel else '#ffffff'};
                    color:{'#ffffff' if is_sel else '#1e293b'};
                    border:2px solid {'#1d4ed8' if is_sel else '#94a3b8'};
                    border-radius:50%;width:30px;height:30px;
                    display:flex;align-items:center;justify-content:center;
                    font-size:11px;font-weight:700;font-family:sans-serif;
                    box-shadow:0 2px 6px rgba(0,0,0,0.15);cursor:pointer;
                ">{city_name[:2]}</div>""",
                icon_size=(30, 30), icon_anchor=(15, 15),
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
        st.markdown(
            f'<div class="city-selected">📍 {selected} &nbsp;'
            f'<span style="font-size:13px;font-weight:400;color:#3b82f6;">'
            f'{lat:.2f}°N, {abs(lon):.2f}°W</span></div>',
            unsafe_allow_html=True)
    else:
        st.markdown('<div class="city-prompt">👆 Click a city marker on the map</div>',
                    unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:8px;">Or click a name</div>',
                unsafe_allow_html=True)
    btn_cols = st.columns(5)
    for i, city_name in enumerate(CITIES):
        with btn_cols[i % 5]:
            if st.button(city_name, key=f"btn_{city_name}",
                         type="primary" if city_name == selected else "secondary",
                         use_container_width=True):
                st.session_state.selected_city = city_name
                st.rerun()

# ══════════════════════════════════════════
# CONTROLS
# ══════════════════════════════════════════
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

    st.markdown('<div class="section-label" style="margin-top:12px;">Sampling</div>', unsafe_allow_html=True)
    n_samples = st.select_slider("Candidates", [500,1000,2000,5000], value=2000)
    use_lhs   = st.toggle("Latin Hypercube Sampling", value=True,
                          help="Better coverage than pure random with the same sample count")

    ready = selected and weights_ok and model_ok
    run = st.button(
        f"▶ Find best retrofit{' for ' + selected if selected else ''}",
        type="primary", use_container_width=True, disabled=not ready,
    )
    if not selected:
        st.caption("← Select a city on the map first")

# ─────────────────────────────────────────
# OPTIMIZATION
# ─────────────────────────────────────────
if run and selected:
    city = selected
    keys = list(RANGES.keys())
    lo   = np.array([RANGES[k][0] for k in keys])
    hi   = np.array([RANGES[k][1] for k in keys])

    with st.spinner(f"Sampling {n_samples:,} candidates for {city}…"):
        if use_lhs:
            sampler = qmc.LatinHypercube(d=len(keys), seed=42)
            samples = qmc.scale(sampler.random(n=n_samples), lo, hi)
        else:
            rng     = np.random.default_rng(42)
            samples = rng.uniform(lo, hi, size=(n_samples, len(keys)))
        df = pd.DataFrame(samples, columns=keys)
        df["City"] = city; df["SSP"] = ssp
        df["BuildingFootprintArea_m2"] = footprint
        df["ElectricityInflationRate"] = elec_inf
        df["FuelInflationRate"]        = fuel_inf

    with st.spinner("Running AI model…"):
        X    = pd.get_dummies(df, columns=["City","SSP"])
        X    = X.reindex(columns=model_columns, fill_value=0)
        pred = model.predict(X)
        df["GHG"]   = pred[:, 3]
        df["Owner"] = pred[:, 2]
        df["Gov"]   = pred[:, 7]

        def norm(s): return (s - s.min()) / (s.max() - s.min() + 1e-9)
        df["GHG_n"]   = 1 - norm(df["GHG"])
        df["Owner_n"] = norm(df["Owner"])
        df["Gov_n"]   = norm(df["Gov"])
        df["Score"]   = w_owner*df["Owner_n"] + w_gov*df["Gov_n"] + w_ghg*df["GHG_n"]

        df_sorted = df.sort_values("Score", ascending=False).reset_index(drop=True)
        best = df_sorted.iloc[0]

    # ── Top metrics ───────────────────────────────
    st.markdown("---")
    st.success(f"✅ Best retrofit found for **{city}** under **{ssp}** from {n_samples:,} candidates")

    m1, m2, m3, m4 = st.columns(4)
    #m1.metric("Composite score", f"{best['Score']:.4f}")
    m2.metric("GHG reduction",   f"{best['GHG']:.1f} tCO₂e/yr")
    m3.metric("Owner savings",   f"${best['Owner']:,.0f}")
    m4.metric("Gov savings",     f"${best['Gov']:,.0f}")

    net_cost = max(float(best["Loan"]) - float(best["Rebate"]), 0)
    annual   = max(float(best["Owner"]) / 25, 1)
    #payback  = round(net_cost / annual)
    #st.caption(f"⏱️ Estimated payback: **{payback if payback > 0 else '< 1'} years** "
               #f"(net cost ${net_cost:,.0f} ÷ ~${annual:,.0f}/yr)")

    st.markdown("---")

    # ─────────────────────────────────────────────
    # RESULTS: TWO GROUPS SIDE BY SIDE
    # ─────────────────────────────────────────────
    def var_card_html(k, meta, best_val, lo, hi, card_class, bar_color):
        pct  = round((best_val - lo) / (hi - lo + 1e-9) * 100)
        # Format value
        if k in ("Loan", "Rebate"):
            val_str = f"${best_val:,.0f}"
        elif meta["unit"] == "%":
            val_str = f"{best_val:.2f} %"
        elif meta["unit"] == "$":
            val_str = f"${best_val:,.0f}"
        else:
            val_str = f"{best_val:.3f}"
        range_str = f"Range: {lo} – {hi} {meta['unit']}"

        return f"""
        <div class="var-card">
          <div class="var-icon {card_class}">{meta['icon']}</div>
          <div class="var-info">
            <div class="var-label">
              <code style="font-size:10px;background:#f1f5f9;padding:1px 4px;border-radius:4px;color:#6366f1;">{meta['symbol']}</code>
              &nbsp;{meta['label']}
            </div>
            <div class="var-value">{val_str}</div>
            <div class="var-range">{range_str}</div>
          </div>
          <div class="bar-wrap">
            <div class="bar-track">
              <div class="bar-fill" style="width:{pct}%;background:{bar_color};"></div>
            </div>
            <div class="bar-pct">{pct}%</div>
          </div>
        </div>"""

    col_build, col_econ = st.columns(2, gap="large")

    # ── Building features ─────────────────────────
    with col_build:
        st.markdown(
            '<div class="result-group-title">🏗️ Building features</div>',
            unsafe_allow_html=True)
        cards_html = ""
        for k, meta in BUILDING_VARS.items():
            lo_v, hi_v = RANGES[k]
            cards_html += var_card_html(k, meta, float(best[k]),
                                        lo_v, hi_v,
                                        "var-icon-building", "#3266ad")
        st.markdown(cards_html, unsafe_allow_html=True)

    # ── Economic parameters ───────────────────────
    with col_econ:
        st.markdown(
            '<div class="result-group-title">💰 Economic parameters</div>',
            unsafe_allow_html=True)
        cards_html = ""
        for k, meta in ECONOMIC_VARS.items():
            lo_v, hi_v = RANGES[k]
            cards_html += var_card_html(k, meta, float(best[k]),
                                        lo_v, hi_v,
                                        "var-icon-economic", "#d97706")
        st.markdown(cards_html, unsafe_allow_html=True)
        '''
        # Payback card inside economic panel
        st.markdown(f"""
        <div class="var-card" style="border-color:#bfdbfe;background:#eff6ff;">
          <div class="var-icon" style="background:#dbeafe;font-size:22px;">⏱️</div>
          <div class="var-info">
            <div class="var-label"><code style="font-size:10px;background:#dbeafe;padding:1px 4px;border-radius:4px;color:#6366f1;">PB</code>&nbsp;Payback period</div>
            <div class="var-value" style="color:#1d4ed8;">{payback if payback > 0 else '< 1'} years</div>
            <div class="var-range">Net cost ${net_cost:,.0f} ÷ ~${annual:,.0f}/yr</div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    '''
    # ─────────────────────────────────────────────
    # CHARTS ROW
    # ─────────────────────────────────────────────
    ch1, ch2, ch3 = st.columns(3, gap="small")

    with ch1:
        fig_hist = px.histogram(df, x="Score", nbins=40,
                                color_discrete_sequence=["#3266ad"],
                                title="Score distribution")
        fig_hist.add_vline(x=best["Score"], line_dash="dash",
                           line_color="#D85A30",
                           annotation_text="Best", annotation_font_size=10)
        fig_hist.update_layout(margin=dict(t=35,b=10,l=10,r=10), height=220,
                               showlegend=False,
                               plot_bgcolor="white", paper_bgcolor="white",
                               xaxis=dict(showgrid=False, title=""),
                               yaxis=dict(showgrid=False, title=""))
        st.plotly_chart(fig_hist, use_container_width=True)

    with ch2:
        fig_radar = go.Figure(go.Scatterpolar(
            r=[best["Owner_n"], best["Gov_n"], best["GHG_n"], best["Owner_n"]],
            theta=["Owner savings","Gov savings","GHG reduction","Owner savings"],
            fill="toself",
            fillcolor="rgba(50,102,173,0.15)",
            line=dict(color="#3266ad", width=2),
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(range=[0,1], showticklabels=False,
                                       gridcolor="#f3f4f6")),
            margin=dict(t=30,b=20,l=40,r=40), height=220,
            paper_bgcolor="white",
            title=dict(text="Objective profile", font_size=12),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with ch3:
        sample_plot = df.sample(min(400, len(df)), random_state=1)
        fig_par = px.scatter(sample_plot, x="Owner", y="GHG",
                             color="Score", color_continuous_scale="Blues",
                             opacity=0.5, title="Pareto space",
                             labels={"Owner":"Owner ($)","GHG":"GHG (tCO₂e/yr)"})
        fig_par.add_scatter(
            x=[best["Owner"]], y=[best["GHG"]], mode="markers",
            marker=dict(size=14, color="#D85A30", symbol="star"),
            name="Best", showlegend=False,
        )
        fig_par.update_layout(margin=dict(t=35,b=10,l=10,r=10), height=220,
                              plot_bgcolor="white", paper_bgcolor="white",
                              coloraxis_showscale=False)
        st.plotly_chart(fig_par, use_container_width=True)

    # ── Sensitivity ───────────────────────────────
    with st.expander("📊 Sensitivity — which variables drive the score?"):
        corrs = {}
        for k in list(RANGES.keys()):
            xs, ys = df[k].values, df["Score"].values
            mx, my = xs.mean(), ys.mean()
            num = ((xs-mx)*(ys-my)).sum()
            den = np.sqrt(((xs-mx)**2).sum()*((ys-my)**2).sum())+1e-9
            corrs[k] = abs(num/den)
        corr_sorted = dict(sorted(corrs.items(), key=lambda x: x[1], reverse=True))

        labels = []
        for k in corr_sorted:
            if k in ALL_VARS:
                meta = ALL_VARS[k]
                labels.append(f"{meta['icon']} {meta['symbol']} — {meta['label']}")
            else:
                labels.append(k)

        fig_sens = go.Figure(go.Bar(
            x=list(corr_sorted.values()),
            y=labels,
            orientation="h",
            marker_color=["#3266ad" if k in BUILDING_VARS else "#d97706"
                          for k in corr_sorted],
            marker_line_width=0,
        ))
        fig_sens.update_layout(
            margin=dict(t=10,b=10,l=10,r=10), height=440,
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(showgrid=False, title="|correlation with score|"),
            yaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig_sens, use_container_width=True)
        st.caption("🔵 Blue = building feature &nbsp;&nbsp; 🟡 Amber = economic parameter")

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