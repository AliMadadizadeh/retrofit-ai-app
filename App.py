import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
import os
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px

# ── Resolve paths relative to this script — works locally and on Streamlit Cloud
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "retrofit_dataset_final_solution1.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "city_models")

# ── Load the real dataset once ────────────────────────────────────────────────
@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)

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
    "Rvalue_roof":  {"label": "Roof R-value",             "unit": r"m^2 \cdot K \cdot W^{-1}", "symbol": r"R_{\text{roof}}",   "icon": "🏠"},
    "Rvalue_wall":  {"label": "Wall R-value",             "unit": r"m^2 \cdot K \cdot W^{-1}", "symbol": r"R_{\text{wall}}",   "icon": "🧱"},
    "Glazing":      {"label": "Glazing ratio",            "unit": r"—",                         "symbol": r"GR",                "icon": "🪟"},
    "SHGC":         {"label": "Solar Heat Gain Coeff.",   "unit": r"—",                         "symbol": r"SHGC",              "icon": "🌤️"},
    "Infiltration": {"label": "Infiltration rate",        "unit": r"ACH",                    "symbol": r"\dot{m}_{\inf}",    "icon": "💨"},
    "Albedo_roof":  {"label": "Roof albedo",              "unit": r"—",                         "symbol": r"\alpha",            "icon": "☀️"},
    "A_PV":         {"label": "PV area ratio",            "unit": r"—",                         "symbol": r"A_{\text{PV}}",     "icon": "⚡"},
    "A_ST":         {"label": "Solar thermal area",       "unit": r"—",                         "symbol": r"A_{\text{ST}}",     "icon": "🌡️"},
    "V_bites":      {"label": "BITES system",             "unit": r"—",                         "symbol": r"V_{\text{BITES}}",  "icon": "🧊"},
}

ECONOMIC_VARS = {
    "Loan":         {"label": "Loan amount",     "unit": "$",  "symbol": r"L",          "icon": "🏦"},
    "Rebate":       {"label": "Rebate amount",   "unit": "$",  "symbol": r"R",          "icon": "💰"},
    "IntRate":      {"label": "Interest rate",   "unit": r"\%","symbol": r"i",          "icon": "📈"},
    "Electax":      {"label": "Electricity tax", "unit": r"\%","symbol": r"\tau_{e}",   "icon": "⚡"},
    "Fueltax":      {"label": "Fuel tax",        "unit": r"\%","symbol": r"\tau_{f}",   "icon": "⛽"},
}

ALL_VARS = {**BUILDING_VARS, **ECONOMIC_VARS}

# ─────────────────────────────────────────
# SHARED OUTPUT METADATA / FORMATTING — used by both tabs' result cards
# ─────────────────────────────────────────
OUTPUT_META_STEMS = [
    ("TotalOperationalCO2Save", {"label": "Operational CO₂ saved",     "icon": "🌿", "group": "carbon", "fmt": "co2"}),
    ("TotalEmbodiedCO2",        {"label": "Embodied CO₂",              "icon": "🏗️", "group": "carbon", "fmt": "co2"}),
    ("TotalCO2Sav",             {"label": "Total CO₂ saved (GHG)",     "icon": "🌍", "group": "carbon", "fmt": "co2"}),
    ("AnnSCCSav",               {"label": "Annual SCC savings",        "icon": "💵", "group": "cost",   "fmt": "money"}),
    ("CostAnnualSysSave",       {"label": "Owner annual savings",      "icon": "💰", "group": "cost",   "fmt": "money"}),
    ("BaseCostAnnual",          {"label": "Base annual cost",          "icon": "🧾", "group": "cost",   "fmt": "money"}),
    ("PercentCostSysSav",       {"label": "% cost system savings",     "icon": "📈", "group": "cost",   "fmt": "percent"}),
    ("AnnGovtCostSav",          {"label": "Government annual savings", "icon": "🏛️", "group": "cost",   "fmt": "money"}),
]

def output_meta(col):
    for stem, meta in OUTPUT_META_STEMS:
        if col.lower().startswith(stem.lower()):
            return meta
    return {"label": col.replace("_", " "), "icon": "📊", "group": "cost", "fmt": "number"}

def format_output_value(meta, val):
    fmt = meta.get("fmt", "number")
    if fmt == "money":
        return f"${val:,.0f}"
    if fmt == "percent":
        return f"{val:.1f}%"
    if fmt == "co2":
        return f"{val:,.0f} kg"
    return f"{val:,.1f}"

def format_input_value(meta, val):
    unit = meta.get("unit", "—")
    if unit == "$":
        return f"${val:,.0f}"
    if unit in ("%", r"\%"):
        return f"{val:.2f}%"
    if unit == r"m^2":
        return f"{val:,.0f} m²"
    return f"{val:.3f}"

def render_metric_card(label, icon, val_str, pct, is_secondary=False):
    """Shared var-card renderer used by both the city and archetype result panels."""
    pct = max(0, min(100, pct))
    icon_class = "var-icon-economic" if is_secondary else "var-icon-building"
    bar_color  = "#78350f" if is_secondary else "#1a1a2e"
    st.markdown(f"""
    <div class="var-card">
      <div class="var-icon {icon_class}">{icon}</div>
      <div class="var-info">
        <div class="var-label">{label}</div>
        <div class="var-value">{val_str}</div>
      </div>
      <div class="bar-wrap">
        <div class="bar-track">
          <div class="bar-fill" style="width:{pct}%;background:{bar_color};"></div>
        </div>
        <div class="bar-pct">{pct}%</div>
      </div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# ── CHANGED: Load one model per city ─────
# Place all city_models/ files next to app.py
# ─────────────────────────────────────────
MODEL_DIR = os.path.join(BASE_DIR, "city_models")  # already set at top, kept for clarity

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

# ─────────────────────────────────────────
# Load the archetype (linear regression) model — for the second tab
# Place archetype_model/ next to app.py, generated by train_archetype_model.py
# ─────────────────────────────────────────
ARCH_MODEL_DIR  = os.path.join(BASE_DIR, "archetype_model")
ARCH_MODEL_PATH = os.path.join(ARCH_MODEL_DIR, "archetype_linear_model.pkl")
ARCH_STATS_PATH = os.path.join(ARCH_MODEL_DIR, "archetype_stats.json")

@st.cache_resource
def load_archetype_model():
    import json as _json
    if not (os.path.exists(ARCH_MODEL_PATH) and os.path.exists(ARCH_STATS_PATH)):
        return None, None
    bundle = joblib.load(ARCH_MODEL_PATH)
    with open(ARCH_STATS_PATH) as f:
        stats = _json.load(f)
    return bundle, stats

arch_bundle, arch_stats = load_archetype_model()
archetype_model_ok = arch_bundle is not None

# ─────────────────────────────────────────
# HEADER + TABS
# ─────────────────────────────────────────
st.markdown("## 🏗️ AI Retrofit Tool")
st.caption("Two tools in one app: an optimizer by city, and a plain regression predictor by building archetype.")

tab_city, tab_arch = st.tabs(["🏙️ By City — Optimizer", "📐 By Archetype — Predictor"])

with tab_city:
    st.markdown("### 🏙️ City Optimizer")
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
        m2.metric("GHG reduction",   f"{best['GHG'] / 1000:,.1f} tCO₂e")
        m3.metric("Owner savings",   f"${best['Owner']:,.0f}")
        m4.metric("Gov savings",     f"${best['Gov']:,.0f}")

        st.markdown("---")

        # ─────────────────────────────────────────────
        # VARIABLE CARD RENDERER  (LaTeX symbols + units)
        # ─────────────────────────────────────────────
        def var_card(k, meta, best_val, is_economic=False):
            lo_v, hi_v = RANGES[k]
            pct = round((best_val - lo_v) / (hi_v - lo_v + 1e-9) * 100)
            pct = max(0, min(100, pct))

            # ── numeric value string (no unit text — unit shown via LaTeX below) ──
            if k in ("Loan", "Rebate") or meta["unit"] == "$":
                val_str = f"${best_val:,.0f}"
            elif meta["unit"] in ("%", r"\%"):
                val_str = f"{best_val:.2f} %"
            else:
                val_str = f"{best_val:.3f}"

            icon_class = "var-icon-economic" if is_economic else "var-icon-building"
            bar_color  = "#78350f"  if is_economic else "#1a1a2e"

            # ── render the card HTML (no symbol/unit text inside HTML) ────────────
            st.markdown(f"""
            <div class="var-card">
              <div class="var-icon {icon_class}">{meta['icon']}</div>
              <div class="var-info">
                <div class="var-label">{meta['label']}</div>
                <div class="var-value">{val_str}</div>
              </div>
              <div class="bar-wrap">
                <div class="bar-track">
                  <div class="bar-fill" style="width:{pct}%;background:{bar_color};"></div>
                </div>
                <div class="bar-pct">{pct}%</div>
              </div>
            </div>""", unsafe_allow_html=True)

            # ── render symbol and unit via LaTeX ──────────────────────────────────
            sym  = meta["symbol"]
            unit = meta["unit"]
            if unit not in ("—", "$"):
                st.markdown(
                    f"$\\quad {sym}$ &nbsp;&nbsp; "
                    f"<span style='font-size:11px;color:#64748b;'>"
                    f"Unit: ${unit}$ &nbsp;|&nbsp; Range: {lo_v} – {hi_v}</span>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"$\\quad {sym}$ &nbsp;&nbsp; "
                    f"<span style='font-size:11px;color:#64748b;'>"
                    f"Range: {lo_v} – {hi_v}</span>",
                    unsafe_allow_html=True
                )

        # ── Two result columns ─────────────────────────
        col_build, col_econ = st.columns(2, gap="large")

        with col_build:
            st.markdown('<div class="result-group-title">🏗️ Building features</div>',
                        unsafe_allow_html=True)
            for k, meta in BUILDING_VARS.items():
                var_card(k, meta, float(best[k]), is_economic=False)

        with col_econ:
            st.markdown('<div class="result-group-title">💰 Economic parameters</div>',
                        unsafe_allow_html=True)
            for k, meta in ECONOMIC_VARS.items():
                var_card(k, meta, float(best[k]), is_economic=True)

        st.markdown("---")

        # ── Predicted outcomes — full breakdown of all 8 output columns,
        # same rounding/formatting and card layout as the archetype tab ──────
        res_carbon, res_cost = st.columns(2, gap="large")

        with res_carbon:
            st.markdown('<div class="result-group-title">🌿 Carbon impact</div>', unsafe_allow_html=True)
            for c in OUTPUT_COLS_ALL:
                meta = output_meta(c)
                if meta["group"] != "carbon":
                    continue
                lo_v, hi_v = float(df[c].min()), float(df[c].max())
                val = float(best[c])
                pct = round((val - lo_v) / (hi_v - lo_v + 1e-9) * 100)
                render_metric_card(meta["label"], meta["icon"], format_output_value(meta, val), pct, is_secondary=False)

        with res_cost:
            st.markdown('<div class="result-group-title">💰 Cost impact</div>', unsafe_allow_html=True)
            for c in OUTPUT_COLS_ALL:
                meta = output_meta(c)
                if meta["group"] != "cost":
                    continue
                lo_v, hi_v = float(df[c].min()), float(df[c].max())
                val = float(best[c])
                pct = round((val - lo_v) / (hi_v - lo_v + 1e-9) * 100)
                render_metric_card(meta["label"], meta["icon"], format_output_value(meta, val), pct, is_secondary=True)

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

            fig_par = go.Figure()

            # All simulation points
            fig_par.add_trace(go.Scatter3d(
                x=sample_plot["Owner"],
                y=sample_plot["Gov"],
                z=sample_plot["GHG"] / 1000,
                mode="markers",
                marker=dict(
                    size=3,
                    color=sample_plot["Score"],
                    colorscale=["#e2e8f0", "#1a1a2e"],
                    opacity=0.5,
                    showscale=False,
                ),
                name="Simulations",
                hovertemplate=(
                    "Owner: $%{x:,.0f}<br>"
                    "Gov: $%{y:,.0f}<br>"
                    "GHG: %{z:,.1f} tCO₂e<br>"
                    "<extra></extra>"
                ),
            ))

            # Best point
            fig_par.add_trace(go.Scatter3d(
                x=[best["Owner"]],
                y=[best["Gov"]],
                z=[best["GHG"] / 1000],
                mode="markers",
                marker=dict(size=10, color="#b91c1c", symbol="diamond"),
                name="Best",
                hovertemplate=(
                    "⭐ Best solution<br>"
                    "Owner: $%{x:,.0f}<br>"
                    "Gov: $%{y:,.0f}<br>"
                    "GHG: %{z:,.1f} tCO₂e<br>"
                    "<extra></extra>"
                ),
            ))

            fig_par.update_layout(
                title=dict(text="Pareto space (3 objectives)", font=dict(size=14, color="#0a0a0a")),
                height=350,
                margin=dict(t=40, b=10, l=10, r=10),
                paper_bgcolor="white",
                font=dict(color="#0a0a0a", size=11),
                scene=dict(
                    xaxis=dict(
                        title=dict(text="Owner savings ($)", font=dict(size=10)),
                        backgroundcolor="white",
                        gridcolor="#e2e8f0",
                        showbackground=True,
                        tickfont=dict(size=9),
                    ),
                    yaxis=dict(
                        title=dict(text="Gov savings ($)", font=dict(size=10)),
                        backgroundcolor="white",
                        gridcolor="#e2e8f0",
                        showbackground=True,
                        tickfont=dict(size=9),
                    ),
                    zaxis=dict(
                        title=dict(text="GHG (tCO₂e)", font=dict(size=10)),
                        backgroundcolor="white",
                        gridcolor="#e2e8f0",
                        showbackground=True,
                        tickfont=dict(size=9),
                    ),
                ),
                legend=dict(
                    x=0.01, y=0.99,
                    font=dict(size=10),
                    bgcolor="rgba(255,255,255,0.8)",
                ),
            )
            st.plotly_chart(fig_par, use_container_width=True)

with tab_arch:
    if not archetype_model_ok:
        st.error(
            "⚠️ No archetype model found — place the `archetype_model/` folder "
            "next to app.py. Run `train_archetype_model.py` to generate it."
        )
    else:
        arch_input_cols  = arch_stats["input_cols"]
        arch_output_cols = arch_stats["output_cols"]
        arch_ranges      = arch_stats["input_ranges"]
        arch_list        = arch_stats["archetypes"]
        arch_cat_col     = arch_stats["categorical_col"]
        arch_out_stats   = arch_stats["output_stats"]

        # ── Apply a pending optimizer result BEFORE any arch_in_* widget is
        # instantiated this run — Streamlit forbids writing to a widget's
        # session_state key after that widget has already been created. ──────
        if "_arch_pending" in st.session_state:
            _pending = st.session_state.pop("_arch_pending")
            for _c, _v in _pending["inputs"].items():
                st.session_state[f"arch_in_{_c}"] = float(_v)
            st.session_state["_arch_last"] = _pending["archetype"]
            st.session_state["_arch_show_optimized"] = _pending

        # ── Icon / label metadata ──────────────────────────────────────────────
        ARCH_EXTRA_INPUT_META = {
            "ElectricityInflationRate": {"label": "Electricity inflation", "unit": r"\%", "symbol": r"i_e",     "icon": "⚡"},
            "FuelInflationRate":        {"label": "Fuel inflation",        "unit": r"\%", "symbol": r"i_f",     "icon": "⛽"},
            "BuildingFootprintArea_m2": {"label": "Building footprint",    "unit": r"m^2","symbol": r"A_{fp}",  "icon": "📏"},
        }

        def input_meta(col):
            if col in BUILDING_VARS:
                m = dict(BUILDING_VARS[col]); m["group"] = "building"; return m
            if col in ECONOMIC_VARS:
                m = dict(ECONOMIC_VARS[col]); m["group"] = "economic"; return m
            if col in ARCH_EXTRA_INPUT_META:
                m = dict(ARCH_EXTRA_INPUT_META[col]); m["group"] = "economic"; return m
            if col.lower().startswith("weight"):
                return {"label": col.replace("_", " "), "unit": r"—", "symbol": r"w", "icon": "⚖️", "group": "economic"}
            return {"label": col.replace("_", " "), "unit": r"—", "symbol": col, "icon": "🔧", "group": "building"}

        # output_meta / format_output_value / format_input_value are defined once,
        # shared with the city tab (see near the top of this file).

        def find_output_col(stem):
            for c in arch_output_cols:
                if c.lower().startswith(stem.lower()):
                    return c
            return None

        OWNER_COL = find_output_col("CostAnnualSysSave")
        GOV_COL   = find_output_col("AnnGovtCostSav")
        GHG_COL   = find_output_col("TotalCO2Sav")

        # ── Resolve actual column names for the 5 screenshot fields ────────────
        def resolve_col(*substrings):
            for c in arch_input_cols:
                cl = c.lower()
                if all(s in cl for s in substrings):
                    return c
            return None

        COL_FOOTPRINT = resolve_col("footprint") or resolve_col("area")
        COL_INFIL     = resolve_col("infiltrat")
        COL_GLAZING   = resolve_col("glaz")
        COL_ROOF_R    = resolve_col("rvalue", "roof") or resolve_col("roof", "r")
        COL_WALL_R    = resolve_col("rvalue", "wall") or resolve_col("wall", "r")

        # ── Objective-weight input columns — these mirror the Owner/Gov/GHG
        # sliders below, so they get sensible non-zero defaults instead of 0 ──
        COL_WEIGHT_OWNER = resolve_col("weight", "cost")
        COL_WEIGHT_GOV   = resolve_col("weight", "govt")
        COL_WEIGHT_GHG   = resolve_col("weight", "ghg")

        # ── Inflation-rate columns — also moved to the left panel ──────────────
        COL_ELEC_INFL = resolve_col("electric", "inflat")
        COL_FUEL_INFL = resolve_col("fuel", "inflat")

        # Columns rendered on the LEFT (with the archetype picker) instead of
        # in the general parameter list on the right.
        LEFT_PARAM_COLS = [c for c in (COL_FOOTPRINT, COL_WEIGHT_OWNER, COL_WEIGHT_GOV, COL_WEIGHT_GHG,
                                        COL_ELEC_INFL, COL_FUEL_INFL) if c]
        LEFT_PARAM_DEFAULTS = {
            COL_WEIGHT_OWNER: 0.60,
            COL_WEIGHT_GOV:   0.20,
            COL_WEIGHT_GHG:   0.20,
            COL_ELEC_INFL:    0.10,
            COL_FUEL_INFL:    0.10,
            COL_FOOTPRINT:    130.0,  # fixed default (matches the city tab), overrides the archetype table value
        }

        # ── Screenshot defaults: {archetype: {BuildingFootprintArea_m2, Infiltration, Glazing, Rvalue_roof, Rvalue_wall}} ──
        # Glazing Ratio [%] from the screenshot is converted to a 0-1 fraction to match the model's Glazing column.
        SCREENSHOT_DEFAULTS = {
            "Pre-1900": {"BuildingFootprintArea_m2": 170, "Infiltration": 10.0, "Glazing": 9.5/100,  "Rvalue_roof": 1.99, "Rvalue_wall": 1.5},
            "1910":     {"BuildingFootprintArea_m2": 150, "Infiltration": 9.44, "Glazing": 9.6/100,  "Rvalue_roof": 4.36, "Rvalue_wall": 1.57},
            "1920":     {"BuildingFootprintArea_m2": 136, "Infiltration": 9.55, "Glazing": 8.73/100, "Rvalue_roof": 3.23, "Rvalue_wall": 1.56},
            "1930":     {"BuildingFootprintArea_m2": 165, "Infiltration": 9.7,  "Glazing": 12.4/100, "Rvalue_roof": 3.66, "Rvalue_wall": 1.6},
            "1940":     {"BuildingFootprintArea_m2": 134, "Infiltration": 9.1,  "Glazing": 8.95/100, "Rvalue_roof": 3.8,  "Rvalue_wall": 1.67},
            "1950":     {"BuildingFootprintArea_m2": 145, "Infiltration": 9.6,  "Glazing": 9.7/100,  "Rvalue_roof": 3.3,  "Rvalue_wall": 1.64},
            "1960":     {"BuildingFootprintArea_m2": 168, "Infiltration": 7.6,  "Glazing": 10.0/100, "Rvalue_roof": 3.6,  "Rvalue_wall": 1.77},
            "1970":     {"BuildingFootprintArea_m2": 185, "Infiltration": 7.0,  "Glazing": 11.0/100, "Rvalue_roof": 4.0,  "Rvalue_wall": 2.17},
            "1980":     {"BuildingFootprintArea_m2": 183, "Infiltration": 7.0,  "Glazing": 11.0/100, "Rvalue_roof": 4.1,  "Rvalue_wall": 2.2},
            "1990":     {"BuildingFootprintArea_m2": 210, "Infiltration": 6.1,  "Glazing": 10.0/100, "Rvalue_roof": 4.38, "Rvalue_wall": 2.3},
            "2000":     {"BuildingFootprintArea_m2": 205, "Infiltration": 5.2,  "Glazing": 11.0/100, "Rvalue_roof": 4.76, "Rvalue_wall": 2.6},
            "2010":     {"BuildingFootprintArea_m2": 283, "Infiltration": 4.7,  "Glazing": 10.0/100, "Rvalue_roof": 6.7,  "Rvalue_wall": 2.9},
            # parameter-variant archetypes
            "High_Infiltration": {"BuildingFootprintArea_m2": 132, "Infiltration": 23.1, "Glazing": 11.5/100, "Rvalue_roof": 1.1, "Rvalue_wall": 1.1},
            "Low_Infiltration":  {"BuildingFootprintArea_m2": 213, "Infiltration": 2.1,  "Glazing": 13.3/100, "Rvalue_roof": 4.7, "Rvalue_wall": 2.2},
            "High_GR":           {"BuildingFootprintArea_m2": 220, "Infiltration": 4.9,  "Glazing": 40.9/100, "Rvalue_roof": 5.9, "Rvalue_wall": 1.9},
            "Low_GR":            {"BuildingFootprintArea_m2": 161, "Infiltration": 5.0,  "Glazing": 2.7/100,  "Rvalue_roof": 4.8, "Rvalue_wall": 2.3},
            "High_Roof_R":       {"BuildingFootprintArea_m2": 147, "Infiltration": 3.9,  "Glazing": 9.3/100,  "Rvalue_roof": 9.9, "Rvalue_wall": 2.2},
            "Low_Roof_R":        {"BuildingFootprintArea_m2": 174, "Infiltration": 7.9,  "Glazing": 10.0/100, "Rvalue_roof": 1.0, "Rvalue_wall": 1.7},
            # NOTE: assumed to be Wall R variants (screenshot mislabeled both as "Roof R Value") — fix here if wrong
            "High_Wall_R":       {"BuildingFootprintArea_m2": 194, "Infiltration": 4.2,  "Glazing": 13.1/100, "Rvalue_roof": 4.3, "Rvalue_wall": 4.0},
            "Low_Wall_R":        {"BuildingFootprintArea_m2": 93,  "Infiltration": 9.6,  "Glazing": 7.4/100,  "Rvalue_roof": 2.2, "Rvalue_wall": 0.5},
        }

        def _norm_label(s):
            return "".join(ch for ch in s.lower() if ch.isalnum())

        _SCREENSHOT_LOOKUP = {_norm_label(k): v for k, v in SCREENSHOT_DEFAULTS.items()}

        def get_screenshot_defaults(archetype_label):
            return _SCREENSHOT_LOOKUP.get(_norm_label(archetype_label))

        FIELD_TO_COL = {
            "BuildingFootprintArea_m2": COL_FOOTPRINT,
            "Infiltration":             COL_INFIL,
            "Glazing":                  COL_GLAZING,
            "Rvalue_roof":              COL_ROOF_R,
            "Rvalue_wall":              COL_WALL_R,
        }

        # ─────────────────────────────────────────
        # HEADER
        # ─────────────────────────────────────────
        st.markdown("### 📐 Archetype Predictor & Retrofit Recommender")
        st.caption(
            "Defaults come from your archetype reference table (footprint, infiltration, "
            "glazing, roof/wall R-value); every other input starts at 0 until you set it "
            "or run the recommender."
        )
        st.caption(
            f"Model: Linear Regression · trained on {arch_stats['n_rows']:,} rows · "
            f"test R² = {arch_stats['model_r2_test']:.3f}"
        )
        st.markdown("---")

        arch_col_pick, arch_col_ctrl = st.columns([1.6, 1], gap="large")

        # ─────────────────────────────────────────
        # LEFT — ARCHETYPE PICKER
        # ─────────────────────────────────────────
        with arch_col_pick:
            st.markdown('<div class="section-label">Select building archetype</div>',
                        unsafe_allow_html=True)

            def _apply_archetype_defaults(a):
                """Screenshot values for the 5 known fields; weight columns default
                to 0.60/0.20/0.20 (matching the objective sliders); everything else = 0."""
                for c in arch_input_cols:
                    st.session_state[f"arch_in_{c}"] = 0.0
                sd = get_screenshot_defaults(a)
                if sd:
                    for field, val in sd.items():
                        col = FIELD_TO_COL.get(field)
                        if col:
                            st.session_state[f"arch_in_{col}"] = float(val)
                for col, val in LEFT_PARAM_DEFAULTS.items():
                    if col:
                        st.session_state[f"arch_in_{col}"] = val
                st.session_state["_arch_last"] = a
                st.session_state["_arch_has_screenshot"] = sd is not None

            if "_arch_last" not in st.session_state:
                _apply_archetype_defaults(arch_list[0])

            selected_archetype = st.session_state["_arch_last"]
            st.markdown(
                f'<div class="city-selected">📐 {selected_archetype}</div>',
                unsafe_allow_html=True,
            )
            if not st.session_state.get("_arch_has_screenshot", True):
                st.caption("⚠️ No screenshot reference values for this archetype — all inputs default to 0.")

            btn_cols = st.columns(5)
            for i, a in enumerate(arch_list):
                with btn_cols[i % 5]:
                    if st.button(a, key=f"arch_btn_{a}",
                                 type="primary" if a == selected_archetype else "secondary",
                                 use_container_width=True):
                        _apply_archetype_defaults(a)
                        st.rerun()

            st.markdown('<div class="section-label" style="margin-top:12px;">Objective weights (for recommender)</div>',
                        unsafe_allow_html=True)
            w_owner_a = st.slider("🏠 Owner savings", 0.0, 1.0, 0.60, 0.05, key="arch_w_owner")
            w_gov_a   = st.slider("🏛️ Gov savings",   0.0, 1.0, 0.20, 0.05, key="arch_w_gov")
            w_ghg_a   = st.slider("🌿 GHG reduction", 0.0, 1.0, 0.20, 0.05, key="arch_w_ghg")
            wsum_a = round(w_owner_a + w_gov_a + w_ghg_a, 2)
            if abs(wsum_a - 1.0) > 0.01:
                st.markdown(f'<p class="warn">⚠️ Weights sum to {wsum_a:.2f} — must equal 1.0</p>',
                            unsafe_allow_html=True)
                weights_ok_a = False
            else:
                st.success(f"Weights ✓ ({wsum_a:.2f})")
                weights_ok_a = True

            st.markdown('<div class="section-label" style="margin-top:12px;">Scenario parameters</div>',
                        unsafe_allow_html=True)
            st.caption(
                "Building footprint defaults to 130 m² for every archetype. The three "
                "model weight inputs mirror the objective sliders above them, and the "
                "two inflation rates default to 10% — edit any of them here if needed."
            )
            for c in LEFT_PARAM_COLS:
                lo, hi = arch_ranges[c]
                step = round(max((hi - lo) / 100, 0.001), 4) if hi > lo else 1.0
                st.number_input(
                    input_meta(c)["label"], step=float(step), format="%.4f",
                    key=f"arch_in_{c}", help=f"Training range: {lo} – {hi}",
                )

        # ─────────────────────────────────────────
        # RIGHT — INPUT PARAMETERS (editable, manual "what-if" prediction)
        # ─────────────────────────────────────────
        with arch_col_ctrl:
            arch_user_inputs = {}
            building_inputs = [c for c in arch_input_cols if input_meta(c)["group"] == "building" and c not in LEFT_PARAM_COLS]
            economic_inputs = [c for c in arch_input_cols if input_meta(c)["group"] == "economic" and c not in LEFT_PARAM_COLS]

            st.markdown('<div class="section-label">Building parameters</div>', unsafe_allow_html=True)
            for c in building_inputs:
                lo, hi = arch_ranges[c]
                step = round(max((hi - lo) / 100, 0.001), 4) if hi > lo else 1.0
                arch_user_inputs[c] = st.number_input(
                    input_meta(c)["label"], step=float(step), format="%.4f",
                    key=f"arch_in_{c}", help=f"Training range: {lo} – {hi}",
                )

            st.markdown('<div class="section-label" style="margin-top:12px;">Economic / scenario parameters</div>',
                        unsafe_allow_html=True)
            for c in economic_inputs:
                lo, hi = arch_ranges[c]
                step = round(max((hi - lo) / 100, 0.001), 4) if hi > lo else 1.0
                arch_user_inputs[c] = st.number_input(
                    input_meta(c)["label"], step=float(step), format="%.4f",
                    key=f"arch_in_{c}", help=f"Training range: {lo} – {hi}",
                )

            st.markdown("<br>", unsafe_allow_html=True)
            arch_predict_clicked = st.button(
                "▶ Predict these exact values", use_container_width=True, key="arch_predict_btn",
            )
            optimize_clicked = st.button(
                f"🔍 Find recommended retrofit for {selected_archetype}",
                type="primary", use_container_width=True, key="arch_optimize_btn",
                disabled=not weights_ok_a,
            )
            if not weights_ok_a:
                st.caption("← Fix the objective weights above to enable the recommender")

        # ─────────────────────────────────────────
        # RESULT RENDERER (shared by both actions)
        # ─────────────────────────────────────────
        def render_arch_results(input_values, output_values, heading):
            st.markdown("---")
            st.success(heading)

            headline_cols = [c for c in (OWNER_COL, GOV_COL, GHG_COL) if c]
            headline_cols += [c for c in arch_output_cols if c not in headline_cols]
            headline_cols = headline_cols[:4]
            mcols = st.columns(len(headline_cols))
            for i, c in enumerate(headline_cols):
                meta = output_meta(c)
                mcols[i].metric(f"{meta['icon']} {meta['label']}",
                                 format_output_value(meta, output_values[c]))

            st.markdown("---")

            arch_var_card = lambda label, icon, val_str, pct, is_cost: render_metric_card(
                label, icon, val_str, pct, is_secondary=is_cost
            )

            # ── Retrofit plan — same variable set as the city tab's result cards
            # (BUILDING_VARS + ECONOMIC_VARS only; footprint/weights/inflation are
            # scenario inputs, not "retrofit choices", so they're left out here) ──
            st.markdown('<div class="result-group-title">🛠️ Recommended retrofit plan</div>',
                        unsafe_allow_html=True)
            plan_build_cols = [c for c in BUILDING_VARS if c in arch_input_cols]
            plan_econ_cols  = [c for c in ECONOMIC_VARS if c in arch_input_cols]

            col_build, col_econ = st.columns(2, gap="large")
            with col_build:
                st.markdown('<div class="section-label">🏗️ Building features</div>', unsafe_allow_html=True)
                for c in plan_build_cols:
                    meta = BUILDING_VARS[c]
                    lo, hi = arch_ranges[c]
                    val = input_values[c]
                    pct = round((val - lo) / (hi - lo + 1e-9) * 100)
                    arch_var_card(meta["label"], meta["icon"], format_input_value(meta, val), pct, is_cost=False)

            with col_econ:
                st.markdown('<div class="section-label">💰 Economic parameters</div>', unsafe_allow_html=True)
                for c in plan_econ_cols:
                    meta = ECONOMIC_VARS[c]
                    lo, hi = arch_ranges[c]
                    val = input_values[c]
                    pct = round((val - lo) / (hi - lo + 1e-9) * 100)
                    arch_var_card(meta["label"], meta["icon"], format_input_value(meta, val), pct, is_cost=True)

            st.markdown("---")

            # ── Predicted outcomes ──────────────────────────────────────────────
            res_carbon, res_cost = st.columns(2, gap="large")
            with res_carbon:
                st.markdown('<div class="result-group-title">🌿 Carbon impact</div>', unsafe_allow_html=True)
                for c in arch_output_cols:
                    meta = output_meta(c)
                    if meta["group"] != "carbon":
                        continue
                    lo_v, hi_v = arch_out_stats[c]["min"], arch_out_stats[c]["max"]
                    val = output_values[c]
                    pct = round((val - lo_v) / (hi_v - lo_v + 1e-9) * 100)
                    arch_var_card(meta["label"], meta["icon"], format_output_value(meta, val), pct, is_cost=False)

            with res_cost:
                st.markdown('<div class="result-group-title">💰 Cost impact</div>', unsafe_allow_html=True)
                for c in arch_output_cols:
                    meta = output_meta(c)
                    if meta["group"] != "cost":
                        continue
                    lo_v, hi_v = arch_out_stats[c]["min"], arch_out_stats[c]["max"]
                    val = output_values[c]
                    pct = round((val - lo_v) / (hi_v - lo_v + 1e-9) * 100)
                    arch_var_card(meta["label"], meta["icon"], format_output_value(meta, val), pct, is_cost=True)

            with st.expander("Model quality (test-set R² per output)"):
                st.json(arch_stats["per_output_r2"])
                st.caption(
                    "R² close to 1.0 means the linear model explains most of the "
                    "variance for that output on held-out data. Lower values mean "
                    "that output is less linear — treat those predictions with more caution."
                )


        # ─────────────────────────────────────────
        # ACTION 1 — predict the exact typed-in values
        # ─────────────────────────────────────────
        if arch_predict_clicked:
            row = {c: float(st.session_state.get(f"arch_in_{c}", 0.0)) for c in arch_input_cols}
            row[arch_cat_col] = selected_archetype
            row_df = pd.DataFrame([row])
            row_df = pd.get_dummies(row_df, columns=[arch_cat_col])
            row_df = row_df.reindex(columns=arch_bundle["feature_cols"], fill_value=0)
            X_sc = arch_bundle["scaler"].transform(row_df.values.astype(float))
            pred = arch_bundle["model"].predict(X_sc)[0]
            arch_result = dict(zip(arch_output_cols, pred.tolist()))
            render_arch_results(row, arch_result, f"✅ Prediction for **{selected_archetype}** with your entered values")

        # ─────────────────────────────────────────
        # ACTION 2 — search for the best retrofit plan
        # ─────────────────────────────────────────
        if optimize_clicked:
            with st.spinner(f"Searching retrofit combinations for {selected_archetype}…"):
                # Fixed (not searched): footprint stays at the archetype baseline;
                # weight/inflation inputs stay at whatever's currently set (0 by default).
                FIXED_COLS = set()
                if COL_FOOTPRINT:
                    FIXED_COLS.add(COL_FOOTPRINT)
                for c in arch_input_cols:
                    if c.lower().startswith("weight") or "inflation" in c.lower():
                        FIXED_COLS.add(c)

                OPT_COLS = [c for c in arch_input_cols if c not in FIXED_COLS]

                N = 4000
                rng = np.random.default_rng(42)
                numeric_df = pd.DataFrame({
                    c: rng.uniform(arch_ranges[c][0], arch_ranges[c][1], N) for c in OPT_COLS
                })
                for c in FIXED_COLS:
                    numeric_df[c] = float(st.session_state.get(f"arch_in_{c}", 0.0))
                numeric_df = numeric_df[arch_input_cols]  # consistent column order

                encode_df = numeric_df.copy()
                encode_df[arch_cat_col] = selected_archetype
                encode_df = pd.get_dummies(encode_df, columns=[arch_cat_col])
                encode_df = encode_df.reindex(columns=arch_bundle["feature_cols"], fill_value=0)

                X_sc = arch_bundle["scaler"].transform(encode_df.values.astype(float))
                preds = arch_bundle["model"].predict(X_sc)
                pred_df = pd.DataFrame(preds, columns=arch_output_cols)

                def _norm(s):
                    return (s - s.min()) / (s.max() - s.min() + 1e-9)

                owner_n = _norm(pred_df[OWNER_COL]) if OWNER_COL else 0
                gov_n   = _norm(pred_df[GOV_COL])   if GOV_COL   else 0
                ghg_n   = 1 - _norm(pred_df[GHG_COL]) if GHG_COL else 0

                score = w_owner_a * owner_n + w_gov_a * gov_n + w_ghg_a * ghg_n
                best_idx = int(score.idxmax())

                best_inputs  = numeric_df.iloc[best_idx].to_dict()
                best_outputs = pred_df.iloc[best_idx].to_dict()

            # Stash the result and rerun — the pending-apply block at the top of
            # this tab will write it into the arch_in_* widgets before they're
            # (re)created, then render the results below.
            st.session_state["_arch_pending"] = {
                "inputs": best_inputs,
                "outputs": best_outputs,
                "archetype": selected_archetype,
                "n_searched": N,
            }
            st.rerun()

        # ─────────────────────────────────────────
        # Show the optimizer result after the rerun above has applied it
        # ─────────────────────────────────────────
        if st.session_state.get("_arch_show_optimized"):
            _shown = st.session_state.pop("_arch_show_optimized")
            st.caption(
                f"Searched {_shown.get('n_searched', 4000):,} random retrofit combinations for "
                f"**{_shown['archetype']}** (footprint fixed at the archetype baseline; "
                f"weight/inflation inputs held fixed) and scored each with your objective weights."
            )
            render_arch_results(
                _shown["inputs"], _shown["outputs"],
                f"🏆 Recommended retrofit plan for **{_shown['archetype']}**",
            )
