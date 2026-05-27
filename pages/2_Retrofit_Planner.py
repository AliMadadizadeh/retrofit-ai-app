import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(
    page_title="Retrofit Planner | Madadiz",
    page_icon="🏗️",
    layout="wide"
)

# ── Load dataset ──────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "retrofit_dataset_final_solution1.csv")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

DATASET = load_data()

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding: 1.5rem 2rem;}
    .section-title {
        font-size: 13px; font-weight: 800; color: #1a1a2e;
        text-transform: uppercase; letter-spacing: 0.1em;
        margin: 1.5rem 0 0.8rem; border-bottom: 2px solid #1a1a2e;
        padding-bottom: 6px;
    }
    .result-card {
        background: #f0f4f8; border: 1.5px solid #c9d4e0;
        border-radius: 12px; padding: 16px 20px; margin-bottom: 12px;
    }
    .result-card-title {
        font-size: 13px; font-weight: 800; color: #1a1a2e;
        text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px;
    }
    .result-card-value {
        font-size: 26px; font-weight: 800; color: #0a0a0a;
        letter-spacing: -0.02em; line-height: 1.1;
    }
    .result-card-sub {
        font-size: 12px; color: #64748b; margin-top: 3px; font-weight: 600;
    }
    .rec-card {
        background: #ffffff; border: 1.5px solid #c9d4e0;
        border-radius: 10px; padding: 14px 16px; margin-bottom: 10px;
        border-left: 4px solid #1a1a2e;
    }
    .rec-card.high { border-left-color: #b91c1c; }
    .rec-card.medium { border-left-color: #d97706; }
    .rec-card.low { border-left-color: #15803d; }
    .rec-title { font-size: 14px; font-weight: 800; color: #0a0a0a; margin-bottom: 4px; }
    .rec-value { font-size: 18px; font-weight: 800; color: #1a1a2e; }
    .rec-desc { font-size: 12px; color: #64748b; margin-top: 4px; line-height: 1.5; }
    .badge {
        display: inline-block; font-size: 10px; font-weight: 800;
        letter-spacing: 0.08em; text-transform: uppercase;
        padding: 2px 8px; border-radius: 4px; margin-left: 6px;
    }
    .badge-high { background: #fee2e2; color: #b91c1c; }
    .badge-med  { background: #fef3c7; color: #d97706; }
    .badge-low  { background: #dcfce7; color: #15803d; }
    .bar-track { height: 8px; background: #e2e8f0; border-radius: 99px; overflow: hidden; margin: 4px 0; }
    .bar-fill  { height: 8px; border-radius: 99px; background: #1a1a2e; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🏗️ Retrofit Planner")
st.caption("Select your building parameters to find the best retrofit plan from our simulation database.")
st.markdown("---")

# ── Available cities and SSPs in dataset ─────────────────────────────────────
available_cities = sorted(DATASET["City"].unique().tolist())
available_ssps   = sorted(DATASET["SSP"].unique().tolist())

# ── FORM ─────────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown('<div class="section-title">📍 Location & Scenario</div>', unsafe_allow_html=True)
    city = st.selectbox("City", available_cities)
    ssp  = st.selectbox("Climate Scenario (SSP)", available_ssps,
                        help="SSP126 = low emissions, SSP245 = moderate, SSP585 = high emissions")

with col2:
    st.markdown('<div class="section-title">🎯 Your Priorities</div>', unsafe_allow_html=True)
    st.caption("Set how much you care about each objective (must total 1.0)")
    w_owner = st.slider("🏠 Owner cost savings",  0.0, 1.0, 0.60, 0.05)
    w_gov   = st.slider("🏛️ Government savings",  0.0, 1.0, 0.20, 0.05)
    w_ghg   = st.slider("🌿 GHG emission reduction", 0.0, 1.0, 0.20, 0.05)
    wsum = round(w_owner + w_gov + w_ghg, 2)
    if abs(wsum - 1.0) > 0.01:
        st.error(f"⚠️ Weights sum to {wsum} — must equal 1.0")
        weights_ok = False
    else:
        st.success(f"✓ Weights sum to {wsum}")
        weights_ok = True

with col3:
    st.markdown('<div class="section-title">🏗️ Building Info</div>', unsafe_allow_html=True)
    footprint = st.number_input("Building Footprint (m²)", min_value=30, max_value=2000, value=130, step=10)
    st.markdown("---")
    st.markdown('<div class="section-title">💰 Budget Preference</div>', unsafe_allow_html=True)
    budget_pref = st.selectbox("Investment Priority", [
        "Lowest cost first",
        "Best GHG reduction first",
        "Best owner savings first",
        "Balanced (default)"
    ])

st.markdown("---")
run = st.button(
    "▶ Find Best Retrofit Plan",
    type="primary",
    use_container_width=True,
    disabled=not weights_ok
)

# ── RESULTS ───────────────────────────────────────────────────────────────────
if run and weights_ok:
    with st.spinner(f"Searching {city} simulation database…"):

        # Filter dataset
        df = DATASET[
            (DATASET["City"] == city) &
            (DATASET["SSP"]  == ssp)
        ].copy().reset_index(drop=True)

        if df.empty:
            st.error(f"No simulation data found for {city} + {ssp}. Try a different combination.")
            st.stop()

        # Score rows
        def norm(s): return (s - s.min()) / (s.max() - s.min() + 1e-9)

        df["Owner_n"] = norm(df["CostAnnualSysSave_CAD"])
        df["Gov_n"]   = norm(df["AnnGovtCostSav_CAD"])
        df["GHG_n"]   = 1 - norm(df["TotalCO2Sav"])  # lower CO2 = better

        # Apply budget preference override
        if budget_pref == "Lowest cost first":
            df["Score"] = 1.0*df["Owner_n"]
        elif budget_pref == "Best GHG reduction first":
            df["Score"] = 1.0*df["GHG_n"]
        elif budget_pref == "Best owner savings first":
            df["Score"] = 1.0*df["Owner_n"]
        else:
            df["Score"] = w_owner*df["Owner_n"] + w_gov*df["Gov_n"] + w_ghg*df["GHG_n"]

        best = df.sort_values("Score", ascending=False).iloc[0]

    # ── Summary metrics ───────────────────────────────────────────────────────
    st.success(f"✅ Best retrofit plan found for **{city}** ({ssp}) from **{len(df):,}** simulations")
    st.markdown("---")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""<div class="result-card">
            <div class="result-card-title">💰 Owner Savings</div>
            <div class="result-card-value">${best['CostAnnualSysSave_CAD']:,.0f}</div>
            <div class="result-card-sub">Per year</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="result-card">
            <div class="result-card-title">🏛️ Gov Savings</div>
            <div class="result-card-value">${best['AnnGovtCostSav_CAD']:,.0f}</div>
            <div class="result-card-sub">Per year</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="result-card">
            <div class="result-card-title">🌿 GHG Reduction</div>
            <div class="result-card-value">{best['TotalCO2Sav']/1000:,.1f} t</div>
            <div class="result-card-sub">CO₂e over 20 years</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        base = best.get('BaseCostAnnual_CAD', 0)
        pct  = best.get('PercentCostSysSav_percent', 0)
        st.markdown(f"""<div class="result-card">
            <div class="result-card-title">📉 Cost Reduction</div>
            <div class="result-card-value">{pct:.1f}%</div>
            <div class="result-card-sub">Of annual energy cost</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Building retrofits ─────────────────────────────────────────────────────
    left, right = st.columns(2, gap="large")

    BUILDING_RECS = [
        {
            "key": "Rvalue_roof", "label": "Roof Insulation",
            "icon": "🏠", "unit": "m²K/W",
            "range": (5.46, 11.0),
            "desc": "Upgrade roof insulation to reduce heat loss in winter and heat gain in summer.",
            "priority": lambda v: "high" if v > 9 else "medium" if v > 7 else "low"
        },
        {
            "key": "Rvalue_wall", "label": "Wall Insulation",
            "icon": "🧱", "unit": "m²K/W",
            "range": (3.60, 8.00),
            "desc": "Improve wall R-value to reduce thermal bridging and improve envelope performance.",
            "priority": lambda v: "high" if v > 6.5 else "medium" if v > 5 else "low"
        },
        {
            "key": "Glazing", "label": "Glazing Ratio",
            "icon": "🪟", "unit": "",
            "range": (0.10, 0.40),
            "desc": "Optimize window-to-wall ratio for daylighting while minimizing heat loss.",
            "priority": lambda v: "medium" if v < 0.20 or v > 0.35 else "low"
        },
        {
            "key": "SHGC", "label": "Solar Heat Gain Coeff.",
            "icon": "🌤️", "unit": "",
            "range": (0.10, 0.70),
            "desc": "Select glazing with appropriate SHGC for your climate zone.",
            "priority": lambda v: "medium"
        },
        {
            "key": "Infiltration", "label": "Air Tightness",
            "icon": "💨", "unit": "ACH",
            "range": (0.50, 1.50),
            "desc": "Air sealing to reduce uncontrolled infiltration and improve comfort.",
            "priority": lambda v: "high" if v > 1.2 else "medium" if v > 0.8 else "low"
        },
        {
            "key": "Albedo_roof", "label": "Cool Roof (Albedo)",
            "icon": "☀️", "unit": "",
            "range": (0.10, 0.70),
            "desc": "High-reflectivity roofing reduces cooling loads in summer.",
            "priority": lambda v: "medium" if v > 0.5 else "low"
        },
        {
            "key": "A_PV", "label": "Photovoltaic System",
            "icon": "⚡", "unit": "area ratio",
            "range": (0.10, 0.60),
            "desc": "Rooftop solar PV to generate on-site electricity and reduce energy costs.",
            "priority": lambda v: "high" if v > 0.4 else "medium" if v > 0.2 else "low"
        },
        {
            "key": "A_ST", "label": "Solar Thermal System",
            "icon": "🌡️", "unit": "area ratio",
            "range": (0.10, 0.60),
            "desc": "Solar thermal collectors for domestic hot water heating.",
            "priority": lambda v: "medium" if v > 0.3 else "low"
        },
        {
            "key": "V_bites", "label": "BITES Thermal Storage",
            "icon": "🧊", "unit": "",
            "range": (0.05, 0.25),
            "desc": "Building-Integrated Thermal Energy Storage for load shifting and peak reduction.",
            "priority": lambda v: "medium" if v > 0.15 else "low"
        },
    ]

    ECONOMIC_RECS = [
        {
            "key": "Loan", "label": "Recommended Loan",
            "icon": "🏦", "unit": "$",
            "range": (0, 10000),
            "desc": "Suggested loan amount for retrofit financing.",
            "priority": lambda v: "medium"
        },
        {
            "key": "Rebate", "label": "Eligible Rebate",
            "icon": "💰", "unit": "$",
            "range": (20000, 50000),
            "desc": "Estimated government rebate available for this retrofit package.",
            "priority": lambda v: "high" if v > 35000 else "medium"
        },
        {
            "key": "IntRate", "label": "Interest Rate",
            "icon": "📈", "unit": "%",
            "range": (0.25, 1.50),
            "desc": "Optimal financing interest rate for this retrofit.",
            "priority": lambda v: "low"
        },
        {
            "key": "Electax", "label": "Electricity Tax Impact",
            "icon": "⚡", "unit": "%",
            "range": (0.0, 10.0),
            "desc": "Electricity tax rate factored into savings calculations.",
            "priority": lambda v: "low"
        },
        {
            "key": "Fueltax", "label": "Fuel Tax Impact",
            "icon": "⛽", "unit": "%",
            "range": (0.0, 10.0),
            "desc": "Fuel/carbon tax rate factored into long-term savings.",
            "priority": lambda v: "medium" if v > 5 else "low"
        },
    ]

    def pct_bar(val, lo, hi):
        pct = max(0, min(100, round((val - lo) / (hi - lo + 1e-9) * 100)))
        return pct

    def format_val(key, val, unit):
        if unit == "$":
            return f"${val:,.0f}"
        elif unit == "%":
            return f"{val:.2f}%"
        elif unit == "ACH":
            return f"{val:.3f} ACH"
        elif unit == "m²K/W":
            return f"{val:.3f} m²K/W"
        elif unit == "":
            return f"{val:.3f}"
        else:
            return f"{val:.3f} {unit}"

    with left:
        st.markdown("#### 🏗️ Building Retrofit Measures")
        for rec in BUILDING_RECS:
            key = rec["key"]
            if key not in best.index:
                continue
            val = float(best[key])
            lo, hi = rec["range"]
            pct = pct_bar(val, lo, hi)
            pri = rec["priority"](val)
            badge_cls = {"high":"badge-high","medium":"badge-med","low":"badge-low"}[pri]
            badge_txt = pri.upper()
            val_str = format_val(key, val, rec["unit"])

            st.markdown(f"""
            <div class="rec-card {pri}">
                <div class="rec-title">
                    {rec['icon']} {rec['label']}
                    <span class="badge {badge_cls}">{badge_txt}</span>
                </div>
                <div class="rec-value">{val_str}</div>
                <div class="bar-track"><div class="bar-fill" style="width:{pct}%"></div></div>
                <div class="rec-desc">{rec['desc']}</div>
            </div>""", unsafe_allow_html=True)

    with right:
        st.markdown("#### 💰 Economic Parameters")
        for rec in ECONOMIC_RECS:
            key = rec["key"]
            if key not in best.index:
                continue
            val = float(best[key])
            lo, hi = rec["range"]
            pct = pct_bar(val, lo, hi)
            pri = rec["priority"](val)
            badge_cls = {"high":"badge-high","medium":"badge-med","low":"badge-low"}[pri]
            badge_txt = pri.upper()
            val_str = format_val(key, val, rec["unit"])

            st.markdown(f"""
            <div class="rec-card {pri}">
                <div class="rec-title">
                    {rec['icon']} {rec['label']}
                    <span class="badge {badge_cls}">{badge_txt}</span>
                </div>
                <div class="rec-value">{val_str}</div>
                <div class="bar-track"><div class="bar-fill" style="width:{pct}%"></div></div>
                <div class="rec-desc">{rec['desc']}</div>
            </div>""", unsafe_allow_html=True)

        # ── Comparison to dataset average ─────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 📊 How This Plan Compares")
        avg_owner = df["CostAnnualSysSave_CAD"].mean()
        avg_ghg   = df["TotalCO2Sav"].mean()
        owner_pct = ((best["CostAnnualSysSave_CAD"] - avg_owner) / avg_owner * 100)
        ghg_pct   = ((best["TotalCO2Sav"] - avg_ghg) / avg_ghg * 100)

        st.metric(
            "Owner Savings vs Average Plan",
            f"${best['CostAnnualSysSave_CAD']:,.0f}",
            f"{owner_pct:+.1f}% vs avg ${avg_owner:,.0f}"
        )
        st.metric(
            "GHG Reduction vs Average Plan",
            f"{best['TotalCO2Sav']/1000:,.1f} tCO₂e",
            f"{ghg_pct:+.1f}% vs avg {avg_ghg/1000:,.1f} tCO₂e"
        )

    st.markdown("---")
    st.info("💡 This plan is based on real building simulation data from the Madadiz research database. "
            "For a detailed on-site energy audit and implementation support, "
            "contact [Madadiz Inc.](https://madadiz.com/contact.html)")
