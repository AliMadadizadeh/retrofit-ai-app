import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(
    page_title="AI Retrofit Planner | Madadiz",
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
available_cities = sorted(DATASET["City"].unique().tolist())
available_ssps   = sorted(DATASET["SSP"].unique().tolist())

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding: 0 !important; max-width: 100% !important;}

    .hero {
        background: #1a1a2e; padding: 3rem 4rem 2.5rem; margin-bottom: 0;
    }
    .hero-tag {
        font-size: 11px; font-weight: 800; letter-spacing: 0.16em;
        text-transform: uppercase; color: #c8922a;
        display: flex; align-items: center; gap: 8px; margin-bottom: 1rem;
    }
    .hero-tag::before { content:''; display:block; width:24px; height:1px; background:#c8922a; }
    .hero h1 { font-size:2.4rem; font-weight:800; color:#f4f0e6; letter-spacing:-0.03em; line-height:1.1; margin-bottom:0.8rem; }
    .hero h1 em { color:#c8922a; font-style:italic; }
    .hero p { font-size:0.95rem; color:rgba(244,240,230,0.6); max-width:560px; line-height:1.7; }

    .step-label {
        font-size:10px; font-weight:800; letter-spacing:0.16em; text-transform:uppercase;
        color:#c8922a; display:flex; align-items:center; gap:6px;
        margin-bottom:0.8rem; margin-top:1.8rem;
    }
    .step-label::after { content:''; flex:1; height:1px; background:#f5e4c0; }

    div[data-testid="stButton"] button[kind="primary"] {
        background: #c8922a !important; color: #1a1a1a !important;
        font-weight: 800 !important; letter-spacing: 0.06em !important;
        text-transform: uppercase !important; border: none !important;
        border-radius: 4px !important; width: 100% !important;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover { background:#e8b85a !important; }

    .empty-state { text-align:center; padding:4rem 2rem; color:#9aaac0; }
    .empty-icon  { font-size:3.5rem; margin-bottom:1rem; opacity:0.35; }
    .empty-title { font-size:1.2rem; font-weight:700; color:#1b2a4a; opacity:0.4; margin-bottom:0.5rem; }
    .empty-sub   { font-size:0.82rem; line-height:1.6; max-width:220px; margin:0 auto; }

    .result-tag   { font-size:11px; font-weight:800; letter-spacing:0.14em; text-transform:uppercase; color:#c8922a; display:flex; align-items:center; gap:6px; margin-bottom:0.5rem; }
    .result-tag::before { content:''; display:block; width:16px; height:1px; background:#c8922a; }
    .result-title { font-size:1.5rem; font-weight:800; color:#1b2a4a; letter-spacing:-0.02em; margin-bottom:1.2rem; }

    .kpi-row { display:flex; gap:10px; margin-bottom:1.5rem; flex-wrap:wrap; }
    .kpi { flex:1; min-width:90px; background:#f0f4f8; border:1px solid #c9d4e0; border-radius:8px; padding:12px 14px; text-align:center; }
    .kpi-n { font-size:1.3rem; font-weight:800; color:#1b2a4a; line-height:1; }
    .kpi-l { font-size:10px; font-weight:700; color:#6b7a9a; margin-top:4px; text-transform:uppercase; letter-spacing:0.06em; }

    .sb-title { font-size:12px; font-weight:800; color:#1b2a4a; text-transform:uppercase; letter-spacing:0.08em; border-bottom:2px solid #1b2a4a; padding-bottom:6px; margin:1.2rem 0 10px; }

    .rec-item { display:flex; align-items:flex-start; gap:12px; padding:12px 14px; border-radius:8px; border:1.5px solid #e8e2d4; background:white; margin-bottom:8px; }
    .rec-icon { font-size:1.3rem; flex-shrink:0; margin-top:2px; }
    .rec-body { flex:1; }
    .rec-name { font-size:0.85rem; font-weight:700; color:#1b2a4a; margin-bottom:2px; }
    .rec-val  { font-size:1rem; font-weight:800; color:#0a0a0a; }
    .rec-desc { font-size:0.75rem; color:#6b7a9a; margin-top:3px; line-height:1.5; }
    .rec-bar-wrap  { width:56px; flex-shrink:0; text-align:right; }
    .rec-bar-track { height:5px; background:#e8e2d4; border-radius:99px; overflow:hidden; margin-bottom:3px; margin-top:6px; }
    .rec-bar-fill  { height:5px; border-radius:99px; background:#1b2a4a; }
    .rec-bar-pct   { font-size:10px; font-weight:700; color:#6b7a9a; }

    .pbadge { display:inline-block; font-size:9px; font-weight:800; letter-spacing:0.08em; text-transform:uppercase; padding:2px 6px; border-radius:3px; margin-left:5px; }
    .p-high { background:#fee2e2; color:#b91c1c; }
    .p-med  { background:#fef3c7; color:#d97706; }
    .p-low  { background:#dcfce7; color:#15803d; }

    .incentive-item { display:flex; gap:10px; align-items:flex-start; padding:10px 12px; background:#f5e4c0; border-radius:6px; margin-bottom:7px; }
    .incentive-icon { font-size:1.1rem; flex-shrink:0; }
    .incentive-name { font-size:0.82rem; font-weight:700; color:#1b2a4a; }
    .incentive-desc { font-size:0.74rem; color:#5c6070; margin-top:2px; line-height:1.4; }

    .roadmap-step { display:flex; gap:14px; align-items:flex-start; padding:12px 0; border-bottom:1px solid #f0ebe3; }
    .roadmap-step:last-child { border-bottom:none; }
    .roadmap-num { width:32px; height:32px; border-radius:50%; background:#1b2a4a; color:white; display:flex; align-items:center; justify-content:center; font-size:12px; font-weight:800; flex-shrink:0; margin-top:2px; }
    .roadmap-phase { font-size:10px; font-weight:800; letter-spacing:0.1em; text-transform:uppercase; color:#c8922a; }
    .roadmap-title { font-size:0.82rem; color:#1b2a4a; margin:2px 0; line-height:1.5; }

    .disclaimer { background:#f5e4c0; border-left:3px solid #c8922a; border-radius:0 6px 6px 0; padding:12px 16px; font-size:0.78rem; line-height:1.6; color:#1a1a2e; margin-top:1.5rem; }
    .disclaimer a { color:#1b2a4a; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-tag">AI-Powered Tool — Madadiz Inc.</div>
  <h1>Get your <em>personalized</em><br>retrofit plan</h1>
  <p>Select your building parameters. Our model searches thousands of real building simulations and returns the best retrofit strategy for your situation — no API, no guessing.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("")

# ── Layout ────────────────────────────────────────────────────────────────────
form_col, result_col = st.columns([1, 1.1], gap="large")

# ══════════════════════════════════════════════════
# LEFT — FORM
# ══════════════════════════════════════════════════
with form_col:

    st.markdown('<div class="step-label">📍 Location & Climate Scenario</div>', unsafe_allow_html=True)
    city = st.selectbox("City", available_cities)

    ssp_map = {
        "SSP126": "SSP126 — Low emissions (optimistic)",
        "SSP245": "SSP245 — Moderate emissions (likely)",
        "SSP585": "SSP585 — High emissions (worst case)",
    }
    ssp_labels  = [ssp_map.get(s, s) for s in available_ssps]
    ssp_choice  = st.selectbox("Climate Scenario", ssp_labels, index=1)
    ssp         = available_ssps[ssp_labels.index(ssp_choice)]

    st.markdown('<div class="step-label">🏗️ Building Details</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    year_built = c1.slider("Year Built", 1920, 2020, 1975, step=5)
    footprint  = c2.number_input("Floor Area (m²)", 30, 2000, 130, 10)

    building_type = st.selectbox("Building Type", [
        "Single-family Home", "Semi-detached / Townhouse",
        "Low-rise Apartment (2–4 storeys)", "Mid-rise (5–11 storeys)",
        "High-rise (12+)", "Commercial / Office", "Institutional"
    ])

    st.markdown('<div class="step-label">⚙️ Current Systems</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    heating    = c3.selectbox("Heating",    ["Gas Furnace","Gas Boiler","Electric Baseboard","Heat Pump","Oil Furnace","District Heating"])
    cooling    = c4.selectbox("Cooling",    ["No Cooling","Central A/C","Mini-split","Window Units"])
    c5, c6     = st.columns(2)
    insulation = c5.selectbox("Insulation", ["Poor (pre-1980s)","Moderate (1980–2005)","Good (post-2005)"])
    windows    = c6.selectbox("Windows",    ["Single-pane","Double-pane","Triple-pane"])

    st.markdown('<div class="step-label">🎯 Your Goals</div>', unsafe_allow_html=True)
    goals = st.multiselect("What matters most?", [
        "💰 Reduce energy bills", "🌿 Lower carbon emissions",
        "🌡 Improve comfort",     "🏠 Increase property value",
        "📋 Regulatory compliance"
    ], default=["💰 Reduce energy bills","🌿 Lower carbon emissions"])

    st.markdown('<div class="step-label">⚖️ Optimization Priority</div>', unsafe_allow_html=True)
    priority = st.radio("Optimize for", [
        "Balanced (recommended)",
        "Maximum owner savings",
        "Maximum GHG reduction",
        "Maximum government savings",
    ])

    w_owner = {"Balanced (recommended)":0.60,"Maximum owner savings":1.0,"Maximum GHG reduction":0.0,"Maximum government savings":0.0}[priority]
    w_gov   = {"Balanced (recommended)":0.20,"Maximum owner savings":0.0,"Maximum GHG reduction":0.0,"Maximum government savings":1.0}[priority]
    w_ghg   = {"Balanced (recommended)":0.20,"Maximum owner savings":0.0,"Maximum GHG reduction":1.0,"Maximum government savings":0.0}[priority]

    st.markdown("")
    run = st.button("▶ Generate My Retrofit Plan", type="primary")

# ══════════════════════════════════════════════════
# RIGHT — RESULTS
# ══════════════════════════════════════════════════
with result_col:

    if not run:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-icon">🏗️</div>
          <div class="empty-title">Your plan will appear here</div>
          <p class="empty-sub">Fill in your building parameters on the left and click Generate to receive your personalized retrofit recommendations.</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        with st.spinner(f"Searching {city} simulation database…"):
            df = DATASET[(DATASET["City"]==city)&(DATASET["SSP"]==ssp)].copy().reset_index(drop=True)
            if df.empty:
                st.error(f"No simulation data found for {city} + {ssp}. Try a different combination.")
                st.stop()

            def norm(s): return (s - s.min()) / (s.max() - s.min() + 1e-9)
            df["Owner_n"] = norm(df["CostAnnualSysSave_CAD"])
            df["Gov_n"]   = norm(df["AnnGovtCostSav_CAD"])
            df["GHG_n"]   = 1 - norm(df["TotalCO2Sav"])
            df["Score"]   = w_owner*df["Owner_n"] + w_gov*df["Gov_n"] + w_ghg*df["GHG_n"]
            best = df.sort_values("Score", ascending=False).iloc[0]

        # Header
        st.markdown(f'<div class="result-tag">Retrofit Plan · {ssp}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-title">{city} — {building_type}</div>', unsafe_allow_html=True)

        # KPIs
        owner_s = best["CostAnnualSysSave_CAD"]
        gov_s   = best["AnnGovtCostSav_CAD"]
        ghg_s   = best["TotalCO2Sav"] / 1000
        pct_s   = best.get("PercentCostSysSav_percent", 0)
        st.markdown(f"""
        <div class="kpi-row">
          <div class="kpi"><div class="kpi-n">${owner_s:,.0f}</div><div class="kpi-l">Owner savings/yr</div></div>
          <div class="kpi"><div class="kpi-n">${gov_s:,.0f}</div><div class="kpi-l">Gov savings/yr</div></div>
          <div class="kpi"><div class="kpi-n">{ghg_s:,.1f}t</div><div class="kpi-l">CO₂e / 20 yrs</div></div>
          <div class="kpi"><div class="kpi-n">{pct_s:.0f}%</div><div class="kpi-l">Energy reduction</div></div>
        </div>""", unsafe_allow_html=True)

        # ── Recommended Measures ──────────────────────────────────────────────
        MEASURES = [
            {"key":"Rvalue_roof","name":"Roof Insulation Upgrade","icon":"🏠","lo":5.46,"hi":11.0,"unit":"m²K/W","desc":"Increase roof R-value to reduce heat loss and cooling loads.","thr":[7.5,9.0]},
            {"key":"Rvalue_wall","name":"Wall Insulation Upgrade","icon":"🧱","lo":3.60,"hi":8.00,"unit":"m²K/W","desc":"Improve wall insulation to reduce thermal bridging and heat transfer.","thr":[5.0,6.5]},
            {"key":"A_PV",       "name":"Rooftop Solar PV",       "icon":"⚡","lo":0.10,"hi":0.60,"unit":"area ratio","desc":"Install PV panels to generate on-site electricity and cut energy costs.","thr":[0.25,0.45]},
            {"key":"Infiltration","name":"Air Sealing",           "icon":"💨","lo":0.50,"hi":1.50,"unit":"ACH","desc":"Seal air leaks to reduce uncontrolled heat loss and improve comfort.","thr":[1.0,1.3]},
            {"key":"Glazing",    "name":"Window Upgrade",         "icon":"🪟","lo":0.10,"hi":0.40,"unit":"ratio","desc":"Optimize glazing ratio and upgrade to high-performance windows.","thr":[0.20,0.30]},
            {"key":"Albedo_roof","name":"Cool Roof Coating",      "icon":"☀️","lo":0.10,"hi":0.70,"unit":"albedo","desc":"High-reflectivity coating reduces solar heat gain through the roof.","thr":[0.35,0.55]},
            {"key":"A_ST",       "name":"Solar Thermal System",   "icon":"🌡️","lo":0.10,"hi":0.60,"unit":"area ratio","desc":"Solar thermal collectors for domestic hot water — reduces gas use.","thr":[0.25,0.40]},
            {"key":"V_bites",    "name":"Thermal Energy Storage", "icon":"🧊","lo":0.05,"hi":0.25,"unit":"ratio","desc":"BITES system for load shifting and peak demand reduction.","thr":[0.12,0.20]},
        ]

        def get_pri(val, thr):
            if val >= thr[1]: return "high","HIGH","p-high"
            if val >= thr[0]: return "medium","MED","p-med"
            return "low","LOW","p-low"

        def fmtv(val, unit):
            if "m²K/W" in unit: return f"{val:.2f} m²K/W"
            if unit=="ACH": return f"{val:.3f} ACH"
            return f"{val:.3f}"

        st.markdown('<div class="sb-title">🏗️ Recommended Retrofit Measures</div>', unsafe_allow_html=True)
        for m in MEASURES:
            if m["key"] not in best.index: continue
            val = float(best[m["key"]])
            pct = max(0,min(100,round((val-m["lo"])/(m["hi"]-m["lo"]+1e-9)*100)))
            _, ptxt, pcls = get_pri(val, m["thr"])
            st.markdown(f"""
            <div class="rec-item">
              <div class="rec-icon">{m['icon']}</div>
              <div class="rec-body">
                <div class="rec-name">{m['name']} <span class="pbadge {pcls}">{ptxt}</span></div>
                <div class="rec-val">{fmtv(val,m['unit'])}</div>
                <div class="rec-desc">{m['desc']}</div>
              </div>
              <div class="rec-bar-wrap">
                <div class="rec-bar-track"><div class="rec-bar-fill" style="width:{pct}%"></div></div>
                <div class="rec-bar-pct">{pct}%</div>
              </div>
            </div>""", unsafe_allow_html=True)

        # ── Incentives ────────────────────────────────────────────────────────
        rebate = float(best.get("Rebate", 35000))
        loan   = float(best.get("Loan",   5000))

        st.markdown('<div class="sb-title">💰 Incentives & Financing</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="incentive-item">
          <div class="incentive-icon">🏛️</div>
          <div><div class="incentive-name">Canada Greener Homes Grant</div>
          <div class="incentive-desc">Up to $5,600 for insulation, windows, doors, heat pumps, and renewable energy systems.</div></div>
        </div>
        <div class="incentive-item">
          <div class="incentive-icon">🏦</div>
          <div><div class="incentive-name">Canada Greener Homes Loan</div>
          <div class="incentive-desc">Interest-free loans up to $40,000 for deep energy retrofits.</div></div>
        </div>
        <div class="incentive-item">
          <div class="incentive-icon">💰</div>
          <div><div class="incentive-name">Estimated Package for This Building</div>
          <div class="incentive-desc">Rebate: <strong>${rebate:,.0f}</strong> · Loan: <strong>${loan:,.0f}</strong> — based on your building profile and the optimal retrofit package from our simulation database.</div></div>
        </div>""", unsafe_allow_html=True)

        # ── Roadmap ───────────────────────────────────────────────────────────
        insulation_bad = "Poor" in insulation or "Moderate" in insulation
        gas_heat       = "Gas" in heating or "Oil" in heating
        single_pane    = windows == "Single-pane"

        p1 = ["Professional energy audit", "Air sealing and draught-proofing"]
        if insulation_bad: p1.append("Attic insulation upgrade")

        p2 = ["HVAC tune-up and maintenance"]
        if gas_heat: p2.append("Heat pump installation (replace gas/oil)")
        if single_pane: p2.append("Window replacement to double or triple-pane")
        if insulation_bad: p2.append("Wall insulation upgrade")

        p3 = ["Rooftop solar PV system", "Solar thermal for hot water", "Building energy management system"]

        st.markdown('<div class="sb-title">🗓️ Implementation Roadmap</div>', unsafe_allow_html=True)
        for i,(phase,timeline,items) in enumerate([
            ("Quick Wins","0–6 months",p1),
            ("Medium Term","6–24 months",p2),
            ("Long Term","2–5 years",p3)
        ]):
            items_html = "".join(f"<li>{it}</li>" for it in items)
            st.markdown(f"""
            <div class="roadmap-step">
              <div class="roadmap-num">{i+1}</div>
              <div>
                <div class="roadmap-phase">{phase} · {timeline}</div>
                <div class="roadmap-title"><ul style="margin:4px 0 0 16px;line-height:1.7">{items_html}</ul></div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="disclaimer">
          <strong>Note:</strong> This plan is generated from real building simulation data and is a starting point.
          For a detailed on-site energy audit and professional implementation support,
          <a href="https://madadiz.com/contact.html">contact Madadiz Inc.</a>
        </div>""", unsafe_allow_html=True)