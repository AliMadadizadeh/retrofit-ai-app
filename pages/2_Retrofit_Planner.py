import streamlit as st
import anthropic

st.set_page_config(
    page_title="Retrofit Planner | Madadiz",
    page_icon="🏗️",
    layout="wide"
)

st.markdown("## 🏗️ AI Retrofit Planner")
st.caption("Fill in your building details to get a personalized retrofit recommendation.")
st.markdown("---")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("#### Building Information")
    building_type = st.selectbox("Building Type", [
        "Single-family Home", "Semi-detached / Townhouse",
        "Low-rise Apartment (2–4 storeys)", "Mid-rise Apartment (5–11 storeys)",
        "High-rise (12+ storeys)", "Commercial / Office", "Institutional"
    ])
    province = st.selectbox("Province / Territory", [
        "Ontario", "Quebec", "British Columbia", "Alberta",
        "Manitoba", "Saskatchewan", "Nova Scotia", "New Brunswick",
        "Newfoundland & Labrador", "Northwest Territories", "Yukon", "Nunavut"
    ])
    year_built = st.slider("Year Built", 1900, 2020, 1975)
    floor_area = st.number_input("Floor Area (m²)", min_value=50, max_value=5000, value=200, step=10)
    storeys    = st.selectbox("Number of Storeys", ["1", "2", "3", "4–6", "7–12", "12+"])

with col2:
    st.markdown("#### Current Systems & Envelope")
    heating    = st.selectbox("Heating System", [
        "Gas Furnace", "Gas Boiler", "Electric Baseboard",
        "Heat Pump", "Oil Furnace", "District Heating"
    ])
    cooling    = st.selectbox("Cooling System", [
        "Central A/C", "Mini-split", "Window Units", "No Cooling"
    ])
    insulation = st.selectbox("Insulation Level", [
        "Poor (pre-1980s)", "Moderate (1980–2005)", "Good (post-2005)"
    ])
    windows    = st.selectbox("Window Type", [
        "Single-pane", "Double-pane", "Triple-pane"
    ])
    ventilation = st.selectbox("Ventilation", [
        "Natural only", "Basic mechanical", "HRV/ERV system"
    ])

st.markdown("---")
col3, col4 = st.columns(2, gap="large")

with col3:
    st.markdown("#### Goals")
    goals = st.multiselect("Primary Goals (select all that apply)", [
        "Reduce energy bills",
        "Lower carbon emissions",
        "Improve comfort",
        "Increase property value",
        "Comply with regulations / net-zero targets"
    ], default=["Reduce energy bills", "Lower carbon emissions"])

with col4:
    st.markdown("#### Budget & Ownership")
    budget = st.selectbox("Budget Range (CAD)", [
        "Under $10,000", "$10,000–$30,000", "$30,000–$75,000",
        "$75,000–$200,000", "$200,000+", "Flexible / Unknown"
    ])
    ownership = st.selectbox("Ownership", [
        "Owner-occupied", "Landlord / Rental", "Government / Non-profit"
    ])
    notes = st.text_area("Additional Notes (optional)",
                         placeholder="Known issues, previous upgrades, specific concerns…",
                         height=80)

st.markdown("---")

if st.button("▶ Generate Retrofit Plan", type="primary", use_container_width=True):
    prompt = f"""You are a senior building energy engineer at Madadiz Inc., a Canadian sustainability consulting firm. Generate a detailed, practical retrofit plan based on these parameters:

BUILDING PARAMETERS:
- Type: {building_type}
- Province: {province}
- Year Built: {year_built}
- Floor Area: {floor_area} m²
- Storeys: {storeys}
- Heating: {heating}
- Cooling: {cooling}
- Insulation: {insulation}
- Windows: {windows}
- Ventilation: {ventilation}
- Goals: {', '.join(goals) if goals else 'Not specified'}
- Budget: {budget}
- Ownership: {ownership}
- Notes: {notes or 'None'}

Generate a comprehensive retrofit plan with these sections:

### Building Assessment Summary
2-3 sentences on current energy performance.

### Recommended Retrofit Measures
4-6 specific measures in priority order. For each: what to do, why, estimated energy savings, rough CAD cost, and priority (HIGH/MEDIUM/LOW).

### Estimated Overall Impact
Ranges for: annual energy cost savings, GHG reduction, simple payback period.

### Available Incentives
2-4 relevant Canadian federal or provincial programs for this building and province.

### Implementation Roadmap
3 phases: Quick wins (0-6 months), Medium-term (6-24 months), Long-term (2-5 years).

### Next Steps
2-3 concrete actions including recommending a professional energy audit from Madadiz Inc.

Be specific to Canadian conditions and climate. Use plain language accessible to non-engineers."""

    with st.spinner("Generating your personalized retrofit plan…"):
        try:
            client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )
            result = message.content[0].text

            st.success("✅ Your retrofit plan is ready!")
            st.markdown("---")

            # Display formatted result
            for line in result.split('\n'):
                if line.startswith('### '):
                    st.markdown(f"#### {line[4:]}")
                elif line.startswith('- HIGH'):
                    st.markdown(f"🔴 {line[2:]}")
                elif line.startswith('- MEDIUM'):
                    st.markdown(f"🟡 {line[2:]}")
                elif line.startswith('- LOW'):
                    st.markdown(f"🟢 {line[2:]}")
                else:
                    st.markdown(line)

            st.info("💡 This plan is AI-generated based on your inputs. For a detailed engineering assessment and energy modeling, contact [Madadiz Inc.](https://madadiz.com/contact.html) for a professional consultation.")

        except Exception as e:
            st.error(f"Something went wrong: {e}")