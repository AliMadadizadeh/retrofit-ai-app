import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------
# LOAD MODEL
# -------------------------
model = joblib.load("retrofit_forward_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("AI Retrofit Decision Tool")

# -------------------------
# USER INPUT
# -------------------------
city = st.selectbox("City", [
    "Toronto", "Vancouver", "Montreal", "Calgary", "StJohns",
    "Halifax", "Winnipeg", "Saskatoon",
    "Whitehorse", "Yellowknife"
])

footprint = st.number_input("Building footprint (m²)", value=130)

ssp = st.selectbox("SSP Scenario", ["SSP126", "SSP245", "SSP585"])

elec_inf = st.number_input("Electricity inflation", value=0.01)
fuel_inf = st.number_input("Fuel inflation", value=0.05)

st.subheader("Weights")
w_owner = st.number_input("Owner saving weight", value=0.6)
w_gov   = st.number_input("Government saving weight", value=0.2)
w_ghg   = st.number_input("GHG saving weight", value=0.2)

# -------------------------
# OPTIMIZATION BUTTON
# -------------------------
if st.button("Find Best Retrofit"):

    np.random.seed(42)
    n_samples = 2000

    ranges = {
        "V_bites": (0.05, 0.25),
        "Albedo_roof": (0.1, 0.7),
        "A_ST": (0.1, 0.6),
        "Rvalue_roof": (5.46, 11.0),
        "Loan": (0, 35000),
        "Rebate": (0, 35000),
        "Rvalue_wall": (3.6, 8.0),
        "Glazing": (0.2, 0.4),
        "IntRate": (0.75, 5.0),
        "Infiltration": (0.5, 1.5),
        "Electax": (0.0, 4.0),
        "SHGC": (0.1, 0.7),
        "Fueltax": (0.0, 8.0),
        "A_PV": (0.1, 0.6)
    }

    rows = []
    for _ in range(n_samples):
        row = {
            "City": city,
            "BuildingFootprintArea_m2": footprint,
            "SSP": ssp,
            "ElectricityInflationRate": elec_inf,
            "FuelInflationRate": fuel_inf
        }
        for k, (lo, hi) in ranges.items():
            row[k] = np.random.uniform(lo, hi)
        rows.append(row)

    df = pd.DataFrame(rows)

    # -------------------------
    # PREDICTION
    # -------------------------
    X = pd.get_dummies(df, columns=["City", "SSP"])
    X = X.reindex(columns=model_columns, fill_value=0)

    pred = model.predict(X)

    df["GHG"] = pred[:, 3]
    df["Owner"] = pred[:, 2]
    df["Gov"] = pred[:, 7]

    # normalize
    df["GHG_n"] = (df["GHG"] - df["GHG"].min()) / (df["GHG"].max() - df["GHG"].min())
    df["Owner_n"] = (df["Owner"] - df["Owner"].min()) / (df["Owner"].max() - df["Owner"].min())
    df["Gov_n"] = (df["Gov"] - df["Gov"].min()) / (df["Gov"].max() - df["Gov"].min())

    df["Score"] = w_owner*df["Owner_n"] + w_gov*df["Gov_n"] + w_ghg*df["GHG_n"]

    best = df.loc[df["Score"].idxmax()]

    st.success("Best retrofit found!")

    st.subheader("Best Retrofit Variables")
    st.write(best)
