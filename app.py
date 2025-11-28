import streamlit as st
from model import HitProbabilityModel

@st.cache_resource
def load_hit_model():
    return HitProbabilityModel()

hit_model = load_hit_model()

st.header("ML Hit Probability (Semi-Realistic Data)")

col1, col2 = st.columns(2)

with col1:
    range_yd = st.number_input("Range (yd)", 100, 1500, 600, 25)
    mv_fps = st.number_input("Muzzle velocity (fps)", 1500, 4000, 2800, 25)
    mv_sd = st.number_input("Muzzle velocity SD (fps)", 0, 100, 10, 1)
    group_moa = st.number_input("Shooter/rifle group size (MOA)", 0.1, 5.0, 1.0, 0.1)
    target_size_moa = st.number_input("Target size (MOA)", 0.5, 10.0, 2.0, 0.1)

with col2:
    wind_mph = st.number_input("Crosswind speed (mph)", 0, 40, 10, 1)
    DA_ft = st.number_input("Density altitude (ft)", -2000, 15000, 2500, 100)
    temp_F = st.number_input("Temperature (°F)", -20, 120, 70, 1)
    pressure_inHg = st.number_input("Pressure (inHg)", 25.00, 32.00, 29.92, 0.01)
    rh_pct = st.number_input("Relative humidity (%)", 0, 100, 40, 1)

if st.button("Estimate hit probability"):
    features = {
        "range_yd": range_yd,
        "mv_fps": mv_fps,
        "mv_sd": mv_sd,
        "wind_mph": wind_mph,
        "DA_ft": DA_ft,
        "temp_F": temp_F,
        "pressure_inHg": pressure_inHg,
        "rh_pct": rh_pct,
        "group_moa": group_moa,
        "target_size_moa": target_size_moa,
    }

    prob = hit_model.predict_proba_single(features)

    st.metric("Estimated hit probability", f"{prob * 100:.1f}%")
    st.progress(min(max(prob, 0.0), 1.0))

    st.caption(
        "Predicted probability of hit given your inputs, using a logistic regression "
        "model trained on semi-realistic simulated shooting data."
    )

