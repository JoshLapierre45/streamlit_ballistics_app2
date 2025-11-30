import streamlit as st
from model import HitProbabilityModel
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import altair as alt   # ✅ NEW

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

def simulate_group_from_prob(
    hit_prob: float,
    target_radius_moa: float,
    n_shots: int = 5,
    miss_radius_factor: float = 3.0,
):
    """
    Simulate n_shots impact points given a hit probability.
    Returns list of dicts: {"x": float, "y": float, "hit": bool}
    Coordinates are in MOA relative to point of aim.
    """

    shots = []
    hit_prob = float(np.clip(hit_prob, 0.0, 1.0))

    for _ in range(n_shots):
        is_hit = np.random.rand() < hit_prob

        # Random angle
        theta = 2 * np.pi * np.random.rand()

        if is_hit:
            # Hits: uniformly inside target radius
            r = target_radius_moa * np.sqrt(np.random.rand())
        else:
            # Misses: uniformly outside target
            r_inner = target_radius_moa
            r_outer = target_radius_moa * miss_radius_factor
            r = np.sqrt((r_outer**2 - r_inner**2) * np.random.rand() + r_inner**2)

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        shots.append({"x": x, "y": y, "hit": is_hit})

    return shots

def plot_group(shots, target_radius_moa, title="Simulated 5-shot group"):
    fig, ax = plt.subplots(figsize=(4, 4))

    # Target (circle)
    circle = plt.Circle((0, 0), target_radius_moa, fill=False, linewidth=2)
    ax.add_patch(circle)

    # Plot shots
    for s in shots:
        color = "tab:green" if s["hit"] else "tab:red"
        ax.scatter(s["x"], s["y"], c=color, s=60, edgecolors="black", zorder=3)

    ax.set_aspect("equal")
    margin = target_radius_moa * 3.2
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.set_xlabel("Horizontal Offset (MOA)")
    ax.set_ylabel("Vertical Offset (MOA)")
    ax.set_title(title)

    st.pyplot(fig)

if st.button("Estimate hit probability"):
    # Base feature dict for current user inputs
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

    # --- Single prediction for current range ---
    prob = hit_model.predict_proba_single(features)

    st.metric("Estimated hit probability", f"{prob * 100:.1f}%")
    st.progress(min(max(prob, 0.0), 1.0))

    st.caption(
        "Predicted probability of hit given your inputs, using a logistic regression "
        "model trained on semi-realistic simulated shooting data."
    )

    # -----------------------------
    # Simulated 5-shot group
    # -----------------------------
    st.subheader("Simulated 5-shot Group Visualization")

    target_radius_moa = target_size_moa / 2.0

    shots = simulate_group_from_prob(
        hit_prob=prob,
        target_radius_moa=target_radius_moa,
        n_shots=5,
        miss_radius_factor=3.0,
    )

    plot_group(
        shots,
        target_radius_moa,
        title=f"Simulated Group (Hit Prob = {prob*100:.1f}%)"
    )

    # -----------------------------
    # NEW: Probability vs Range curve
    # -----------------------------
    st.subheader("Hit Probability vs Range")

    ranges = list(range(100, 1501, 50))
    probs_curve = []

    for r in ranges:
        f_r = dict(features)          # copy base features
        f_r["range_yd"] = r           # vary only range
        p_r = hit_model.predict_proba_single(f_r)
        probs_curve.append(p_r)

    df = pd.DataFrame({
        "Range (yd)": ranges,
        "Hit Probability": probs_curve,
    })

    chart = alt.Chart(df).mark_line().encode(
        x="Range (yd)",
        y=alt.Y("Hit Probability", axis=alt.Axis(format=".0%"))
    )

    st.altair_chart(chart, use_container_width=True)
