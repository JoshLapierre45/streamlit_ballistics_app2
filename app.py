import streamlit as st
from model import HitProbabilityModel
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import altair as alt


# =========================
# Load lightweight model
# =========================
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
    group_moa = st.number_input(
        "Shooter/rifle group size (MOA)", 0.1, 5.0, 1.0, 0.1
    )
    target_size_moa = st.number_input("Target size (MOA)", 0.5, 10.0, 2.0, 0.1)

with col2:
    wind_mph = st.number_input("Crosswind speed (mph)", 0, 40, 10, 1)
    DA_ft = st.number_input("Density altitude (ft)", -2000, 15000, 2500, 100)
    temp_F = st.number_input("Temperature (°F)", -20, 120, 70, 1)
    pressure_inHg = st.number_input("Pressure (inHg)", 25.00, 32.00, 29.92, 0.01)
    rh_pct = st.number_input("Relative humidity (%)", 0, 100, 40, 1)


# =========================
# Group simulation helpers
# =========================
def simulate_group_from_prob(
    hit_prob: float,
    target_radius_moa: float,
    n_shots: int = 5,
    miss_radius_factor: float = 3.0,
):
    """
    Simulate n_shots impact points given a hit probability.
    Returns list[dict]: {"x": float, "y": float, "hit": bool}
    Coordinates are in MOA relative to point of aim.
    """
    shots = []
    hit_prob = float(np.clip(hit_prob, 0.0, 1.0))

    # Slightly scale miss radius with (1 - hit_prob) so very low probs look wilder
    miss_factor = miss_radius_factor * (0.6 + 0.4 * (1.0 - hit_prob))

    for _ in range(n_shots):
        is_hit = np.random.rand() < hit_prob

        theta = 2 * np.pi * np.random.rand()  # random angle

        if is_hit:
            # Hits: uniformly inside target disk
            r = target_radius_moa * np.sqrt(np.random.rand())
        else:
            # Misses: outside target, up to miss_factor * radius
            r_inner = target_radius_moa
            r_outer = target_radius_moa * miss_factor
            r = np.sqrt((r_outer**2 - r_inner**2) * np.random.rand() + r_inner**2)

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        shots.append({"x": x, "y": y, "hit": is_hit})

    return shots


def plot_group(shots, target_radius_moa, title="Simulated 5-shot group"):
    """
    Nicely formatted group plot:
    - Compact figure
    - Dynamic, symmetric MOA axes
    - Equal aspect so target looks circular
    """
    if not shots:
        st.info("No shots to display.")
        return

    # Extract coordinates from list of dicts
    xs = np.array([s["x"] for s in shots], dtype=float)
    ys = np.array([s["y"] for s in shots], dtype=float)

    # How far the shots actually landed
    radial = np.sqrt(xs**2 + ys**2)
    max_spread = float(radial.max()) if radial.size > 0 else 0.0

    # Choose a sensible symmetric limit
    base_lim = max(target_radius_moa * 1.5, max_spread * 1.2, 2.0)
    lim = round(base_lim + 0.5, 1)

    fig, ax = plt.subplots(figsize=(4, 4))

    # Target circle
    circle = plt.Circle(
        (0, 0),
        target_radius_moa,
        edgecolor="black",
        facecolor="none",
        linewidth=2,
    )
    ax.add_patch(circle)

    # Color shots by hit / miss using the original dicts
    colors = ["tab:green" if s["hit"] else "tab:red" for s in shots]

    ax.scatter(
        xs,
        ys,
        s=60,
        c=colors,
        edgecolor="k",
        zorder=3,
    )

    # Crosshair
    ax.axhline(0, color="0.85", linewidth=1, zorder=1)
    ax.axvline(0, color="0.85", linewidth=1, zorder=1)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", "box")

    ax.set_xlabel("Horizontal Offset (MOA)")
    ax.set_ylabel("Vertical Offset (MOA)")
    ax.set_title(title, fontsize=12)

    ax.set_xticks(np.linspace(-lim, lim, 5))
    ax.set_yticks(np.linspace(-lim, lim, 5))

    ax.grid(False)
    fig.tight_layout()

    st.pyplot(fig, use_container_width=False)


# =========================
# Main interaction
# =========================
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
        "Predicted probability of hit given your inputs, using a calibrated logistic "
        "regression model trained on semi-realistic simulated shooting data."
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
        title=f"Simulated Group (Hit Prob = {prob*100:.1f}%)",
    )

    # -----------------------------
    # Probability vs Range curve
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

    chart = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("Range (yd)", title="Range (yd)"),
            y=alt.Y(
                "Hit Probability",
                axis=alt.Axis(format=".0%", title="Hit Probability"),
            ),
        )
        .properties(height=250)
    )

    st.altair_chart(chart, use_container_width=True)
