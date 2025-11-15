# app.py — Ballistics Explorer (Home)
import os
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Ballistics Toolkit", layout="wide")

# ------------------------------
# Data loading & normalization
# ------------------------------

@st.cache_data
def load_ballistics():
    # If you want to use this somewhere later
    return pd.read_excel("Ballistics.xlsx")

@st.cache_data
@st.cache_data
def load_default_hit_prob():
    # Try several possible locations / formats
    candidates = [
        os.path.join("Data", "hit_probability_semi_realistic.xlsx"),
        os.path.join("Data", "hit_probability_semi_realistic.csv"),
        "hit_probability_semi_realistic.xlsx",
        "hit_probability_semi_realistic.csv",
    ]

    for path in candidates:
        if os.path.exists(path):
            if path.lower().endswith(".csv"):
                return pd.read_csv(path)
            else:
                return pd.read_excel(path)

    # If we get here, nothing was found
    st.error(
        "No built-in hit probability file found. "
        "Expected one of: hit_probability_semi_realistic.[xlsx/csv] "
        "in the current folder or Data/."
    )
    return pd.DataFrame()


# Hit probability: auto-load built-in, allow optional override upload
uploaded_hit = st.file_uploader(
    "Upload a different hit probability file (optional)",
    type=["xlsx", "csv"]
)

if uploaded_hit is not None:
    # User uploaded a file → use that
    if uploaded_hit.name.lower().endswith(".csv"):
        hit_prob_df = pd.read_csv(uploaded_hit)
    else:
        hit_prob_df = pd.read_excel(uploaded_hit)
    hit_source = f"Using uploaded hit probability file: {uploaded_hit.name}"
else:
    # No upload → use the built-in file automatically
    hit_prob_df = load_default_hit_prob()
    hit_source = "Using built-in hit_probability_semi_realistic.xlsx."

st.caption(hit_source)

@st.cache_data
def load_data():
    # Prefer CSV first (no Excel dep), then Excel — from ./Data/
    candidates = [
        ("CSV", os.path.join("Data", "ballistics_cartridge_performance_v4.csv")),
        ("Excel", os.path.join("Data", "Ballistics.xlsx")),
    ]
    found = None
    for label, fname in candidates:
        if os.path.exists(fname):
            found = (label, fname)
            break

    if found is None:
        st.warning("No local data found. Upload a CSV/Excel in the sidebar or place a file in ./Data/.")
        return pd.DataFrame(), "No local data"

    label, fname = found
    if label == "Excel":
        try:
            df = pd.read_excel(fname, engine="openpyxl")
        except Exception:
            df = pd.read_excel(fname)  # fallback if engine not pinned
        source_note = f"Loaded: {os.path.basename(fname)} (Excel)"
    else:
        df = pd.read_csv(fname)
        source_note = f"Loaded: {os.path.basename(fname)} (CSV)"

    # Normalize common schemas
    rename_map = {
        "Cartridge": "cartridge",
        "Bullet_Weight": "bullet_weight_gr",
        "B.C": "ballistic_coefficient_G1",
        "Range": "distance_yd",
        "Velocity": "retained_velocity_fps",
        "Energy": "retained_energy_ftlb",
        "TimeOfFlight": "time_of_flight_s",
        "ComeUp": "drop_from_100yd_zero_in",
        "WindDrift": "wind_drift_10mph_in",
    }
    for k, v in rename_map.items():
        if k in df.columns:
            df = df.rename(columns={k: v})

    # Compute MIL/MOA if possible
    if {"drop_from_100yd_zero_in", "distance_yd"}.issubset(df.columns):
        dist_hund = pd.to_numeric(df["distance_yd"], errors="coerce") / 100.0
        df["drop_moa"] = pd.to_numeric(df["drop_from_100yd_zero_in"], errors="coerce") / (
            (1.047 * dist_hund).replace(0, pd.NA)
        )
        df["drop_mil"] = pd.to_numeric(df["drop_from_100yd_zero_in"], errors="coerce") / (
            (3.6 * dist_hund).replace(0, pd.NA)
        )

    # Coerce numerics commonly used
    num_cols = [
        "retained_velocity_fps", "retained_energy_ftlb", "time_of_flight_s",
        "drop_from_100yd_zero_in", "wind_drift_10mph_in", "drop_moa", "drop_mil",
        "distance_yd", "bullet_weight_gr", "ballistic_coefficient_G1"
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "cartridge" in df.columns:
        df["cartridge"] = df["cartridge"].astype(str)

    return df, source_note


def read_uploaded(file):
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    elif name.endswith((".xlsx", ".xls")):
        try:
            return pd.read_excel(file, engine="openpyxl")
        except Exception:
            return pd.read_excel(file)
    else:
        st.error("Unsupported file type. Please upload .csv or .xlsx")
        return pd.DataFrame()

# ------------------------------
# Header / Branding
# ------------------------------
st.title("🔭 Ballistics Explorer")
st.caption("by Josh Lapierre • jlapier1@msudenver.edu")
df_local, source_note = load_data()
st.caption(source_note)

# Sidebar upload — optional override
st.sidebar.subheader("📤 Upload data (optional)")
upl = st.sidebar.file_uploader("CSV or Excel", type=["csv","xlsx","xls"], accept_multiple_files=False)
if upl is not None:
    df_upl = read_uploaded(upl)
    if not df_upl.empty:
        st.sidebar.success(f"Using uploaded file: {upl.name}")
        df = df_upl
        # Apply same normalization as load_data()
        rename_map = {
            "Cartridge": "cartridge",
            "Bullet_Weight": "bullet_weight_gr",
            "B.C": "ballistic_coefficient_G1",
            "Range": "distance_yd",
            "Velocity": "retained_velocity_fps",
            "Energy": "retained_energy_ftlb",
            "TimeOfFlight": "time_of_flight_s",
            "ComeUp": "drop_from_100yd_zero_in",
            "WindDrift": "wind_drift_10mph_in",
        }
        for k, v in rename_map.items():
            if k in df.columns:
                df = df.rename(columns={k: v})
        if {"drop_from_100yd_zero_in", "distance_yd"}.issubset(df.columns):
            dist_hund = pd.to_numeric(df["distance_yd"], errors="coerce") / 100.0
            df["drop_moa"] = pd.to_numeric(df["drop_from_100yd_zero_in"], errors="coerce") / ((1.047 * dist_hund).replace(0, pd.NA))
            df["drop_mil"] = pd.to_numeric(df["drop_from_100yd_zero_in"], errors="coerce") / ((3.6 * dist_hund).replace(0, pd.NA))
    else:
        df = df_local.copy()
else:
    df = df_local.copy()

if df is None or df.empty:
    st.info("Load or upload a dataset to begin.")
    st.stop()

cartridges = sorted(df["cartridge"].dropna().unique().tolist()) if "cartridge" in df.columns else []
if not cartridges:
    st.error("No 'cartridge' values found. Check your upload or header row.")
    st.stop()

min_d = int(df["distance_yd"].min()) if "distance_yd" in df.columns else 0
max_d = int(df["distance_yd"].max()) if "distance_yd" in df.columns else 0

# Helper for charts
def line_chart(data, x, y, title, y_title):
    chart = alt.Chart(data).mark_line(point=True).encode(
        x=alt.X(x, title="Distance (yd)"),
        y=alt.Y(y, title=y_title),
        tooltip=[alt.Tooltip("distance_yd", title="Distance (yd)"),
                 alt.Tooltip(y, title=y_title)]
    ).properties(title=title, height=320)
    st.altair_chart(chart, use_container_width=True)

# ------------------------------
# Tabs (ALL content stays inside)
# ------------------------------
tab1, tab2 = st.tabs(["🎯 Single Caliber", "🆚 Compare Calibers"])

# ---- SINGLE CALIBER ----
with tab1:
    st.sidebar.header("Single Caliber Controls")
    cart = st.sidebar.selectbox("Caliber", options=cartridges, index=0, key="single_cart")
    df_s = df[df["cartridge"] == cart].copy()

    # Bullet weight filter (if available)
    w = None
    if "bullet_weight_gr" in df_s.columns and not df_s.empty:
        weights = sorted(df_s["bullet_weight_gr"].dropna().unique().tolist())
        w = st.sidebar.selectbox("Bullet weight (gr)", options=weights, index=0, key="single_weight_selector")
        df_s = df_s[df_s["bullet_weight_gr"] == w]

    # Distance filter
    rng = st.sidebar.slider("Distance range (yd)",
                            min_value=min_d, max_value=max_d,
                            value=(min_d, max_d),
                            step=50 if (max_d - min_d) >= 50 else 1,
                            key="single_range")
    df_s = df_s[(df_s["distance_yd"] >= rng[0]) & (df_s["distance_yd"] <= rng[1])].sort_values("distance_yd")

    # Header
    title = f"{cart} — {w} gr" if w is not None else cart
    fname_suffix = f"{cart}_{w}" if w is not None else cart
    st.subheader(title)

    # Info metrics
    info_cols = st.columns(4)
    if "ballistic_coefficient_G1" in df_s.columns and not df_s.empty:
        bc_vals = df_s["ballistic_coefficient_G1"].dropna().unique()
        info_cols[0].metric("BC (G1)", f"{bc_vals[0]:.3f}" if len(bc_vals) > 0 else "—")
    if "retained_velocity_fps" in df_s.columns and not df_s.empty:
        info_cols[1].metric("Start Velocity (fps)", f"{df_s['retained_velocity_fps'].iloc[0]:.0f}")
    if "retained_energy_ftlb" in df_s.columns and not df_s.empty:
        info_cols[2].metric("Start Energy (ft·lb)", f"{df_s['retained_energy_ftlb'].iloc[0]:.0f}")
    if "time_of_flight_s" in df_s.columns and not df_s.empty:
        info_cols[3].metric("Max TOF (s)", f"{df_s['time_of_flight_s'].max():.3f}")

    # Charts
    c1, c2 = st.columns(2)
    if "drop_from_100yd_zero_in" in df_s.columns:
        line_chart(df_s, "distance_yd:Q", "drop_from_100yd_zero_in:Q",
                   "Trajectory (Drop from 100 yd Zero)", "Drop (in)")
    if "retained_velocity_fps" in df_s.columns:
        line_chart(df_s, "distance_yd:Q", "retained_velocity_fps:Q",
                   "Velocity vs Distance", "Velocity (fps)")

    c3, c4 = st.columns(2)
    if "retained_energy_ftlb" in df_s.columns:
        line_chart(df_s, "distance_yd:Q", "retained_energy_ftlb:Q",
                   "Energy vs Distance", "Energy (ft·lb)")
    if "wind_drift_10mph_in" in df_s.columns:
        line_chart(df_s, "distance_yd:Q", "wind_drift_10mph_in:Q",
                   "Wind Drift (10 mph full value)", "Drift (in)")

    with st.expander("📋 DOPE Table"):
        cols_to_show = ["distance_yd","drop_from_100yd_zero_in","drop_moa","drop_mil",
                        "retained_velocity_fps","retained_energy_ftlb","wind_drift_10mph_in","time_of_flight_s"]
        present = [c for c in cols_to_show if c in df_s.columns]
        st.dataframe(df_s[present].round(3), use_container_width=True)

    st.download_button("Download filtered (CSV)",
                       data=df_s.to_csv(index=False),
                       file_name=f"{fname_suffix}_filtered.csv",
                       mime="text/csv")

# ---- COMPARE CALIBERS ----
with tab2:
    st.sidebar.header("Compare Calibers Controls")
    sel = st.sidebar.multiselect("Select calibers", options=cartridges,
                                 default=cartridges[:2] if len(cartridges) >= 2 else cartridges,
                                 key="cmp_carts")

    df_c = df[df["cartridge"].isin(sel)].copy()

    rng2 = st.sidebar.slider("Distance range (yd)",
                             min_value=min_d, max_value=max_d,
                             value=(min_d, max_d),
                             step=50 if (max_d - min_d) >= 50 else 1,
                             key="cmp_range")
    df_c = df_c[(df_c["distance_yd"] >= rng2[0]) & (df_c["distance_yd"] <= rng2[1])].sort_values(["cartridge","distance_yd"])

    def line_chart_multi(data, y, title, y_title):
        chart = (
            alt.Chart(data)
            .mark_line(point=True)
            .encode(
                x=alt.X("distance_yd:Q", title="Distance (yd)"),
                y=alt.Y(f"{y}:Q", title=y_title),
                color=alt.Color("cartridge:N", legend=alt.Legend(title="Caliber")),
                tooltip=["cartridge","distance_yd", y] + (["bullet_weight_gr"] if "bullet_weight_gr" in data.columns else [])
            )
            .properties(title=title, height=320)
        )
        st.altair_chart(chart, use_container_width=True)

    st.subheader("🆚 Caliber Comparison")
    c1, c2 = st.columns(2)
    if "drop_from_100yd_zero_in" in df_c.columns:
        line_chart_multi(df_c, "drop_from_100yd_zero_in", "Trajectory (Drop from 100 yd Zero)", "Drop (in)")
    if "retained_velocity_fps" in df_c.columns:
        line_chart_multi(df_c, "retained_velocity_fps", "Velocity vs Distance", "Velocity (fps)")

    c3, c4 = st.columns(2)
    if "retained_energy_ftlb" in df_c.columns:
        line_chart_multi(df_c, "retained_energy_ftlb", "Energy vs Distance", "Energy (ft·lb)")
    if "wind_drift_10mph_in" in df_c.columns:
        line_chart_multi(df_c, "wind_drift_10mph_in", "Wind Drift (10 mph full value)", "Drift (in)")

    with st.expander("📋 Comparison Table"):
        cols = ["cartridge","bullet_weight_gr","distance_yd","drop_from_100yd_zero_in","drop_moa",
                "drop_mil","retained_velocity_fps","retained_energy_ftlb","wind_drift_10mph_in","time_of_flight_s"]
        present = [c for c in cols if c in df_c.columns]
        st.dataframe(df_c[present].round(3), use_container_width=True)

    st.download_button("Download comparison (CSV)",
                       data=df_c.to_csv(index=False),
                       file_name="comparison_filtered.csv",
                       mime="text/csv")

st.markdown("---")
st.caption("Drop values are relative to a 100 yd zero (positive = low). 1 MOA ≈ 1.047\" per 100 yd; 1 MIL ≈ 3.6\" per 100 yd.")
