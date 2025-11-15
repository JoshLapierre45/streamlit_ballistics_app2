
import os
import math
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Ballistics Explorer", layout="wide")

# ---------------------------------
# Data loading & normalization
# ---------------------------------
@st.cache_data
def load_data():
    # Prefer CSV first (no Excel dependency), then Excel
    candidates = [
        ("CSV", "ballistics_cartridge_performance_v4.csv"),
        ("Excel", "Ballistics.xlsx"),
    ]
    found = None
    for label, fname in candidates:
        if os.path.exists(fname):
            found = (label, fname)
            break

    if found is None:
        st.warning("No local data found. You can upload a CSV/Excel in the sidebar.")
        return pd.DataFrame(), "No local data"

    label, fname = found
    if label == "Excel":
        try:
            df = pd.read_excel(fname, engine="openpyxl")
        except Exception:
            df = pd.read_excel(fname)  # fallback if engine available
        # Normalize columns from the Excel schema
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
        source_note = f"Loaded: {fname} (Excel)"
    else:
        df = pd.read_csv(fname)
        source_note = f"Loaded: {fname} (CSV)"

    # MIL/MOA if available
    if "drop_from_100yd_zero_in" in df.columns and "distance_yd" in df.columns:
        dist_hund = pd.to_numeric(df["distance_yd"], errors="coerce") / 100.0
        denom_moa = (1.047 * dist_hund).replace(0, pd.NA)
        denom_mil = (3.6 * dist_hund).replace(0, pd.NA)
        df["drop_moa"] = pd.to_numeric(df["drop_from_100yd_zero_in"], errors="coerce") / denom_moa
        df["drop_mil"] = pd.to_numeric(df["drop_from_100yd_zero_in"], errors="coerce") / denom_mil

    # Types
    for col in ["retained_velocity_fps","retained_energy_ftlb","time_of_flight_s","drop_from_100yd_zero_in","wind_drift_10mph_in","drop_moa","drop_mil","distance_yd","bullet_weight_gr","ballistic_coefficient_G1"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "cartridge" in df.columns:
        df["cartridge"] = df["cartridge"].astype(str)

    return df, source_note

def read_uploaded(file):
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        try:
            return pd.read_excel(file, engine="openpyxl")
        except Exception:
            return pd.read_excel(file)
    else:
        st.error("Unsupported file type. Please upload .csv or .xlsx")
        return pd.DataFrame()

df_local, source_note = load_data()

st.title("🔭 Ballistics Explorer")
st.caption(source_note)

# Sidebar upload
st.sidebar.subheader("📤 Upload data (optional)")
upl = st.sidebar.file_uploader("CSV or Excel", type=["csv","xlsx","xls"], accept_multiple_files=False)
if upl is not None:
    df_upl = read_uploaded(upl)
    if not df_upl.empty:
        st.sidebar.success(f"Using uploaded file: {upl.name}")
        df = df_upl
        # Attempt same normalization as load_data for common schemas
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
        # Recompute MIL/MOA if possible
        if "drop_from_100yd_zero_in" in df.columns and "distance_yd" in df.columns:
            dist_hund = pd.to_numeric(df["distance_yd"], errors="coerce") / 100.0
            denom_moa = (1.047 * dist_hund).replace(0, pd.NA)
            denom_mil = (3.6 * dist_hund).replace(0, pd.NA)
            df["drop_moa"] = pd.to_numeric(df["drop_from_100yd_zero_in"], errors="coerce") / denom_moa
            df["drop_mil"] = pd.to_numeric(df["drop_from_100yd_zero_in"], errors="coerce") / denom_mil
    else:
        df = df_local.copy()
else:
    df = df_local.copy()

if df is None or df.empty:
    st.info("Load or upload a dataset to begin.")
    st.stop()

# Common lists
cartridges = sorted(df["cartridge"].dropna().unique().tolist()) if "cartridge" in df.columns else []
min_d = int(df["distance_yd"].min()) if "distance_yd" in df.columns else 0
max_d = int(df["distance_yd"].max()) if "distance_yd" in df.columns else 0

# Tabs: Single vs Compare
tab1, tab2 = st.tabs(["🎯 Single Caliber", "🆚 Compare Calibers"])

# ---------------------------------
# Single Caliber tab
# ---------------------------------
with tab1:
    st.sidebar.header("Single Caliber Controls")
    cart = st.sidebar.selectbox("Caliber", options=cartridges, index=0 if cartridges else None, key="single_cart")
    df_s = df[df["cartridge"]==cart].copy()

    if "bullet_weight_gr" in df_s.columns and not df_s.empty:
        weights = sorted(df_s["bullet_weight_gr"].dropna().unique().tolist())
        w = st.sidebar.selectbox("Bullet weight (gr)", options=weights, index=0 if weights else None, key="single_weight")
        df_s = df_s[df_s["bullet_weight_gr"]==w]

    rng = st.sidebar.slider("Distance range (yd)", min_value=min_d, max_value=max_d, value=(min_d, max_d), step=50 if (max_d-min_d)>=50 else 1, key="single_range")
    df_s = df_s[(df_s["distance_yd"]>=rng[0]) & (df_s["distance_yd"]<=rng[1])].sort_values("distance_yd")

    # Info metrics
    info_cols = st.columns(4)
    if "ballistic_coefficient_G1" in df_s.columns and not df_s.empty:
        bc_vals = df_s["ballistic_coefficient_G1"].dropna().unique()
        info_cols[0].metric("BC (G1)", f"{bc_vals[0]:.3f}" if len(bc_vals)>0 else "—")
    if "retained_velocity_fps" in df_s.columns and not df_s.empty:
        info_cols[1].metric("Start Velocity (fps)", f"{df_s['retained_velocity_fps'].iloc[0]:.0f}")
    if "retained_energy_ftlb" in df_s.columns and not df_s.empty:
        info_cols[2].metric("Start Energy (ft·lb)", f"{df_s['retained_energy_ftlb'].iloc[0]:.0f}")
    if "time_of_flight_s" in df_s.columns and not df_s.empty:
        info_cols[3].metric("Max TOF (s)", f"{df_s['time_of_flight_s'].max():.3f}")

    def line_chart(data, x, y, title, y_title):
        chart = alt.Chart(data).mark_line(point=True).encode(
            x=alt.X(x, title="Distance (yd)"),
            y=alt.Y(y, title=y_title),
            tooltip=[alt.Tooltip("distance_yd", title="Distance (yd)"),
                     alt.Tooltip(y, title=y_title)]
        ).properties(title=title, height=320)
        st.altair_chart(chart, use_container_width=True)

    st.subheader(f"{cart} — {w} gr" if "bullet_weight_gr" in df_s.columns and not df_s.empty else cart)

    row1 = st.columns(2)
    if "drop_from_100yd_zero_in" in df_s.columns:
        line_chart(df_s, "distance_yd:Q", "drop_from_100yd_zero_in:Q", "Trajectory (Drop from 100 yd Zero)", "Drop (in)")
    if "retained_velocity_fps" in df_s.columns:
        line_chart(df_s, "distance_yd:Q", "retained_velocity_fps:Q", "Velocity vs Distance", "Velocity (fps)")

    row2 = st.columns(2)
    if "retained_energy_ftlb" in df_s.columns:
        line_chart(df_s, "distance_yd:Q", "retained_energy_ftlb:Q", "Energy vs Distance", "Energy (ft·lb)")
    if "wind_drift_10mph_in" in df_s.columns:
        line_chart(df_s, "distance_yd:Q", "wind_drift_10mph_in:Q", "Wind Drift (10 mph full value)", "Drift (in)")

    with st.expander("📋 DOPE Table"):
        cols_to_show = ["distance_yd","drop_from_100yd_zero_in","drop_moa","drop_mil","retained_velocity_fps","retained_energy_ftlb","wind_drift_10mph_in","time_of_flight_s"]
        present = [c for c in cols_to_show if c in df_s.columns]
        st.dataframe(df_s[present].round(3), use_container_width=True)

    st.download_button("Download filtered (CSV)", data=df_s.to_csv(index=False), file_name=f"{cart}_{w}_filtered.csv", mime="text/csv")

# ---------------------------------
# Compare Calibers tab
# ---------------------------------
with tab2:
    st.sidebar.header("Compare Calibers Controls")
    sel = st.sidebar.multiselect("Select calibers", options=cartridges, default=cartridges[:2] if len(cartridges)>=2 else cartridges, key="cmp_carts")

    df_c = df[df["cartridge"].isin(sel)].copy()

    weight_filter = st.sidebar.checkbox("Match bullet weights across calibers (±10gr)", value=True, key="cmp_weightf")
    target_weight = None
    if weight_filter and "bullet_weight_gr" in df_c.columns and not df_c.empty:
        target_weight = int(df_c["bullet_weight_gr"].median())
        df_c = df_c.loc[(df_c["bullet_weight_gr"] - target_weight).abs() <= 10]

    rng2 = st.sidebar.slider("Distance range (yd)", min_value=min_d, max_value=max_d, value=(min_d, max_d), step=50 if (max_d-min_d)>=50 else 1, key="cmp_range")
    df_c = df_c[(df_c["distance_yd"]>=rng2[0]) & (df_c["distance_yd"]<=rng2[1])].sort_values(["cartridge","distance_yd"])

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
    if target_weight:
        st.caption(f"Weight filter applied: ~{target_weight} gr (±10gr)")

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
        cols = ["cartridge","bullet_weight_gr","distance_yd","drop_from_100yd_zero_in","drop_moa","drop_mil","retained_velocity_fps","retained_energy_ftlb","wind_drift_10mph_in","time_of_flight_s"]
        present = [c for c in cols if c in df_c.columns]
        st.dataframe(df_c[present].round(3), use_container_width=True)

    st.download_button("Download comparison (CSV)", data=df_c.to_csv(index=False), file_name="comparison_filtered.csv", mime="text/csv")

st.markdown("---")
st.caption("Drop values are relative to a 100 yd zero (positive = low). MIL/MOA approximations: 1 MOA ≈ 1.047\" per 100 yd; 1 MIL ≈ 3.6\" per 100 yd.")
