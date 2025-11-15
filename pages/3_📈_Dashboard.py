# pages/3_📈_Dashboard.py
import streamlit as st
import altair as alt
import pandas as pd
from datetime import datetime
from utils.data_loader import load_ballistics

st.set_page_config(page_title="Dashboard", layout="wide")
st.title("📈 Ballistics Dashboard")

df, src = load_ballistics()
if df.empty:
    st.error("No ballistics data found.")
    st.stop()

# --- Filters ---
c1, c2, c3 = st.columns(3)
with c1:
    cartridges = sorted(df["cartridge"].dropna().astype(str).unique())
    cart_sel = st.multiselect("Cartridge", options=cartridges, default=cartridges[:2] if len(cartridges)>=2 else cartridges)
with c2:
    min_d, max_d = int(df["distance_yd"].min()), int(df["distance_yd"].max())
    dist_sel = st.slider("Distance (yd)", min_value=min_d, max_value=max_d, value=(min_d, max_d), step=25)
with c3:
    has_bw = "bullet_weight_gr" in df.columns
    bw = st.slider("Bullet weight (gr)", 
                   min_value=int(df["bullet_weight_gr"].min()) if has_bw else 100,
                   max_value=int(df["bullet_weight_gr"].max()) if has_bw else 300,
                   value=(int(df["bullet_weight_gr"].min()), int(df["bullet_weight_gr"].max())) if has_bw else (140, 180)
                   ) if has_bw else (None, None)

mask = df["distance_yd"].between(*dist_sel)
if cart_sel: mask &= df["cartridge"].isin(cart_sel)
if has_bw and bw != (None, None):
    mask &= df["bullet_weight_gr"].between(*bw)

d = df.loc[mask].copy()
st.caption(f"Source: {src} • Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# --- KPIs ---
k1, k2, k3 = st.columns(3)
k1.metric("Rows", f"{len(d):,}")
if "retained_velocity_fps" in d.columns:
    k2.metric("Avg Velocity (fps)", f"{d['retained_velocity_fps'].mean():.0f}")
if "wind_drift_10mph_in" in d.columns:
    k3.metric("Avg Drift (in)", f"{d['wind_drift_10mph_in'].mean():.2f}")

# --- Linked visuals ---
left, right = st.columns(2)

with left:
    st.subheader("Velocity vs Distance")
    vchart = (
        alt.Chart(d)
        .mark_line(point=True)
        .encode(
            x=alt.X("distance_yd:Q", title="Distance (yd)"),
            y=alt.Y("retained_velocity_fps:Q", title="Velocity (fps)"),
            color=alt.Color("cartridge:N", legend=alt.Legend(title="Cartridge")),
            tooltip=["cartridge","distance_yd","retained_velocity_fps"]
        ).properties(height=320)
    )
    st.altair_chart(vchart, use_container_width=True)

with right:
    st.subheader("Energy vs Distance")
    echart = (
        alt.Chart(d)
        .mark_line(point=True)
        .encode(
            x=alt.X("distance_yd:Q", title="Distance (yd)"),
            y=alt.Y("retained_energy_ftlb:Q", title="Energy (ft·lb)"),
            color=alt.Color("cartridge:N", legend=alt.Legend(title="Cartridge")),
            tooltip=["cartridge","distance_yd","retained_energy_ftlb"]
        ).properties(height=320)
    )
    st.altair_chart(echart, use_container_width=True)

with st.expander("Insights & Notes", expanded=True):
    st.markdown("""
- Velocity and energy both decline with distance; slope differences reflect ballistic efficiency.  
- Heavier bullet weights often maintain energy better at longer ranges.  
- Wind drift (see EDA page) typically grows with distance—consider it alongside energy.  
- **Limitations:** This dashboard is descriptive; it doesn’t infer causes. Data may mix loads/conditions.
""")
