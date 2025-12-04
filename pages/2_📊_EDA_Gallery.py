# pages/2_📊_EDA_Gallery.py
import streamlit as st
import altair as alt
import pandas as pd
from utils.data_loader import load_ballistics

st.set_page_config(page_title="EDA Gallery", layout="wide")
st.title("📊 EDA Gallery")

df, src = load_ballistics()
st.caption(f"Source: {src}")

if df.empty:
    st.error("No ballistics data found. Place `Ballistics.xlsx` (or CSV) in Data/ or upload on other pages.")
    st.stop()

# Simple UI filters for interactivity
cartridges = sorted(df.get("cartridge", pd.Series(dtype=str)).dropna().unique().tolist())
colA, colB = st.columns(2)
with colA:
    cart_sel = st.multiselect("Filter by cartridge", options=cartridges, default=cartridges[:2] if len(cartridges)>=2 else cartridges)
with colB:
    yd = st.slider("Distance range (yd)", min_value=int(df["distance_yd"].min()),
                   max_value=int(df["distance_yd"].max()), value=(int(df["distance_yd"].min()), int(df["distance_yd"].max())))
mask = (df["distance_yd"].between(*yd)) & (df["cartridge"].isin(cart_sel) if cart_sel else True)
d = df.loc[mask].copy()

# 1) Histogram (chart type #1)
st.subheader("Distribution of Retained Velocity")
st.markdown("_Question:_ How is **retained velocity** distributed within the selected range?")
h = (
    alt.Chart(d).mark_bar().encode(
        x=alt.X("retained_velocity_fps:Q", bin=alt.Bin(maxbins=30), title="Velocity (fps)"),
        y=alt.Y("count()", title="Count"),
        tooltip=["count()"]
    ).properties(height=260)
)
st.altair_chart(h, use_container_width=True)
st.markdown("**How to read:**")
st.write("- X-axis groups velocities into bins; Y-axis shows how many rows fall in each bin.")
st.write("- Taller bars → more rows around that velocity.")
st.write("- Use the filters to focus the distribution.")
st.markdown("**Observations:**")
st.write("- Look for skew (more low/high velocities) or multi-modal patterns.")
st.write("- Unusually wide distributions may indicate mixed loads or measurement differences.")

# 2) Scatter (chart type #2)
st.subheader("Velocity vs Distance by Cartridge")
st.markdown("_Question:_ How does **velocity** change with **distance**, and does it differ by cartridge?")
sc = (
    alt.Chart(d)
    .mark_line()
    .encode(
        x=alt.X("distance_yd:Q", title="Distance (yd)"),
        y=alt.Y("retained_velocity_fps:Q", title="Velocity (fps)"),
        color=alt.Color("cartridge:N", legend=alt.Legend(title="Cartridge")),
        tooltip=["cartridge", "distance_yd", "retained_velocity_fps"]
    )
    .properties(height=300)
)
st.altair_chart(sc, use_container_width=True)
st.markdown("**How to read:**")
st.write("- Each dot is a (distance, velocity) point; color = cartridge.")
st.write("- Hover for exact values; use the filters to subset.")
st.markdown("**Observations:**")
st.write("- Expect decreasing velocity with distance; slope differences reflect BC/weight.")
st.write("- Clusters may indicate different bullet weights or loads.")

# 3) Box plot (chart type #3)
st.subheader("Energy Distribution by Cartridge")
st.markdown("_Question:_ How does **retained energy** vary across cartridges?")
bx = (
    alt.Chart(d)
    .mark_boxplot()
    .encode(
        x=alt.X("cartridge:N", title="Cartridge"),
        y=alt.Y("retained_energy_ftlb:Q", title="Energy (ft·lb)")
    ).properties(height=300)
)
st.altair_chart(bx, use_container_width=True)
st.markdown("**How to read:**")
st.write("- Box shows median and interquartile range; whiskers show range (w/outliers).")
st.write("- Taller boxes/medians = generally higher energy.")
st.markdown("**Observations:**")
st.write("- Compare medians and spread; outliers can indicate unusual configurations.")

# 4) Line chart (chart type #4)
st.subheader("Wind Drift vs Distance")
st.markdown("_Question:_ How does **10 mph wind drift** change with distance?")
ln = (
    alt.Chart(d)
    .mark_line(point=True)
    .encode(
        x=alt.X("distance_yd:Q", title="Distance (yd)"),
        y=alt.Y("wind_drift_10mph_in:Q", title="Drift (in)"),
        color=alt.Color("cartridge:N", legend=alt.Legend(title="Cartridge")),
        tooltip=["cartridge", "distance_yd", "wind_drift_10mph_in"]
    )
    .properties(height=300)
)
st.altair_chart(ln, use_container_width=True)
st.markdown("**How to read:**")
st.write("- Each line traces drift vs distance; higher = more drift.")
st.write("- Hover shows exact drift at a distance.")
st.markdown("**Observations:**")
st.write("- Drift usually increases with distance; differences show ballistic performance.")

st.caption("Accessibility: color-blind-friendly defaults; all axes labeled with units where applicable.")
