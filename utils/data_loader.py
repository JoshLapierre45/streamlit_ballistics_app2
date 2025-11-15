# utils/data_loader.py
import os
import pandas as pd
import streamlit as st

@st.cache_data
def load_ballistics():
    """Load & normalize the Ballistics.xlsx or CSV (if present)."""
    # Prefer CSV if present, otherwise Excel
    candidates = [
        ("CSV", os.path.join("Data", "ballistics_cartridge_performance_v4.csv")),
        ("Excel", os.path.join("Data", "Ballistics.xlsx")),
    ]
    found = None
    for label, path in candidates:
        if os.path.exists(path):
            found = (label, path)
            break
    if not found:
        return pd.DataFrame(), "No local data found. Upload a file."
    label, path = found
    if label == "Excel":
        try:
            df = pd.read_excel(path, engine="openpyxl")
        except Exception:
            df = pd.read_excel(path)
        source_note = f"{os.path.basename(path)} (Excel)"
    else:
        df = pd.read_csv(path)
        source_note = f"{os.path.basename(path)} (CSV)"

    # Normalize common column names if they exist
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

    # Add MOA/MIL if possible
    if {"drop_from_100yd_zero_in", "distance_yd"}.issubset(df.columns):
        dist_hund = pd.to_numeric(df["distance_yd"], errors="coerce") / 100.0
        df["drop_moa"] = pd.to_numeric(df["drop_from_100yd_zero_in"], errors="coerce") / ((1.047 * dist_hund).replace(0, pd.NA))
        df["drop_mil"] = pd.to_numeric(df["drop_from_100yd_zero_in"], errors="coerce") / ((3.6 * dist_hund).replace(0, pd.NA))

    return df, source_note


@st.cache_data
def load_hitprob():
    """Load the hit probability dataset if present."""
    path_csv = os.path.join("Data", "hit_probability_semi_realistic.csv")
    if os.path.exists(path_csv):
        df = pd.read_csv(path_csv)
        return df, os.path.basename(path_csv)
    return pd.DataFrame(), "Upload a hit-probability CSV on the Hit Probability page."
