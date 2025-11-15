import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss

# =========================
# Schema / helpers
# =========================
REQ_COLS_PER_SHOT = [
    "shooter_id","date","rifle_id","range_yd","mv_fps","mv_sd",
    "wind_mph","wind_dir_deg","DA_ft","temp_F","pressure_inHg","rh_pct",
    "group_moa","rest","target_size_moa","shot_id","outcome"
]
REQ_COLS_AGG = [
    "shooter_id","date","rifle_id","range_yd","mv_fps","mv_sd",
    "wind_mph","wind_dir_deg","DA_ft","temp_F","pressure_inHg","rh_pct",
    "group_moa","rest","target_size_moa","shots","hits"
]

REST_MAP = {
    "prone_bag": 2, "bipod": 1, "bench": 2, "offhand": 0, "tripod": 1, "unknown": 1
}

@st.cache_data
def load_builtin_hit_history() -> pd.DataFrame:
    """
    Load the default hit-probability dataset that ships with the app.
    Adjust filenames if needed.
    """
    candidates = [
        "hit_probability_semi_realistic.xlsx",
        # "hit_probability_semi_realistic.csv",  # uncomment if you have a CSV instead
    ]
    for fname in candidates:
        if os.path.exists(fname):
            if fname.lower().endswith(".csv"):
                return pd.read_csv(fname)
            else:
                return pd.read_excel(fname)
    # If nothing found, raise so the UI can tell the user to upload
    raise FileNotFoundError("No built-in hit probability file found.")


def is_per_shot(df: pd.DataFrame) -> bool:
    return "outcome" in df.columns and "shot_id" in df.columns


def expand_aggregate_to_shots(df_agg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df_agg.iterrows():
        hits = int(r["hits"])
        shots = int(r["shots"])
        for i in range(shots):
            outcome = 1 if i < hits else 0
            shot_id = f"{r.get('date','')}-{i+1}"
            base = r.drop(labels=["shots","hits"]).to_dict()
            base.update({"shot_id": shot_id, "outcome": outcome})
            rows.append(base)
    return pd.DataFrame(rows)


def featurize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Coerce numerics
    numeric_cols = [
        "range_yd","mv_fps","mv_sd","wind_mph","wind_dir_deg","DA_ft",
        "temp_F","pressure_inHg","rh_pct","group_moa","target_size_moa"
    ]
    for c in numeric_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Dates
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")

    # Wind components (met convention: wind FROM direction)
    th = np.deg2rad(out["wind_dir_deg"].fillna(0))
    out["wind_cross_mph"] = out["wind_mph"] * np.sin(th)
    out["wind_head_mph"]  = out["wind_mph"] * np.cos(th)

    # Rest encoding
    out["rest_norm"] = (
        out["rest"].astype(str).str.lower().map(REST_MAP).fillna(1).astype(int)
        if "rest" in out.columns else 1
    )

    # ---- Curvature & interactions to help logistic regression ----
    out["range_sq"]       = out["range_yd"]**2
    out["wind_sq"]        = out["wind_mph"]**2
    out["range_x_wind"]   = out["range_yd"] * out["wind_mph"]
    out["range_x_target"] = out["range_yd"] * out["target_size_moa"]
    out["group_x_range"]  = out["group_moa"] * out["range_yd"]

    # If any are entirely NaN due to missing inputs, fill with 0 so the scaler/fit won’t fail
    for c in ["range_sq","wind_sq","range_x_wind","range_x_target","group_x_range"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    return out


def make_feature_matrix(df: pd.DataFrame):
    feature_cols = [
        "range_yd","mv_fps","mv_sd",
        "wind_cross_mph","wind_head_mph",
        "DA_ft","temp_F","pressure_inHg","rh_pct",
        "group_moa","target_size_moa","rest_norm",
        "range_sq","wind_sq","range_x_wind","range_x_target","group_x_range"
    ]
    X = df[feature_cols].astype(float)
    y = df["outcome"].astype(int)
    return X, y, feature_cols


def train_calibrated(Xtr, ytr, method="isotonic", cv=5):
    base = LogisticRegression(max_iter=2000, class_weight="balanced")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(base, method=method, cv=cv))
    ])
    pipe.fit(Xtr, ytr)
    return pipe


# =========================
# Streamlit Tab
# =========================
def render():
    st.header("🎯 Hit Probability")

    # helper to exit this tab without killing app
    def bail(msg, level="info"):
        {"info": st.info, "warning": st.warning, "error": st.error}[level](msg)
        return

    st.markdown(
        "Upload shooter history to train a model that predicts **p(hit)**. "
        "If you don't upload anything, the built-in semi-realistic dataset is used. "
        "**Per-shot format** is preferred (`outcome` = 1/0). "
    )

    # ---- Template download ----
    template_cols = REQ_COLS_PER_SHOT
    template_df = pd.DataFrame(columns=template_cols)
    st.download_button(
        "Download CSV template",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="hit_probability_template.csv",
        mime="text/csv",
        key="hitprob_template_btn"
    )

    # ---- Upload or fallback to built-in ----
    uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"], key="hitprob_uploader")

    # Case 1: user uploaded their own history
    if uploaded is not None:
        uploaded.seek(0)
        try:
            raw = pd.read_csv(uploaded)
        except Exception as e:
            return bail(f"Could not read uploaded CSV: {e}", "error")
        st.caption(f"Using uploaded hit history: {uploaded.name}")

    # Case 2: no upload → fall back to built-in dataset
    else:
        try:
            raw = load_builtin_hit_history()
        except FileNotFoundError:
            return bail(
                "No built-in hit history file found. "
                "Please upload a CSV using the template above.",
                "info",
            )
        st.caption("Using built-in hit_probability_semi_realistic dataset.")

    if raw.empty:
        return bail("Your CSV appears to be empty.", "error")

    st.write("Preview", raw.head())

    # ---- Schema detection ----
    if is_per_shot(raw):
        missing = [c for c in REQ_COLS_PER_SHOT if c not in raw.columns]
        if missing:
            st.warning(f"Missing expected columns (per-shot): {missing}")
        data = raw.copy()
    else:
        missing = [c for c in REQ_COLS_AGG if c not in raw.columns]
        if missing:
            return bail(f"Your file is missing columns: {missing}", "error")
        st.info("Detected aggregate format with `shots` and `hits`. Expanding to per-shot rows...")
        data = expand_aggregate_to_shots(raw)

    # ---- Feature engineering ----
    data = featurize(data).dropna(subset=["outcome"])

    # ---- Filters ----
    left, right = st.columns(2)
    with left:
        shooter = st.selectbox(
            "Shooter",
            sorted(data["shooter_id"].dropna().astype(str).unique()),
            key="hitprob_shooter"
        )
    with right:
        rifle = st.selectbox(
            "Rifle",
            sorted(data["rifle_id"].dropna().astype(str).unique()),
            key="hitprob_rifle"
        )

    df = data.query("shooter_id == @shooter and rifle_id == @rifle").copy()
    if df.empty:
        return bail("No rows after filtering. Pick another shooter/rifle or upload more data.", "warning")

    # ---- Matrices ----
    X, y, feature_cols = make_feature_matrix(df)

    # ---- Data sufficiency checks ----
    counts = y.value_counts()
    if y.nunique() < 2:
        return bail(
            "Only one class (all hits or all misses) after filtering. "
            "Add more mixed data or broaden filters.",
            "error"
        )
    if len(y) < 20:
        st.warning(f"Very small dataset (n={len(y)}). Training on all data without holdout; log loss not shown.")
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
        ])
        model.fit(X, y)
        show_predict_ui(model, df)
        return

    # ---- Split ----
    min_class = int(counts.min())
    # ensure at least 1 minority sample in test; cap test_size
    test_size = max(0.2, min(0.4, 1.0 / max(min_class, 2)))
    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    except ValueError:
        # fallback no stratify if tiny/bad
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)

    # ---- Train (calibrated logistic) with fallback ----
    try:
        model = train_calibrated(Xtr, ytr, method="isotonic", cv=5)
    except Exception:
        model = train_calibrated(Xtr, ytr, method="sigmoid", cv=3)
        st.info("Used sigmoid calibration fallback due to limited data.")

    # ---- Evaluate: Log Loss only ----
    pte = model.predict_proba(Xte)[:, 1]
    try:
        loss = log_loss(yte, pte)
    except ValueError:
        loss = float("nan")

    with st.expander("Model performance", expanded=True):
        st.metric(
            "Log Loss",
            f"{loss:.4f}" if np.isfinite(loss) else "—",
            help="Cross-entropy; lower is better. Penalizes confident wrong predictions most."
        )

    # ---- SHAP feature importance ----
    with st.expander("Feature importance (SHAP)", expanded=False):
        try:
            # We train a simple logistic model just for SHAP explanations,
            # using the same features Xtr/ytr. This keeps things fast & interpretable.
            expl_scaler = StandardScaler()
            Xtr_scaled = expl_scaler.fit_transform(Xtr)

            expl_lr = LogisticRegression(max_iter=2000, class_weight="balanced")
            expl_lr.fit(Xtr_scaled, ytr)

            # Use LinearExplainer for logistic regression
            explainer = shap.LinearExplainer(expl_lr, Xtr_scaled, feature_names=feature_cols)
            shap_values = explainer.shap_values(Xtr_scaled)

            st.caption("Higher SHAP value → higher predicted hit probability for that feature value.")

            fig, ax = plt.subplots()
            shap.summary_plot(
                shap_values,
                Xtr,
                feature_names=feature_cols,
                plot_type="bar",
                show=False
            )
            st.pyplot(fig)

        except Exception as e:
            st.info(f"Could not generate SHAP plot: {e}")

    # ---- Predict UI ----
    show_predict_ui(model, df)



def show_predict_ui(model, df):
    """UI to predict p(hit) vs range using current conditions."""
    st.subheader("Predict Hit % vs Range")

    # Defaults from filtered data
    def _med(col, default):
        return float(df[col].median()) if col in df.columns and not df[col].isna().all() else default

    c1, c2, c3 = st.columns(3)
    with c1:
        mv_fps = st.number_input("Muzzle velocity (fps)", value=_med("mv_fps", 2800.0), key="hitprob_mv")
        mv_sd  = st.number_input("MV SD (fps)", value=_med("mv_sd", 12.0), key="hitprob_mvsd")
        group  = st.number_input("Group (MOA)", value=_med("group_moa", 0.9), key="hitprob_group")
    with c2:
        wind_mph = st.number_input("Wind speed (mph)", value=_med("wind_mph", 8.0), key="hitprob_wspd")
        wind_dir = st.number_input("Wind from (deg, met)", value=_med("wind_dir_deg", 270.0), key="hitprob_wdir")
        target_moa = st.number_input("Target size (MOA)", value=_med("target_size_moa", 1.0), key="hitprob_tsize")
    with c3:
        DA_ft = st.number_input("Density Altitude (ft)", value=_med("DA_ft", 8000.0), key="hitprob_DA")
        tempF = st.number_input("Temperature (°F)", value=_med("temp_F", 60.0), key="hitprob_temp")
        press = st.number_input("Pressure (inHg)", value=_med("pressure_inHg", 24.8), key="hitprob_press")
        rh    = st.number_input("RH (%)", value=_med("rh_pct", 20.0), key="hitprob_rh")

    rest_str = st.selectbox("Rest", options=list(REST_MAP.keys()), index=1, key="hitprob_rest")

    # Range band from existing data if possible
    rmin = int(df["range_yd"].min()) if "range_yd" in df.columns else 100
    rmax = int(df["range_yd"].max()) if "range_yd" in df.columns else 1000
    rmin, rmax = min(rmin, rmax), max(rmin, rmax)
    grid = st.slider(
        "Range band (yd)",
        min_value=rmin,
        max_value=max(100, rmax),
        value=(rmin, rmax),
        step=25,
        key="hitprob_range_slider"
    )
    ranges = np.arange(grid[0], grid[1] + 1, 25)

    th = np.deg2rad(wind_dir)
    wind_cross = wind_mph * np.sin(th)
    wind_head  = wind_mph * np.cos(th)
    rest_norm = REST_MAP.get(rest_str, 1)

    pred_df = pd.DataFrame({
        "range_yd": ranges,
        "mv_fps": mv_fps,
        "mv_sd": mv_sd,
        "wind_cross_mph": wind_cross,
        "wind_head_mph": wind_head,
        "DA_ft": DA_ft,
        "temp_F": tempF,
        "pressure_inHg": press,
        "rh_pct": rh,
        "group_moa": group,
        "target_size_moa": target_moa,
        "rest_norm": rest_norm,
    })

    # ---- Add engineered features ----
    pred_df["range_sq"]       = pred_df["range_yd"] ** 2
    pred_df["wind_sq"]        = wind_mph ** 2
    pred_df["range_x_wind"]   = pred_df["range_yd"] * wind_mph
    pred_df["range_x_target"] = pred_df["range_yd"] * target_moa
    pred_df["group_x_range"]  = group * pred_df["range_yd"]

    # Ensure column order and missing fill (robustness)
    needed = [
        "range_yd","mv_fps","mv_sd",
        "wind_cross_mph","wind_head_mph",
        "DA_ft","temp_F","pressure_inHg","rh_pct",
        "group_moa","target_size_moa","rest_norm",
        "range_sq","wind_sq","range_x_wind","range_x_target","group_x_range"
    ]
    for c in needed:
        if c not in pred_df.columns:
            pred_df[c] = 0.0
    pred_df = pred_df[needed]

    phit = model.predict_proba(pred_df)[:, 1]
    out = pd.DataFrame({"range_yd": ranges, "p_hit": phit})
    out["p_hit_%"] = (out["p_hit"] * 100).round(1)

    st.line_chart(out.set_index("range_yd")[["p_hit"]])
    st.dataframe(out, use_container_width=True)
    st.caption("⚠️ Advisory only. Not a substitute for pressure-tested data or safe shooting practices.")
