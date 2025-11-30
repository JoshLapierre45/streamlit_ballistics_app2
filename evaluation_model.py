#!/usr/bin/env python3
"""
evaluate_model.py

Evaluate the pretrained HitProbabilityModel on a labeled dataset.

Usage:
    python evaluate_model.py \
        --data-path Data/hit_probability_semi_realistic.csv

This script:
- Loads a CSV (or Excel) dataset with an 'outcome' column (0/1 hits/misses)
- Builds feature dictionaries matching the Streamlit app
- Uses HitProbabilityModel.predict_proba_single(...) for predictions
- Computes Accuracy, ROC AUC, and Log Loss on a held-out test set
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, confusion_matrix
from sklearn.model_selection import train_test_split

from model import HitProbabilityModel


# Columns expected by HitProbabilityModel (as used in the Streamlit app)
FEATURE_COLS = [
    "range_yd",
    "mv_fps",
    "mv_sd",
    "wind_mph",
    "DA_ft",
    "temp_F",
    "pressure_inHg",
    "rh_pct",
    "group_moa",
    "target_size_moa",
]

TARGET_COL = "outcome"


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load dataset from CSV or Excel.

    The file must contain:
    - outcome (0/1)
    - the feature columns listed in FEATURE_COLS
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext} (use .csv or .xlsx)")

    return df


def check_columns(df: pd.DataFrame):
    """Verify that all required columns are present."""
    missing_features = [c for c in FEATURE_COLS if c not in df.columns]
    missing_target = TARGET_COL not in df.columns

    if missing_features or missing_target:
        msg_parts = []
        if missing_features:
            msg_parts.append(f"Missing feature columns: {missing_features}")
        if missing_target:
            msg_parts.append(f"Missing target column: '{TARGET_COL}'")
        raise ValueError(
            "Dataset is missing required columns.\n" + "\n".join(msg_parts)
        )


def build_features_and_labels(df: pd.DataFrame):
    """
    Create:
    - X_dicts: list of per-row feature dicts
    - y: numpy array of 0/1 labels
    """
    # Coerce numerics to be safe
    for c in FEATURE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

    # Drop rows with missing target or features
    df_clean = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()

    X_dicts = [row[FEATURE_COLS].to_dict() for _, row in df_clean.iterrows()]
    y = df_clean[TARGET_COL].astype(int).to_numpy()

    return X_dicts, y, df_clean


def evaluate_model(data_path: str, test_size: float = 0.2, random_state: int = 42):
    print("=== Evaluating HitProbabilityModel ===")
    print(f"Data path: {data_path}\n")

    # 1) Load and validate data
    df = load_dataset(data_path)
    print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")
    check_columns(df)

    # 2) Build features and labels
    X_dicts, y, df_clean = build_features_and_labels(df)
    print(f"After cleaning: {len(df_clean)} rows remain.")

    if len(df_clean) < 30:
        print(
            "WARNING: Very small dataset (< 30 rows). "
            "Metrics may be unstable; consider more data."
        )

    # 3) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_dicts, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Train size: {len(y_train)}  |  Test size: {len(y_test)}")
    print(f"Class counts (whole dataset): {np.bincount(y)}\n")

    # 4) Load pretrained model
    model = HitProbabilityModel()

    # 5) Predict on test set
    probs_test = []
    preds_test = []

    for feat_dict in X_test:
        p = model.predict_proba_single(feat_dict)
        probs_test.append(p)
        preds_test.append(int(p >= 0.5))

    probs_test = np.array(probs_test)
    preds_test = np.array(preds_test)

    # 6) Compute metrics
    acc = accuracy_score(y_test, preds_test)
    try:
        roc = roc_auc_score(y_test, probs_test)
    except ValueError:
        roc = float("nan")  # e.g. if only one class present in y_test

    try:
        ll = log_loss(y_test, probs_test)
    except ValueError:
        ll = float("nan")

    cm = confusion_matrix(y_test, preds_test)

    print("=== Metrics on Test Set ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"ROC AUC  : {roc:.4f}" if np.isfinite(roc) else "ROC AUC  : N/A")
    print(f"Log Loss : {ll:.4f}" if np.isfinite(ll) else "Log Loss : N/A")
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm)
    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the pretrained HitProbabilityModel on a dataset."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="Data/hit_probability_semi_realistic.csv",
        help="Path to CSV or Excel file with feature columns and 'outcome'.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to use as test set (default: 0.2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split (default: 42).",
    )

    args = parser.parse_args()
    evaluate_model(args.data_path, test_size=args.test_size, random_state=args.seed)


if __name__ == "__main__":
    main()
