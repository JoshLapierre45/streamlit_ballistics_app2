# pages/train_model.py

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import joblib


def main():
    # --- Paths: go from /pages up to project root ---
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / "Data" / "hit_probability_cleaned.csv"
    MODEL_PATH = BASE_DIR / "models" / "hit_model.pkl"

    print(f"Loading data from: {DATA_PATH}")

    # --- 1. Load data ---
    # utf-8-sig handles BOMs from Excel exports
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    print("Columns:", df.columns.tolist())

    # --- 2. Select features and target ---
    feature_cols = [
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

    target_col = "outcome"   # assumed 0/1 for miss/hit

    # Drop rows that have missing values in any of these
    df = df.dropna(subset=feature_cols + [target_col])

    X = df[feature_cols]
    y = df[target_col]

    # --- 3. Train / test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- 4. Scale numeric features ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- 5. Fit classification model (logistic regression) ---
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # --- 6. Evaluate ---
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        # In case only one class present in y_test
        auc = float("nan")

    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC: {auc:.4f}")
    print("Confusion matrix:\n", cm)
    print("\nClassification report:\n", classification_report(y_test, y_pred))

    # --- 7. Save model + scaler + feature names ---
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "scaler": scaler,
        "features": feature_cols,
        "target": target_col,
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
