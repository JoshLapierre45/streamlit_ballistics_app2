import pandas as pd
import numpy as np

# Load the file using correct path
df = pd.read_csv("hit_probability_semi_realistic.csv")

# Drop garbage columns
df = df.drop(columns=[col for col in df.columns if col.strip() in ["2", "Unnamed: 0"]], errors="ignore")

# Fix date field
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Normalize string identifiers
df["shooter_id"] = df["shooter_id"].str.strip().str.lower()
df["rifle_id"] = df["rifle_id"].str.strip()

# One-hot encode 'rest'
df = pd.get_dummies(df, columns=["rest"], prefix="rest", drop_first=True)

# Add clean engineered features
df["range_sq"] = df["range_yd"] ** 2
df["wind_sq"] = df["wind_mph"] ** 2
df["range_x_wind"] = df["range_yd"] * df["wind_mph"]
df["group_x_range"] = df["group_moa"] * df["range_yd"]
df["range_x_target"] = df["range_yd"] * df["target_size_moa"]
df["wind_cross_mph"] = df["wind_mph"] * np.cos(np.deg2rad(df["wind_dir_deg"]))
df["wind_head_mph"] = df["wind_mph"] * np.sin(np.deg2rad(df["wind_dir_deg"]))

# Save clean version
df.to_csv("hit_probability_cleaned.csv", index=False)

print("Cleaned dataset saved to hit_probability_cleaned.csv")
