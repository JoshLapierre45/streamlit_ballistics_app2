# model.py
from pathlib import Path
import numpy as np
import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "hit_model.pkl"


class HitProbabilityModel:
    """
    Wrapper for the trained classifier predicting hit probability
    given shooting and environmental conditions.
    Logistic regression model estimating hit probability for long-range shooting.

    Trained on semi-realistic simulated data using features:
    - range_yd
    - muzzle velocity & SD
    - group size (MOA)
    - target size (MOA)
    - crosswind (mph)
    """

    def __init__(self, model_path: Path = MODEL_PATH):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.scaler = bundle["scaler"]
        self.feature_names = bundle["features"]  # list of feature column names
        self.target_name = bundle.get("target", "outcome")

    def predict_proba_single(self, features_dict: dict) -> float:
        """
        features_dict MUST have keys matching self.feature_names, e.g.:

        {
            "range_yd": 600,
            "mv_fps": 2800,
            "mv_sd": 12,
            "wind_mph": 8,
            "DA_ft": 2500,
            "temp_F": 70,
            "pressure_inHg": 29.92,
            "rh_pct": 40,
            "group_moa": 1.0,
            "target_size_moa": 2.0,
        }

        Returns: probability of hit (float between 0 and 1).
        """
        x = np.array([[features_dict[name] for name in self.feature_names]])
        x_scaled = self.scaler.transform(x)
        prob = self.model.predict_proba(x_scaled)[0, 1]
        return float(prob)
