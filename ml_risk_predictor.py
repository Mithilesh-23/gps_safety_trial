from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# RANDOM FOREST ML RISK PREDICTOR
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR /
    "random_forest_risk_model.pkl"
)


# Load trained model once when Flask starts.
_model_package = joblib.load(
    MODEL_PATH
)

MODEL = _model_package["model"]

FEATURES = _model_package["features"]

DAY_MAPPING = _model_package["day_mapping"]


# ============================================================
# RISK LEVEL
# ============================================================

def risk_level_from_score(score):

    score = float(score)

    if score >= 65:
        return "High"

    if score >= 40:
        return "Medium"

    return "Low"


# ============================================================
# PREDICT RISK
# ============================================================

def predict_risk(
    crime_risk,
    lighting_level,
    crowd_density,
    traffic_level,
    police_presence,
    hour,
    day_of_week
):

    day_code = DAY_MAPPING.get(
        str(day_of_week),
        -1
    )

    input_data = pd.DataFrame(
        [{
            "crime_risk":
                float(crime_risk),

            "lighting_level":
                float(lighting_level),

            "crowd_density":
                float(crowd_density),

            "traffic_level":
                float(traffic_level),

            "police_presence":
                float(police_presence),

            "hour":
                float(hour),

            "day_code":
                float(day_code)
        }]
    )

    prediction = MODEL.predict(
        input_data[FEATURES]
    )[0]

    prediction = max(
        0.0,
        min(
            100.0,
            float(prediction)
        )
    )

    return {
        "predicted_risk_score":
            round(
                prediction,
                2
            ),

        "predicted_risk_level":
            risk_level_from_score(
                prediction
            )
    }
