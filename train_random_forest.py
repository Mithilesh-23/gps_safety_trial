from pathlib import Path
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
DATASET = BASE_DIR / "safety_ml_dataset.csv"
MODEL_FILE = BASE_DIR / "random_forest_risk_model.pkl"

DAY_MAPPING = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6,
    "BASELINE": -1
}

FEATURES = [
    "crime_risk",
    "lighting_level",
    "crowd_density",
    "traffic_level",
    "police_presence",
    "hour",
    "day_code"
]

TARGET = "risk_score"

df = pd.read_csv(DATASET)

df["day_code"] = df["day_of_week"].map(DAY_MAPPING)

required = FEATURES + [TARGET]
missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError("Missing columns: " + ", ".join(missing))

data = df[required].dropna().copy()

X = data[FEATURES]
y = data[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

predictions = np.clip(model.predict(X_test), 0, 100)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\\n========================================")
print(" RANDOM FOREST RISK MODEL")
print("========================================")
print(f"Total rows    : {len(data)}")
print(f"Training rows : {len(X_train)}")
print(f"Testing rows  : {len(X_test)}")
print(f"MAE           : {mae:.3f}")
print(f"RMSE          : {rmse:.3f}")
print(f"R2            : {r2:.3f}")

print("\\nFeature importance:")
for feature, value in sorted(
    zip(FEATURES, model.feature_importances_),
    key=lambda item: item[1],
    reverse=True
):
    print(f"{feature:20s}: {value:.4f}")

package = {
    "model": model,
    "features": FEATURES,
    "target": TARGET,
    "day_mapping": DAY_MAPPING,
    "dataset_type": "SYNTHETIC_ML_PROTOTYPE"
}

joblib.dump(package, MODEL_FILE)

print("\\nModel saved successfully:")
print(MODEL_FILE)
