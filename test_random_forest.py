import joblib
import pandas as pd

# Load trained model
model_package = joblib.load("random_forest_risk_model.pkl")

model = model_package["model"]
features = model_package["features"]
day_mapping = model_package["day_mapping"]

# New unseen safety situation
new_data = pd.DataFrame([{
    "crime_risk": 60,
    "lighting_level": 50,
    "crowd_density": 70,
    "traffic_level": 75,
    "police_presence": 40,
    "hour": 20,
    "day_code": day_mapping["Saturday"]
}])

# Predict risk
predicted_risk = model.predict(
    new_data[features]
)[0]

predicted_risk = max(
    0,
    min(100, predicted_risk)
)

# Convert numerical score to risk level
if predicted_risk >= 65:
    risk_level = "High"
elif predicted_risk >= 40:
    risk_level = "Medium"
else:
    risk_level = "Low"

print("\n==============================")
print(" RANDOM FOREST PREDICTION")
print("==============================")
print(f"Predicted Risk Score : {predicted_risk:.2f}/100")
print(f"Predicted Risk Level : {risk_level}")
print("==============================")