from ml_risk_predictor import predict_risk


result = predict_risk(
    crime_risk=60,
    lighting_level=50,
    crowd_density=70,
    traffic_level=75,
    police_presence=40,
    hour=20,
    day_of_week="Saturday"
)


print("\n==============================")
print(" ML RISK PREDICTION")
print("==============================")

print(
    "Predicted Risk Score:",
    result["predicted_risk_score"],
    "/100"
)

print(
    "Predicted Risk Level:",
    result["predicted_risk_level"]
)

print("==============================")