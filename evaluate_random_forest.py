import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET = "safety_ml_dataset.csv"
MODEL_FILE = "random_forest_risk_model.pkl"

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# LOAD DATA + TRAINED MODEL
# ============================================================

df = pd.read_csv(DATASET)

model_package = joblib.load(
    MODEL_FILE
)

model = model_package["model"]
features = model_package["features"]
day_mapping = model_package["day_mapping"]


# ============================================================
# SAME PREPROCESSING USED BY THE MODEL
# ============================================================

df["day_code"] = (
    df["day_of_week"]
    .map(day_mapping)
)

if df["day_code"].isna().any():

    unknown_days = (
        df.loc[
            df["day_code"].isna(),
            "day_of_week"
        ]
        .unique()
        .tolist()
    )

    raise ValueError(
        "Unknown day_of_week values: "
        + str(unknown_days)
    )


X = df[features]
y = df["risk_score"]


# ============================================================
# RECREATE THE SAME 240 / 60 SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)


# ============================================================
# PREDICT ONLY ON THE HELD-OUT TEST SET
# ============================================================

predictions = model.predict(
    X_test
)

predictions = np.clip(
    predictions,
    0,
    100
)


# ============================================================
# METRICS
# ============================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)


print()
print("=" * 60)
print(" RANDOM FOREST - HELD-OUT TEST EVALUATION")
print("=" * 60)

print(
    f"Total rows     : {len(df)}"
)

print(
    f"Training rows  : {len(X_train)}"
)

print(
    f"Testing rows   : {len(X_test)}"
)

print(
    f"MAE            : {mae:.3f}"
)

print(
    f"RMSE           : {rmse:.3f}"
)

print(
    f"R2             : {r2:.3f}"
)

print("=" * 60)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({

    "feature":
        features,

    "importance":
        model.feature_importances_

})

importance = importance.sort_values(
    "importance",
    ascending=False
)


print()
print("FEATURE IMPORTANCE")
print("-" * 45)

for _, row in importance.iterrows():

    print(
        f"{row['feature']:<20}"
        f": {row['importance']:.4f}"
    )


# ============================================================
# ACTUAL VS PREDICTED - TEST SET ONLY
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    y_test,
    predictions,
    alpha=0.75
)

plt.plot(
    [0, 100],
    [0, 100],
    linestyle="--"
)

plt.xlabel(
    "Actual Risk Score"
)

plt.ylabel(
    "Predicted Risk Score"
)

plt.title(
    "Random Forest - Test Set: Actual vs Predicted"
)

plt.xlim(
    0,
    100
)

plt.ylim(
    0,
    100
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "test_actual_vs_predicted.png",
    dpi=300
)

plt.show()


# ============================================================
# TEST-SET PREDICTION RESULTS
# ============================================================

results = pd.DataFrame({

    "actual_risk":
        y_test.values,

    "predicted_risk":
        np.round(
            predictions,
            2
        )

})

results["error"] = np.round(
    results["actual_risk"]
    -
    results["predicted_risk"],
    2
)


print()
print("HELD-OUT TEST PREDICTIONS")
print("-" * 60)

print(
    results.head(15).to_string(
        index=False
    )
)


results.to_csv(
    "random_forest_test_results.csv",
    index=False
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = pd.DataFrame([{

    "total_rows":
        len(df),

    "training_rows":
        len(X_train),

    "testing_rows":
        len(X_test),

    "MAE":
        round(mae, 3),

    "RMSE":
        round(rmse, 3),

    "R2":
        round(r2, 3)

}])

metrics.to_csv(
    "random_forest_test_metrics.csv",
    index=False
)


print()
print("Evaluation completed successfully.")

print()
print("Generated files:")

print(
    "1. test_actual_vs_predicted.png"
)

print(
    "2. random_forest_test_results.csv"
)

print(
    "3. random_forest_test_metrics.csv"
)
