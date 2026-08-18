import pandas as pd
import joblib


# ============================================================
# STEP 18: BATCH CUSTOMER PREDICTION
# ============================================================

print("========== STEP 18: BATCH PREDICTION ==========")


# ============================================================
# 1. PATHS
# ============================================================

data_path = r"D:\olist_ai_project\data\processed"

model_file = (
    r"D:\olist_ai_project\models"
    r"\final_model.pkl"
)

output_file = (
    data_path +
    r"\customer_predictions.csv"
)


# ============================================================
# 2. LOAD FINAL MODEL
# ============================================================

print("\nLoading final model...")

model = joblib.load(model_file)

print("Final model loaded successfully!")


# ============================================================
# 3. LOAD TEST DATA
# ============================================================

print("\nLoading customer data...")

X_test = pd.read_csv(
    data_path + r"\X_test.csv"
)

y_test = pd.read_csv(
    data_path + r"\y_test.csv"
).squeeze()


print("Customers:", len(X_test))
print("Features:", X_test.shape[1])


# ============================================================
# 4. MAKE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)[:, 1]


# ============================================================
# 5. CREATE RESULT DATAFRAME
# ============================================================

results = pd.DataFrame({
    "customer_index": range(len(X_test)),
    "actual_inactive_90d": y_test.values,
    "predicted_inactive_90d": predictions,
    "inactive_probability": probabilities
})


# ============================================================
# 6. CONVERT PROBABILITY TO PERCENTAGE
# ============================================================

results["inactive_probability_percent"] = (
    results["inactive_probability"] * 100
).round(2)


# ============================================================
# 7. CREATE RISK LEVEL
# ============================================================

def get_risk(probability):

    if probability >= 0.80:
        return "HIGH"

    elif probability >= 0.50:
        return "MEDIUM"

    else:
        return "LOW"


results["risk_level"] = results[
    "inactive_probability"
].apply(get_risk)


# ============================================================
# 8. SORT BY RISK
# ============================================================

results = results.sort_values(
    "inactive_probability",
    ascending=False
)


# ============================================================
# 9. SAVE RESULTS
# ============================================================

results.to_csv(
    output_file,
    index=False
)


# ============================================================
# 10. DISPLAY SUMMARY
# ============================================================

print("\n========== PREDICTION SUMMARY ==========")

print(
    "Total customers:",
    len(results)
)

print(
    "\nRisk distribution:"
)

print(
    results["risk_level"].value_counts()
)


print(
    "\nPredicted inactive customers:"
)

print(
    (results["predicted_inactive_90d"] == 1).sum()
)


# ============================================================
# 11. SHOW TOP CUSTOMERS
# ============================================================

print("\n========== TOP 10 HIGH-RISK CUSTOMERS ==========")

print(
    results[
        [
            "customer_index",
            "predicted_inactive_90d",
            "inactive_probability_percent",
            "risk_level"
        ]
    ].head(10).to_string(index=False)
)


# ============================================================
# SUCCESS
# ============================================================

print("\n========== SUCCESS ==========")

print("Prediction file saved:")

print(output_file)

print("\n========== STEP 18 COMPLETED SUCCESSFULLY ==========")