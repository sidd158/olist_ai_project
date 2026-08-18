import pandas as pd
import joblib


# ============================================================
# STEP 17: CUSTOMER PREDICTION
# ============================================================

print("========== STEP 17: CUSTOMER PREDICTION ==========")


# ============================================================
# 1. PATHS
# ============================================================

data_path = r"D:\olist_ai_project\data\processed"

model_file = (
    r"D:\olist_ai_project\models"
    r"\final_model.pkl"
)


# ============================================================
# 2. LOAD FINAL MODEL
# ============================================================

print("\nLoading final model...")

model = joblib.load(model_file)

print("Final model loaded successfully!")


# ============================================================
# 3. LOAD TEST CUSTOMER DATA
# ============================================================

print("\nLoading customer data...")

X_test = pd.read_csv(
    data_path + r"\X_test.csv"
)

y_test = pd.read_csv(
    data_path + r"\y_test.csv"
).squeeze()


print("Customers loaded:", len(X_test))
print("Features:", X_test.shape[1])


# ============================================================
# 4. SELECT ONE CUSTOMER
# ============================================================

customer_index = 0

customer = X_test.iloc[
    customer_index:customer_index + 1
]


# ============================================================
# 5. MAKE PREDICTION
# ============================================================

prediction = model.predict(
    customer
)[0]

probability = model.predict_proba(
    customer
)[0][1]


# ============================================================
# 6. DISPLAY RESULT
# ============================================================

print("\n========== CUSTOMER PREDICTION ==========")

print("Customer index:", customer_index)

print(
    "Inactive prediction:",
    prediction
)

print(
    "Inactive probability:",
    round(probability * 100, 2),
    "%"
)


# ============================================================
# 7. INTERPRET RESULT
# ============================================================

if prediction == 1:

    print(
        "\nPrediction: CUSTOMER IS LIKELY INACTIVE"
    )

else:

    print(
        "\nPrediction: CUSTOMER IS LIKELY ACTIVE"
    )


# ============================================================
# 8. RISK LEVEL
# ============================================================

if probability >= 0.80:

    risk = "HIGH"

elif probability >= 0.50:

    risk = "MEDIUM"

else:

    risk = "LOW"


print(
    "Risk Level:",
    risk
)


# ============================================================
# SUCCESS
# ============================================================

print(
    "\n========== STEP 17 COMPLETED SUCCESSFULLY =========="
)