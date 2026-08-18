import pandas as pd
import joblib


# ============================================================
# STEP 20: CUSTOMER AI EXPLANATION
# ============================================================

print("========== STEP 20: AI CUSTOMER EXPLANATION ==========")


# ============================================================
# 1. PATHS
# ============================================================

data_path = r"D:\olist_ai_project\data\processed"

model_file = (
    r"D:\olist_ai_project\models"
    r"\final_model.pkl"
)

prediction_file = (
    data_path +
    r"\customer_predictions.csv"
)

importance_file = (
    data_path +
    r"\feature_importance.csv"
)


# ============================================================
# 2. LOAD MODEL
# ============================================================

print("\nLoading final model...")

model = joblib.load(model_file)

print("Final model loaded successfully!")


# ============================================================
# 3. LOAD PREDICTIONS
# ============================================================

print("\nLoading customer predictions...")

predictions = pd.read_csv(
    prediction_file
)

print(
    "Customers loaded:",
    len(predictions)
)


# ============================================================
# 4. LOAD FEATURE IMPORTANCE
# ============================================================

print("\nLoading feature importance...")

importance_df = pd.read_csv(
    importance_file
)

print(
    "Features loaded:",
    len(importance_df)
)


# ============================================================
# 5. SELECT HIGH-RISK CUSTOMERS
# ============================================================

high_risk = predictions[
    predictions["risk_level"] == "HIGH"
].copy()


print("\n========== RISK SUMMARY ==========")

print(
    predictions["risk_level"].value_counts()
)

print(
    "\nHigh-risk customers:",
    len(high_risk)
)


# ============================================================
# 6. CUSTOMER EXPLANATION FUNCTION
# ============================================================

def generate_explanation(customer):

    probability = customer[
        "inactive_probability"
    ]

    risk = customer[
        "risk_level"
    ]

    # --------------------------------------------------------
    # Risk description
    # --------------------------------------------------------

    if risk == "HIGH":

        risk_text = (
            "This customer has a high probability "
            "of becoming inactive within 90 days."
        )

        recommendation = (
            "Send a personalized retention offer, "
            "discount, or targeted re-engagement campaign."
        )

    elif risk == "MEDIUM":

        risk_text = (
            "This customer has a moderate probability "
            "of becoming inactive within 90 days."
        )

        recommendation = (
            "Send a personalized reminder or "
            "product recommendation."
        )

    else:

        risk_text = (
            "This customer currently has a low probability "
            "of becoming inactive within 90 days."
        )

        recommendation = (
            "Maintain regular engagement and "
            "personalized recommendations."
        )


    # --------------------------------------------------------
    # Explanation
    # --------------------------------------------------------

    explanation = (
        f"Risk Level: {risk}\n"
        f"Inactive Probability: {probability * 100:.2f}%\n\n"
        f"Analysis:\n"
        f"{risk_text}\n\n"
        f"Recommended Action:\n"
        f"{recommendation}"
    )

    return explanation


# ============================================================
# 7. SHOW SAMPLE HIGH-RISK CUSTOMERS
# ============================================================

print(
    "\n========== CUSTOMER AI EXPLANATIONS =========="
)


sample_customers = high_risk.head(5)


for _, customer in sample_customers.iterrows():

    print("\n----------------------------------------")

    print(
        "Customer:",
        customer["customer_index"]
    )

    print(
        generate_explanation(customer)
    )


# ============================================================
# 8. CREATE EXPLANATION DATA
# ============================================================

explanation_results = []


for _, customer in predictions.iterrows():

    explanation = generate_explanation(
        customer
    )

    explanation_results.append({
        "customer_index":
            customer["customer_index"],

        "risk_level":
            customer["risk_level"],

        "inactive_probability":
            customer["inactive_probability"],

        "explanation":
            explanation
    })


explanation_df = pd.DataFrame(
    explanation_results
)


# ============================================================
# 9. SAVE EXPLANATIONS
# ============================================================

output_file = (
    data_path +
    r"\customer_ai_explanations.csv"
)

explanation_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# SUCCESS
# ============================================================

print("\n========== SUCCESS ==========")

print(
    "AI explanations saved:"
)

print(output_file)

print(
    "\n========== STEP 20 COMPLETED SUCCESSFULLY =========="
)