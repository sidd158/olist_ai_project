import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


# ============================================================
# STEP 15: FINAL MODEL COMPARISON
# ============================================================

print("========== STEP 15: FINAL MODEL COMPARISON ==========")


# ============================================================
# 1. PATHS
# ============================================================

data_path = r"D:\olist_ai_project\data\processed"

logistic_path = (
    r"D:\olist_ai_project\models"
    r"\logistic_regression_baseline.pkl"
)

random_forest_path = (
    r"D:\olist_ai_project\models"
    r"\random_forest_baseline.pkl"
)

xgboost_path = (
    r"D:\olist_ai_project\models"
    r"\xgboost_model.pkl"
)


# ============================================================
# 2. LOAD TEST DATA
# ============================================================

print("\nLoading test data...")

X_test = pd.read_csv(
    data_path + r"\X_test.csv"
)

y_test = pd.read_csv(
    data_path + r"\y_test.csv"
).squeeze()


print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ============================================================
# 3. LOAD MODELS
# ============================================================

print("\nLoading models...")

logistic_model = joblib.load(
    logistic_path
)

random_forest_model = joblib.load(
    random_forest_path
)

xgboost_model = joblib.load(
    xgboost_path
)

print("All models loaded successfully!")


# ============================================================
# 4. CREATE MODEL LIST
# ============================================================

models = {
    "Logistic Regression": logistic_model,
    "Random Forest": random_forest_model,
    "XGBoost": xgboost_model
}


# ============================================================
# 5. EVALUATE MODELS
# ============================================================

results = []


for name, model in models.items():

    print(f"\nEvaluating {name}...")

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,

        "Accuracy": accuracy_score(
            y_test,
            y_pred
        ),

        "Precision": precision_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "F1": f1_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "ROC_AUC": roc_auc_score(
            y_test,
            y_prob
        ),

        "PR_AUC": average_precision_score(
            y_test,
            y_prob
        )
    })


# ============================================================
# 6. CREATE COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame(results)


print("\n========== FINAL MODEL COMPARISON ==========")

print(
    comparison.to_string(index=False)
)


# ============================================================
# 7. SAVE COMPARISON
# ============================================================

comparison_file = (
    data_path +
    r"\final_model_comparison.csv"
)

comparison.to_csv(
    comparison_file,
    index=False
)


# ============================================================
# 8. SELECT BEST MODEL
# ============================================================

best_index = comparison["PR_AUC"].idxmax()

best_model_name = comparison.loc[
    best_index,
    "Model"
]

best_pr_auc = comparison.loc[
    best_index,
    "PR_AUC"
]


print("\n========== BEST MODEL ==========")

print("Best Model:", best_model_name)

print("Best PR-AUC:", best_pr_auc)


# ============================================================
# 9. SAVE BEST MODEL NAME
# ============================================================

with open(
    data_path + r"\best_model.txt",
    "w"
) as file:

    file.write(best_model_name)


# ============================================================
# SUCCESS
# ============================================================

print("\n========== SUCCESS ==========")

print("Comparison saved:")
print(comparison_file)

print("\nBest model saved to:")
print(data_path + r"\best_model.txt")

print("\n========== STEP 15 COMPLETED SUCCESSFULLY ==========")