import pandas as pd
import joblib

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)


# ============================================================
# STEP 11: TEST SAVED MODEL
# ============================================================

print("========== STEP 11: TEST SAVED MODEL ==========")


# ============================================================
# 1. PATHS
# ============================================================

data_path = r"D:\olist_ai_project\data\processed"

model_file = (
    r"D:\olist_ai_project\models"
    r"\logistic_regression_baseline.pkl"
)


# ============================================================
# 2. LOAD SAVED MODEL
# ============================================================

print("\nLoading saved model...")

model = joblib.load(model_file)

print("Model loaded successfully!")


# ============================================================
# 3. LOAD TEST DATA
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
# 4. MAKE PREDICTIONS
# ============================================================

print("\nMaking predictions...")

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]


# ============================================================
# 5. CLASSIFICATION REPORT
# ============================================================

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# 6. CONFUSION MATRIX
# ============================================================

print("\n========== CONFUSION MATRIX ==========")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


# ============================================================
# 7. ROC-AUC
# ============================================================

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

print("\n========== ROC-AUC ==========")

print("ROC-AUC:", roc_auc)


# ============================================================
# 8. PR-AUC
# ============================================================

pr_auc = average_precision_score(
    y_test,
    y_prob
)

print("\n========== PR-AUC ==========")

print("PR-AUC:", pr_auc)


# ============================================================
# 9. SAMPLE PREDICTIONS
# ============================================================

print("\n========== SAMPLE PREDICTIONS ==========")

results = pd.DataFrame({
    "actual": y_test.values,
    "predicted": y_pred,
    "probability_class_1": y_prob
})

print(
    results.head(10)
)


# ============================================================
# SUCCESS
# ============================================================

print("\n========== STEP 11 COMPLETED SUCCESSFULLY ==========")