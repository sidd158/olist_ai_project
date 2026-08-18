import os
import json
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# STEP 35: FINAL MODEL EVALUATION
# ============================================================

print("=" * 60)
print("STEP 35: FINAL MODEL EVALUATION")
print("=" * 60)


# ============================================================
# PATHS
# ============================================================

BASE_PATH = r"D:\olist_ai_project"

DATA_PATH = os.path.join(
    BASE_PATH,
    "data",
    "processed"
)

MODEL_PATH = os.path.join(
    BASE_PATH,
    "models",
    "final_model.pkl"
)

OUTPUT_PATH = os.path.join(
    DATA_PATH,
    "model_performance.json"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading final model...")

model = joblib.load(MODEL_PATH)

print("Final model loaded successfully!")


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test data...")

X_test = pd.read_csv(
    os.path.join(
        DATA_PATH,
        "X_test.csv"
    )
)

y_test = pd.read_csv(
    os.path.join(
        DATA_PATH,
        "y_test.csv"
    )
)


# ============================================================
# FIX TARGET FORMAT
# ============================================================

if isinstance(y_test, pd.DataFrame):

    if "inactive_90d" in y_test.columns:
        y_test = y_test["inactive_90d"]

    else:
        y_test = y_test.iloc[:, 0]


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_pred = model.predict(
    X_test
)

y_probability = model.predict_proba(
    X_test
)[:, 1]


print("Predictions completed!")


# ============================================================
# METRICS
# ============================================================

print("\nCalculating metrics...")


accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    y_pred,
    zero_division=0,
    output_dict=True
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(
    f"\nAccuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)


print("\nConfusion Matrix:")
print(cm)


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

performance = {

    "accuracy":
        round(
            float(accuracy),
            6
        ),

    "precision":
        round(
            float(precision),
            6
        ),

    "recall":
        round(
            float(recall),
            6
        ),

    "f1_score":
        round(
            float(f1),
            6
        ),

    "roc_auc":
        round(
            float(roc_auc),
            6
        ),

    "confusion_matrix":
        cm.tolist(),

    "classification_report":
        report
}


with open(
    OUTPUT_PATH,
    "w"
) as file:

    json.dump(
        performance,
        file,
        indent=4
    )


print("\nPerformance saved to:")

print(
    OUTPUT_PATH
)


print("\n" + "=" * 60)
print("STEP 35 COMPLETED SUCCESSFULLY")
print("=" * 60)