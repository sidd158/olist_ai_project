import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


# ============================================================
# STEP 13: MODEL COMPARISON
# ============================================================

print("========== STEP 13: MODEL COMPARISON ==========")


# ============================================================
# 1. LOAD DATA
# ============================================================

data_path = r"D:\olist_ai_project\data\processed"

X_train = pd.read_csv(
    data_path + r"\X_train.csv"
)

X_test = pd.read_csv(
    data_path + r"\X_test.csv"
)

y_train = pd.read_csv(
    data_path + r"\y_train.csv"
).squeeze()

y_test = pd.read_csv(
    data_path + r"\y_test.csv"
).squeeze()


print("\nData loaded successfully!")


# ============================================================
# 2. LOGISTIC REGRESSION
# ============================================================

print("\nTraining Logistic Regression...")

logistic_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

logistic_model.fit(
    X_train,
    y_train
)

logistic_pred = logistic_model.predict(X_test)

logistic_prob = logistic_model.predict_proba(X_test)[:, 1]

print("Logistic Regression completed!")


# ============================================================
# 3. RANDOM FOREST
# ============================================================

print("\nTraining Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(
    X_train,
    y_train
)

rf_pred = rf_model.predict(X_test)

rf_prob = rf_model.predict_proba(X_test)[:, 1]

print("Random Forest completed!")


# ============================================================
# 4. CALCULATE METRICS
# ============================================================

results = []


# Logistic Regression

results.append({
    "Model": "Logistic Regression",
    "Accuracy": accuracy_score(y_test, logistic_pred),
    "Precision": precision_score(
        y_test,
        logistic_pred,
        zero_division=0
    ),
    "Recall": recall_score(
        y_test,
        logistic_pred,
        zero_division=0
    ),
    "F1": f1_score(
        y_test,
        logistic_pred,
        zero_division=0
    ),
    "ROC_AUC": roc_auc_score(
        y_test,
        logistic_prob
    ),
    "PR_AUC": average_precision_score(
        y_test,
        logistic_prob
    )
})


# Random Forest

results.append({
    "Model": "Random Forest",
    "Accuracy": accuracy_score(y_test, rf_pred),
    "Precision": precision_score(
        y_test,
        rf_pred,
        zero_division=0
    ),
    "Recall": recall_score(
        y_test,
        rf_pred,
        zero_division=0
    ),
    "F1": f1_score(
        y_test,
        rf_pred,
        zero_division=0
    ),
    "ROC_AUC": roc_auc_score(
        y_test,
        rf_prob
    ),
    "PR_AUC": average_precision_score(
        y_test,
        rf_prob
    )
})


# ============================================================
# 5. DISPLAY RESULTS
# ============================================================

comparison = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")

print(
    comparison.to_string(index=False)
)


# ============================================================
# 6. SAVE RESULTS
# ============================================================

comparison.to_csv(
    data_path + r"\model_comparison.csv",
    index=False
)


print("\n========== SUCCESS ==========")

print(
    "Saved:",
    data_path + r"\model_comparison.csv"
)

print("\n========== STEP 13 COMPLETED SUCCESSFULLY ==========")