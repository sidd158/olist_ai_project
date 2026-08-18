import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)


# ============================================================
# STEP 8: BASELINE LOGISTIC REGRESSION MODEL
# ============================================================

print("========== STEP 8: BASELINE MODEL ==========")


# ============================================================
# 1. LOAD TRAINING DATA
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


print("\n========== DATA LOADED ==========")

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# ============================================================
# 2. CREATE LOGISTIC REGRESSION MODEL
# ============================================================

print("\n========== CREATING MODEL ==========")

model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)


# ============================================================
# 3. TRAIN MODEL
# ============================================================

print("\nTraining Logistic Regression...")

model.fit(
    X_train,
    y_train
)

print("Training completed successfully!")


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
# 9. SUCCESS
# ============================================================

print("\n========== STEP 8 COMPLETED SUCCESSFULLY ==========")