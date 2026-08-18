import pandas as pd
import joblib
import os

from xgboost import XGBClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)


# ============================================================
# STEP 14: XGBOOST MODEL
# ============================================================

print("========== STEP 14: XGBOOST MODEL ==========")


# ============================================================
# 1. PATHS
# ============================================================

data_path = r"D:\olist_ai_project\data\processed"
model_path = r"D:\olist_ai_project\models"

os.makedirs(model_path, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading training and test data...")

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


print("\n========== DATA ==========")

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# ============================================================
# 3. CALCULATE CLASS IMBALANCE
# ============================================================

class_0 = (y_train == 0).sum()
class_1 = (y_train == 1).sum()

scale_pos_weight = class_0 / class_1

print("\n========== CLASS INFORMATION ==========")

print("Class 0:", class_0)
print("Class 1:", class_1)

print("scale_pos_weight:", scale_pos_weight)


# ============================================================
# 4. CREATE XGBOOST MODEL
# ============================================================

print("\nCreating XGBoost model...")

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 5. TRAIN MODEL
# ============================================================

print("\n========== TRAINING ==========")

print("Training XGBoost...")
print("This may take some time...")

model.fit(
    X_train,
    y_train
)

print("XGBoost training completed!")


# ============================================================
# 6. PREDICTIONS
# ============================================================

print("\nMaking predictions...")

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]


# ============================================================
# 7. CLASSIFICATION REPORT
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
# 8. CONFUSION MATRIX
# ============================================================

print("\n========== CONFUSION MATRIX ==========")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


# ============================================================
# 9. ROC-AUC
# ============================================================

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

print("\n========== ROC-AUC ==========")

print("ROC-AUC:", roc_auc)


# ============================================================
# 10. PR-AUC
# ============================================================

pr_auc = average_precision_score(
    y_test,
    y_prob
)

print("\n========== PR-AUC ==========")

print("PR-AUC:", pr_auc)


# ============================================================
# 11. SAVE MODEL
# ============================================================

model_file = (
    model_path +
    r"\xgboost_model.pkl"
)

joblib.dump(
    model,
    model_file
)


# ============================================================
# 12. SUCCESS
# ============================================================

print("\n========== SUCCESS ==========")

print("Model saved at:")

print(model_file)

print("\n========== STEP 14 COMPLETED SUCCESSFULLY ==========")