import pandas as pd
import joblib
import os

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)


# ============================================================
# STEP 12: RANDOM FOREST MODEL
# ============================================================

print("========== STEP 12: RANDOM FOREST ==========")


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
# 3. CREATE RANDOM FOREST
# ============================================================

print("\nCreating Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 4. TRAIN MODEL
# ============================================================

print("\n========== TRAINING ==========")

print("Training Random Forest...")
print("This may take some time...")

model.fit(
    X_train,
    y_train
)

print("Training completed successfully!")


# ============================================================
# 5. PREDICTIONS
# ============================================================

print("\nMaking predictions...")

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]


# ============================================================
# 6. CLASSIFICATION REPORT
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
# 7. CONFUSION MATRIX
# ============================================================

print("\n========== CONFUSION MATRIX ==========")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


# ============================================================
# 8. ROC-AUC
# ============================================================

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

print("\n========== ROC-AUC ==========")

print("ROC-AUC:", roc_auc)


# ============================================================
# 9. PR-AUC
# ============================================================

pr_auc = average_precision_score(
    y_test,
    y_prob
)

print("\n========== PR-AUC ==========")

print("PR-AUC:", pr_auc)


# ============================================================
# 10. SAVE MODEL
# ============================================================

model_file = (
    model_path +
    r"\random_forest_baseline.pkl"
)

joblib.dump(
    model,
    model_file
)


# ============================================================
# SUCCESS
# ============================================================

print("\n========== SUCCESS ==========")

print("Random Forest saved at:")

print(model_file)

print("\n========== STEP 12 COMPLETED SUCCESSFULLY ==========")