import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression


# ============================================================
# STEP 10: TRAIN AND SAVE MODEL
# ============================================================

print("========== STEP 10: SAVE MODEL ==========")


# ============================================================
# 1. PATHS
# ============================================================

data_path = r"D:\olist_ai_project\data\processed"

model_path = r"D:\olist_ai_project\models"


# ============================================================
# 2. LOAD TRAINING DATA
# ============================================================

print("\nLoading training data...")

X_train = pd.read_csv(
    data_path + r"\X_train.csv"
)

y_train = pd.read_csv(
    data_path + r"\y_train.csv"
).squeeze()


print("X_train:", X_train.shape)
print("y_train:", y_train.shape)


# ============================================================
# 3. TRAIN MODEL
# ============================================================

print("\nTraining Logistic Regression...")

model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

print("Training completed!")


# ============================================================
# 4. CREATE MODEL DIRECTORY
# ============================================================

import os

os.makedirs(
    model_path,
    exist_ok=True
)


# ============================================================
# 5. SAVE MODEL
# ============================================================

model_file = model_path + r"\logistic_regression_baseline.pkl"

joblib.dump(
    model,
    model_file
)


# ============================================================
# 6. SUCCESS
# ============================================================

print("\n========== SUCCESS ==========")

print("Model saved at:")

print(model_file)

print("\n========== STEP 10 COMPLETED SUCCESSFULLY ==========")