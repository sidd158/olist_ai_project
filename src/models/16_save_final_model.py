import pandas as pd
import joblib
import shutil
import os


# ============================================================
# STEP 16: SAVE FINAL MODEL
# ============================================================

print("========== STEP 16: SAVE FINAL MODEL ==========")


# ============================================================
# 1. PATHS
# ============================================================

data_path = r"D:\olist_ai_project\data\processed"

model_path = r"D:\olist_ai_project\models"

best_model_file = data_path + r"\best_model.txt"

final_model_file = model_path + r"\final_model.pkl"


# ============================================================
# 2. READ BEST MODEL NAME
# ============================================================

if not os.path.exists(best_model_file):
    raise FileNotFoundError(
        "best_model.txt not found. Run Step 15 first."
    )


with open(best_model_file, "r") as file:
    best_model_name = file.read().strip()


print("\nBest model selected:")
print(best_model_name)


# ============================================================
# 3. SELECT MODEL FILE
# ============================================================

if best_model_name == "Logistic Regression":

    source_model = (
        model_path +
        r"\logistic_regression_baseline.pkl"
    )

elif best_model_name == "Random Forest":

    source_model = (
        model_path +
        r"\random_forest_baseline.pkl"
    )

elif best_model_name == "XGBoost":

    source_model = (
        model_path +
        r"\xgboost_model.pkl"
    )

else:

    raise ValueError(
        "Unknown model name: " + best_model_name
    )


# ============================================================
# 4. CHECK SOURCE MODEL
# ============================================================

if not os.path.exists(source_model):

    raise FileNotFoundError(
        f"Model file not found: {source_model}"
    )


print("\nSource model:")
print(source_model)


# ============================================================
# 5. CREATE MODELS DIRECTORY
# ============================================================

os.makedirs(
    model_path,
    exist_ok=True
)


# ============================================================
# 6. COPY BEST MODEL
# ============================================================

shutil.copy2(
    source_model,
    final_model_file
)


# ============================================================
# 7. VERIFY FINAL MODEL
# ============================================================

final_model = joblib.load(
    final_model_file
)


print("\nFinal model loaded successfully!")


# ============================================================
# 8. SAVE MODEL INFORMATION
# ============================================================

model_info = pd.DataFrame({
    "final_model": [best_model_name],
    "model_file": [final_model_file]
})

model_info.to_csv(
    data_path + r"\final_model_info.csv",
    index=False
)


# ============================================================
# 9. SUCCESS
# ============================================================

print("\n========== SUCCESS ==========")

print("Final model:")
print(final_model_file)

print("\nModel information:")
print(data_path + r"\final_model_info.csv")

print("\n========== STEP 16 COMPLETED SUCCESSFULLY ==========")