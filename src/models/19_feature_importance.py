import pandas as pd
import joblib
import os

import matplotlib.pyplot as plt


# ============================================================
# STEP 19: FEATURE IMPORTANCE / EXPLAINABILITY
# ============================================================

print("========== STEP 19: FEATURE IMPORTANCE ==========")


# ============================================================
# 1. PATHS
# ============================================================

data_path = r"D:\olist_ai_project\data\processed"

model_file = (
    r"D:\olist_ai_project\models"
    r"\final_model.pkl"
)

feature_file = (
    data_path +
    r"\feature_names.csv"
)

output_csv = (
    data_path +
    r"\feature_importance.csv"
)

output_image = (
    data_path +
    r"\feature_importance.png"
)


# ============================================================
# 2. LOAD MODEL
# ============================================================

print("\nLoading final model...")

model = joblib.load(model_file)

print("Model loaded successfully!")

print("Model type:", type(model).__name__)


# ============================================================
# 3. LOAD FEATURE NAMES
# ============================================================

print("\nLoading feature names...")

feature_df = pd.read_csv(feature_file)

print("Feature names loaded!")

print("Feature file shape:", feature_df.shape)


# ============================================================
# 4. GET FEATURE NAMES
# ============================================================

if feature_df.shape[1] == 1:

    feature_names = feature_df.iloc[:, 0].astype(str).tolist()

else:

    # Use the first column if multiple columns exist
    feature_names = feature_df.iloc[:, 0].astype(str).tolist()


print(
    "Number of feature names:",
    len(feature_names)
)


# ============================================================
# 5. GET FEATURE IMPORTANCE
# ============================================================

if hasattr(model, "feature_importances_"):

    print("\nUsing tree-based feature importance...")

    importance = model.feature_importances_


elif hasattr(model, "coef_"):

    print("\nUsing Logistic Regression coefficients...")

    importance = abs(model.coef_[0])


else:

    raise ValueError(
        "This model does not support feature importance."
    )


# ============================================================
# 6. CHECK FEATURE COUNT
# ============================================================

print("\nModel features:", len(importance))
print("Feature names:", len(feature_names))


if len(importance) != len(feature_names):

    raise ValueError(
        "Feature count does not match feature names."
    )


# ============================================================
# 7. CREATE FEATURE IMPORTANCE DATAFRAME
# ============================================================

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importance
})


# ============================================================
# 8. SORT FEATURES
# ============================================================

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)


# ============================================================
# 9. SAVE CSV
# ============================================================

importance_df.to_csv(
    output_csv,
    index=False
)


# ============================================================
# 10. DISPLAY TOP 20 FEATURES
# ============================================================

print("\n========== TOP 20 IMPORTANT FEATURES ==========")

print(
    importance_df.head(20).to_string(index=False)
)


# ============================================================
# 11. CREATE VISUALIZATION
# ============================================================

top_features = importance_df.head(20)

plt.figure(figsize=(10, 8))

plt.barh(
    top_features["feature"][::-1],
    top_features["importance"][::-1]
)

plt.xlabel("Importance")

plt.ylabel("Feature")

plt.title(
    "Top 20 Features Influencing Customer Inactivity"
)

plt.tight_layout()

plt.savefig(
    output_image,
    dpi=300
)

plt.show()


# ============================================================
# SUCCESS
# ============================================================

print("\n========== SUCCESS ==========")

print("Feature importance saved:")

print(output_csv)

print("\nVisualization saved:")

print(output_image)

print("\n========== STEP 19 COMPLETED SUCCESSFULLY ==========")