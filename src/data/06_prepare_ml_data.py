from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# STEP 6 - EDA + ML FEATURE PREPARATION
# ============================================================

print("STEP 6 STARTED")


# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DIR = BASE_DIR / "data" / "processed"

print("\nProcessed data folder:")
print(PROCESSED_DIR)


# ------------------------------------------------------------
# 2. INPUT FILE
# ------------------------------------------------------------

input_file = (
    PROCESSED_DIR
    / "customer_ml_dataset.csv"
)


if not input_file.exists():

    raise FileNotFoundError(
        f"Missing file: {input_file}\n"
        "Run Step 5 first."
    )


# ------------------------------------------------------------
# 3. LOAD DATA
# ------------------------------------------------------------

print("\nLoading ML dataset...")

df = pd.read_csv(input_file)

print("Dataset loaded.")


# ------------------------------------------------------------
# 4. BASIC INFORMATION
# ------------------------------------------------------------

print("\n========== BASIC INFORMATION ==========")

print("Rows:", len(df))

print("Columns:", len(df.columns))

print("\nColumns:")

for column in df.columns:
    print("-", column)


# ------------------------------------------------------------
# 5. TARGET INFORMATION
# ------------------------------------------------------------

print("\n========== TARGET ==========")

print(
    df["inactive_90d"]
    .value_counts()
)

print("\nTarget percentage:")

target_percentage = (
    df["inactive_90d"]
    .value_counts(normalize=True)
    * 100
)

print(target_percentage)


# ------------------------------------------------------------
# 6. MISSING VALUES
# ------------------------------------------------------------

print("\n========== MISSING VALUES ==========")

missing = (
    df.isna()
    .sum()
    .sort_values(ascending=False)
)

missing_percentage = (
    df.isna()
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

missing_table = pd.DataFrame({
    "missing_count": missing,
    "missing_percentage": missing_percentage
})

print(missing_table)


# ------------------------------------------------------------
# 7. DUPLICATES
# ------------------------------------------------------------

print("\n========== DUPLICATES ==========")

duplicate_count = (
    df["customer_unique_id"]
    .duplicated()
    .sum()
)

print(
    "Duplicate customers:",
    duplicate_count
)


# ------------------------------------------------------------
# 8. NUMERIC SUMMARY
# ------------------------------------------------------------

print("\n========== NUMERIC SUMMARY ==========")

numeric_columns = df.select_dtypes(
    include=np.number
).columns

print(
    df[numeric_columns]
    .describe()
    .T
)


# ------------------------------------------------------------
# 9. CHECK EXTREME VALUES
# ------------------------------------------------------------

print("\n========== EXTREME VALUES ==========")

important_numeric = [
    "historical_orders",
    "historical_spend",
    "historical_average_order_value",
    "historical_items",
    "historical_recency_days",
    "historical_average_review",
    "historical_average_delivery_delay",
    "historical_late_rate"
]


for column in important_numeric:

    if column in df.columns:

        print(
            f"\n{column}"
        )

        print(
            "Minimum:",
            df[column].min()
        )

        print(
            "Maximum:",
            df[column].max()
        )

        print(
            "Median:",
            df[column].median()
        )


# ------------------------------------------------------------
# 10. CORRELATION WITH TARGET
# ------------------------------------------------------------

print("\n========== TARGET CORRELATION ==========")

correlation_columns = [
    column
    for column in numeric_columns
    if column != "inactive_90d"
    and not column.startswith("future_")
]

correlation = (
    df[
        correlation_columns
        + ["inactive_90d"]
    ]
    .corr()["inactive_90d"]
    .sort_values()
)

print(correlation)


# ------------------------------------------------------------
# 11. REMOVE FUTURE INFORMATION
# ------------------------------------------------------------

print("\n========== REMOVING FUTURE INFORMATION ==========")

future_columns = [
    "future_orders",
    "future_spend",
    "inactive_90d",
    "activity_label"
]


# Only keep columns that actually exist

future_columns = [
    column
    for column in future_columns
    if column in df.columns
]


print(
    "Future/target columns:",
    future_columns
)


# ------------------------------------------------------------
# 12. DEFINE TARGET
# ------------------------------------------------------------

y = df["inactive_90d"].copy()


# ------------------------------------------------------------
# 13. DEFINE FEATURES
# ------------------------------------------------------------

X = df.drop(
    columns=future_columns
)


# ------------------------------------------------------------
# 14. REMOVE IDENTIFIERS
# ------------------------------------------------------------

identifier_columns = [
    "customer_unique_id",
    "historical_last_purchase",
    "first_purchase_date",
    "last_purchase_date"
]


identifier_columns = [
    column
    for column in identifier_columns
    if column in X.columns
]


print(
    "\nRemoving identifiers:"
)

print(
    identifier_columns
)


X = X.drop(
    columns=identifier_columns
)


# ------------------------------------------------------------
# 15. REMOVE NON-ML DATE COLUMNS
# ------------------------------------------------------------

date_columns = X.select_dtypes(
    include=["datetime64[ns]"]
).columns.tolist()


if len(date_columns) > 0:

    print(
        "\nRemoving date columns:"
    )

    print(date_columns)

    X = X.drop(
        columns=date_columns
    )


# ------------------------------------------------------------
# 16. HANDLE MISSING NUMERIC VALUES
# ------------------------------------------------------------

numeric_features = X.select_dtypes(
    include=np.number
).columns


for column in numeric_features:

    X[column] = X[column].fillna(
        X[column].median()
    )


# ------------------------------------------------------------
# 17. HANDLE MISSING CATEGORICAL VALUES
# ------------------------------------------------------------

categorical_features = X.select_dtypes(
    exclude=np.number
).columns


for column in categorical_features:

    X[column] = X[column].fillna(
        "Unknown"
    )


# ------------------------------------------------------------
# 18. ONE-HOT ENCODE CATEGORICAL FEATURES
# ------------------------------------------------------------

print(
    "\nEncoding categorical features..."
)


X = pd.get_dummies(
    X,
    columns=categorical_features,
    drop_first=True,
    dtype=int
)


# ------------------------------------------------------------
# 19. FINAL CHECK
# ------------------------------------------------------------

print("\n========== FINAL ML DATA ==========")

print(
    "X rows:",
    len(X)
)

print(
    "X columns:",
    len(X.columns)
)

print(
    "Target rows:",
    len(y)
)

print(
    "Missing values in X:",
    X.isna().sum().sum()
)

print(
    "Target missing values:",
    y.isna().sum()
)


# ------------------------------------------------------------
# 20. TARGET DISTRIBUTION
# ------------------------------------------------------------

print(
    "\n========== TARGET DISTRIBUTION =========="
)

print(
    y.value_counts()
)

print(
    "\nTarget percentages:"
)

print(
    y.value_counts(
        normalize=True
    ) * 100
)


# ------------------------------------------------------------
# 21. SAVE FEATURES
# ------------------------------------------------------------

X_file = (
    PROCESSED_DIR
    / "X_ml.csv"
)

y_file = (
    PROCESSED_DIR
    / "y_ml.csv"
)


X.to_csv(
    X_file,
    index=False
)


y.to_csv(
    y_file,
    index=False,
    header=["inactive_90d"]
)


# ------------------------------------------------------------
# 22. SAVE FEATURE NAMES
# ------------------------------------------------------------

feature_names = pd.DataFrame({
    "feature": X.columns
})


feature_names.to_csv(
    PROCESSED_DIR
    / "feature_names.csv",
    index=False
)


# ------------------------------------------------------------
# 23. FINAL MESSAGE
# ------------------------------------------------------------

print("\n========== SUCCESS ==========")

print(
    "Saved:"
)

print(
    X_file
)

print(
    y_file
)

print(
    PROCESSED_DIR
    / "feature_names.csv"
)

print(
    "\nSTEP 6 COMPLETED SUCCESSFULLY!"
)