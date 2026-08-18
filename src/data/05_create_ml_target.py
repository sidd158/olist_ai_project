from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# STEP 5 - CREATE ML TARGET
# ============================================================

print("STEP 5 STARTED")


# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DIR = BASE_DIR / "data" / "processed"

print("\nProcessed data folder:")
print(PROCESSED_DIR)


# ------------------------------------------------------------
# 2. CHECK INPUT FILES
# ------------------------------------------------------------

customer_file = PROCESSED_DIR / "customer_features.csv"
order_file = PROCESSED_DIR / "order_level.csv"

if not customer_file.exists():
    raise FileNotFoundError(
        f"Missing: {customer_file}\n"
        "Run Step 4 first."
    )

if not order_file.exists():
    raise FileNotFoundError(
        f"Missing: {order_file}\n"
        "Run Step 3 first."
    )

print("\nRequired files found.")


# ------------------------------------------------------------
# 3. LOAD DATA
# ------------------------------------------------------------

customers = pd.read_csv(customer_file)

orders = pd.read_csv(order_file)

print("\nCustomer rows:", len(customers))
print("Order rows:", len(orders))


# ------------------------------------------------------------
# 4. CONVERT DATES
# ------------------------------------------------------------

customers["first_purchase_date"] = pd.to_datetime(
    customers["first_purchase_date"],
    errors="coerce"
)

customers["last_purchase_date"] = pd.to_datetime(
    customers["last_purchase_date"],
    errors="coerce"
)

orders["order_purchase_timestamp"] = pd.to_datetime(
    orders["order_purchase_timestamp"],
    errors="coerce"
)


# ============================================================
# PART A
# CREATE A TIME-BASED PREDICTION SET
# ============================================================

print("\nCreating prediction cutoff...")


# ------------------------------------------------------------
# 5. FIND DATASET END DATE
# ------------------------------------------------------------

dataset_end_date = (
    orders["order_purchase_timestamp"].max()
)

print(
    "Dataset end date:",
    dataset_end_date
)


# ------------------------------------------------------------
# 6. SET PREDICTION CUTOFF
# ------------------------------------------------------------

# We need 90 days of future data to determine
# whether a customer became inactive.

prediction_cutoff = (
    dataset_end_date
    - pd.Timedelta(days=90)
)

print(
    "Prediction cutoff:",
    prediction_cutoff
)


# ------------------------------------------------------------
# 7. CREATE CUSTOMER FEATURES USING ONLY PAST DATA
# ------------------------------------------------------------

historical_orders = orders[
    (
        orders["order_purchase_timestamp"]
        <= prediction_cutoff
    )
    &
    (
        orders["order_status"]
        == "delivered"
    )
].copy()


print(
    "\nHistorical orders:",
    len(historical_orders)
)


# ------------------------------------------------------------
# 8. LAST PURCHASE BEFORE CUTOFF
# ------------------------------------------------------------

historical_last_purchase = (
    historical_orders
    .groupby("customer_unique_id")
    ["order_purchase_timestamp"]
    .max()
    .reset_index()
)


historical_last_purchase = (
    historical_last_purchase.rename(
        columns={
            "order_purchase_timestamp":
            "historical_last_purchase"
        }
    )
)


# ------------------------------------------------------------
# 9. HISTORICAL CUSTOMER FEATURES
# ------------------------------------------------------------

historical_features = (
    historical_orders
    .groupby("customer_unique_id")
    .agg(
        historical_orders=(
            "order_id",
            "nunique"
        ),

        historical_spend=(
            "order_value",
            "sum"
        ),

        historical_average_order_value=(
            "order_value",
            "mean"
        ),

        historical_items=(
            "total_items",
            "sum"
        ),

        historical_average_review=(
            "average_review_score",
            "mean"
        ),

        historical_average_delivery_delay=(
            "delivery_delay_days",
            "mean"
        ),

        historical_late_orders=(
            "is_late",
            "sum"
        )
    )
    .reset_index()
)


# ------------------------------------------------------------
# 10. MERGE LAST PURCHASE
# ------------------------------------------------------------

historical_features = historical_features.merge(
    historical_last_purchase,
    on="customer_unique_id",
    how="left"
)


# ------------------------------------------------------------
# 11. HISTORICAL RECENCY
# ------------------------------------------------------------

historical_features["historical_recency_days"] = (
    prediction_cutoff
    - historical_features["historical_last_purchase"]
).dt.days


# ------------------------------------------------------------
# 12. HISTORICAL LATE RATE
# ------------------------------------------------------------

historical_features["historical_late_rate"] = np.where(
    historical_features["historical_orders"] > 0,

    historical_features["historical_late_orders"]
    / historical_features["historical_orders"],

    0
)


# ============================================================
# PART B
# CREATE FUTURE TARGET
# ============================================================

print("\nCreating future behavior target...")


# ------------------------------------------------------------
# 13. FUTURE 90-DAY ORDERS
# ------------------------------------------------------------

future_orders = orders[
    (
        orders["order_purchase_timestamp"]
        > prediction_cutoff
    )
    &
    (
        orders["order_purchase_timestamp"]
        <= dataset_end_date
    )
    &
    (
        orders["order_status"]
        == "delivered"
    )
].copy()


print(
    "Future orders:",
    len(future_orders)
)


# ------------------------------------------------------------
# 14. COUNT FUTURE ORDERS
# ------------------------------------------------------------

future_activity = (
    future_orders
    .groupby("customer_unique_id")
    .agg(
        future_orders=(
            "order_id",
            "nunique"
        ),

        future_spend=(
            "order_value",
            "sum"
        )
    )
    .reset_index()
)


# ------------------------------------------------------------
# 15. MERGE FUTURE ACTIVITY
# ------------------------------------------------------------

historical_features = historical_features.merge(
    future_activity,
    on="customer_unique_id",
    how="left"
)


# Customers with no future order get zero.

historical_features["future_orders"] = (
    historical_features["future_orders"]
    .fillna(0)
)

historical_features["future_spend"] = (
    historical_features["future_spend"]
    .fillna(0)
)


# ------------------------------------------------------------
# 16. CREATE BINARY TARGET
# ------------------------------------------------------------

historical_features["inactive_90d"] = (
    historical_features["future_orders"] == 0
).astype(int)


# ------------------------------------------------------------
# 17. TARGET LABEL
# ------------------------------------------------------------

historical_features["activity_label"] = (
    historical_features["inactive_90d"]
    .map({
        0: "Active",
        1: "Inactive"
    })
)


# ============================================================
# PART C
# MERGE SAFE CUSTOMER INFORMATION
# ============================================================

# ------------------------------------------------------------
# 18. CUSTOMER LOCATION
# ------------------------------------------------------------

location_columns = [
    "customer_unique_id",
    "customer_city",
    "customer_state"
]

location_data = (
    customers[location_columns]
    .drop_duplicates(
        subset="customer_unique_id"
    )
)


historical_features = historical_features.merge(
    location_data,
    on="customer_unique_id",
    how="left"
)


# ============================================================
# PART D
# VALIDATION
# ============================================================

print("\n========== TARGET VALIDATION ==========")


print(
    "Total customers:",
    len(historical_features)
)


print(
    "Active customers:",
    (
        historical_features["inactive_90d"] == 0
    ).sum()
)


print(
    "Inactive customers:",
    (
        historical_features["inactive_90d"] == 1
    ).sum()
)


print(
    "\nTarget distribution:"
)

print(
    historical_features["inactive_90d"]
    .value_counts()
)


print(
    "\nTarget percentage:"
)

print(
    historical_features["inactive_90d"]
    .value_counts(
        normalize=True
    )
    * 100
)


# ------------------------------------------------------------
# 19. CHECK DUPLICATES
# ------------------------------------------------------------

duplicate_customers = (
    historical_features[
        "customer_unique_id"
    ]
    .duplicated()
    .sum()
)


print(
    "\nDuplicate customers:",
    duplicate_customers
)


# ------------------------------------------------------------
# 20. CHECK TARGET LEAKAGE COLUMNS
# ------------------------------------------------------------

print(
    "\nFuture columns intentionally kept:"
)

print(
    "future_orders"
)

print(
    "future_spend"
)

print(
    "inactive_90d"
)


# ============================================================
# PART E
# SAVE DATASET
# ============================================================

output_file = (
    PROCESSED_DIR
    / "customer_ml_dataset.csv"
)


historical_features.to_csv(
    output_file,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n========== SUCCESS ==========")

print(
    "Saved:"
)

print(
    output_file
)

print(
    "\nSTEP 5 COMPLETED SUCCESSFULLY!"
)