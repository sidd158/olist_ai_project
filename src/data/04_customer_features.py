from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# STEP 4 - CUSTOMER FEATURE ENGINEERING
# ============================================================

print("STEP 4 STARTED")


# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DIR = BASE_DIR / "data" / "processed"

print("\nProcessed data folder:")
print(PROCESSED_DIR)


# ------------------------------------------------------------
# 2. CHECK INPUT FILE
# ------------------------------------------------------------

input_file = PROCESSED_DIR / "order_level.csv"

if not input_file.exists():

    raise FileNotFoundError(
        f"Missing file: {input_file}\n"
        "Run Step 3 first."
    )

print("\nFound:")
print(input_file)


# ------------------------------------------------------------
# 3. LOAD ORDER LEVEL DATA
# ------------------------------------------------------------

print("\nLoading order-level data...")

orders = pd.read_csv(
    input_file
)

print("Order-level data loaded.")

print(
    "Rows:",
    len(orders)
)

print(
    "Columns:",
    len(orders.columns)
)


# ------------------------------------------------------------
# 4. CONVERT DATE
# ------------------------------------------------------------

orders["order_purchase_timestamp"] = pd.to_datetime(
    orders["order_purchase_timestamp"],
    errors="coerce"
)


# ------------------------------------------------------------
# 5. CREATE ANALYSIS DATE
# ------------------------------------------------------------

analysis_date = (
    orders["order_purchase_timestamp"].max()
    + pd.Timedelta(days=1)
)

print("\nAnalysis date:")
print(analysis_date)


# ------------------------------------------------------------
# 6. USE DELIVERED/VALID ORDERS
# ------------------------------------------------------------

# For customer behavior we focus on completed orders.
# This prevents cancelled/unavailable orders from
# dominating customer purchase behavior.

customer_orders = orders[
    orders["order_status"] == "delivered"
].copy()


print("\nDelivered orders:")
print(len(customer_orders))


# ------------------------------------------------------------
# 7. BASIC CUSTOMER FEATURES
# ------------------------------------------------------------

customer_features = (
    customer_orders
    .groupby("customer_unique_id")
    .agg(
        total_orders=(
            "order_id",
            "nunique"
        ),

        total_spend=(
            "order_value",
            "sum"
        ),

        average_order_value=(
            "order_value",
            "mean"
        ),

        total_items_purchased=(
            "total_items",
            "sum"
        ),

        average_items_per_order=(
            "total_items",
            "mean"
        )
    )
    .reset_index()
)


# ------------------------------------------------------------
# 8. RECENCY
# ------------------------------------------------------------

last_purchase = (
    customer_orders
    .groupby("customer_unique_id")[
        "order_purchase_timestamp"
    ]
    .max()
    .reset_index()
)

last_purchase = last_purchase.rename(
    columns={
        "order_purchase_timestamp":
        "last_purchase_date"
    }
)


customer_features = customer_features.merge(
    last_purchase,
    on="customer_unique_id",
    how="left"
)


customer_features["recency_days"] = (
    analysis_date
    - customer_features["last_purchase_date"]
).dt.days


# ------------------------------------------------------------
# 9. FIRST PURCHASE
# ------------------------------------------------------------

first_purchase = (
    customer_orders
    .groupby("customer_unique_id")[
        "order_purchase_timestamp"
    ]
    .min()
    .reset_index()
)

first_purchase = first_purchase.rename(
    columns={
        "order_purchase_timestamp":
        "first_purchase_date"
    }
)


customer_features = customer_features.merge(
    first_purchase,
    on="customer_unique_id",
    how="left"
)


# ------------------------------------------------------------
# 10. CUSTOMER LIFETIME
# ------------------------------------------------------------

customer_features["customer_lifetime_days"] = (
    customer_features["last_purchase_date"]
    - customer_features["first_purchase_date"]
).dt.days


# ------------------------------------------------------------
# 11. REVIEW FEATURES
# ------------------------------------------------------------

review_features = (
    customer_orders
    .groupby("customer_unique_id")
    .agg(
        average_review_score=(
            "average_review_score",
            "mean"
        ),

        review_count=(
            "average_review_score",
            "count"
        )
    )
    .reset_index()
)


customer_features = customer_features.merge(
    review_features,
    on="customer_unique_id",
    how="left"
)


# ------------------------------------------------------------
# 12. DELIVERY FEATURES
# ------------------------------------------------------------

delivery_features = (
    customer_orders
    .groupby("customer_unique_id")
    .agg(
        average_delivery_days=(
            "delivery_days",
            "mean"
        ),

        average_delivery_delay=(
            "delivery_delay_days",
            "mean"
        ),

        late_order_count=(
            "is_late",
            "sum"
        ),

        delivered_order_count=(
            "order_id",
            "nunique"
        )
    )
    .reset_index()
)


customer_features = customer_features.merge(
    delivery_features,
    on="customer_unique_id",
    how="left"
)


# ------------------------------------------------------------
# 13. LATE ORDER RATE
# ------------------------------------------------------------

customer_features["late_order_rate"] = np.where(
    customer_features["delivered_order_count"] > 0,

    customer_features["late_order_count"]
    / customer_features["delivered_order_count"],

    0
)


# ------------------------------------------------------------
# 14. PAYMENT FEATURES
# ------------------------------------------------------------

payment_features = (
    customer_orders
    .groupby("customer_unique_id")
    .agg(
        average_payment_installments=(
            "payment_installments",
            "mean"
        ),

        total_payment_value=(
            "payment_value",
            "sum"
        )
    )
    .reset_index()
)


customer_features = customer_features.merge(
    payment_features,
    on="customer_unique_id",
    how="left"
)


# ------------------------------------------------------------
# 15. CUSTOMER LOCATION
# ------------------------------------------------------------

location_features = (
    customer_orders
    .sort_values("order_purchase_timestamp")
    .groupby("customer_unique_id")
    .agg(
        customer_city=(
            "customer_city",
            "last"
        ),

        customer_state=(
            "customer_state",
            "last"
        )
    )
    .reset_index()
)


customer_features = customer_features.merge(
    location_features,
    on="customer_unique_id",
    how="left"
)


# ------------------------------------------------------------
# 16. PURCHASE FREQUENCY
# ------------------------------------------------------------

customer_features["orders_per_lifetime_day"] = np.where(
    customer_features["customer_lifetime_days"] > 0,

    customer_features["total_orders"]
    / customer_features["customer_lifetime_days"],

    0
)


# ------------------------------------------------------------
# 17. SPEND PER ITEM
# ------------------------------------------------------------

customer_features["spend_per_item"] = np.where(
    customer_features["total_items_purchased"] > 0,

    customer_features["total_spend"]
    / customer_features["total_items_purchased"],

    0
)


# ------------------------------------------------------------
# 18. REVIEW TEXT INDICATOR
# ------------------------------------------------------------

review_text_features = (
    customer_orders
    .groupby("customer_unique_id")
    .agg(
        orders_with_review_text=(
            "has_review_text",
            "sum"
        )
    )
    .reset_index()
)


customer_features = customer_features.merge(
    review_text_features,
    on="customer_unique_id",
    how="left"
)


# ------------------------------------------------------------
# 19. REVIEW TEXT RATE
# ------------------------------------------------------------

customer_features["review_text_rate"] = np.where(
    customer_features["total_orders"] > 0,

    customer_features["orders_with_review_text"]
    / customer_features["total_orders"],

    0
)


# ------------------------------------------------------------
# 20. REMOVE IMPOSSIBLE VALUES
# ------------------------------------------------------------

customer_features["customer_lifetime_days"] = (
    customer_features["customer_lifetime_days"]
    .clip(lower=0)
)

customer_features["average_delivery_days"] = (
    customer_features["average_delivery_days"]
    .clip(lower=0)
)

customer_features["average_delivery_delay"] = (
    customer_features["average_delivery_delay"]
    .fillna(0)
)


# ------------------------------------------------------------
# 21. CHECK MISSING VALUES
# ------------------------------------------------------------

print("\n========== MISSING VALUES ==========")

missing_values = (
    customer_features
    .isna()
    .sum()
    .sort_values(ascending=False)
)

print(missing_values)


# ------------------------------------------------------------
# 22. FINAL DATASET INFO
# ------------------------------------------------------------

print("\n========== CUSTOMER DATASET ==========")

print(
    "Customers:",
    len(customer_features)
)

print(
    "Features:",
    len(customer_features.columns)
)

print("\nColumns:")

for column in customer_features.columns:

    print("-", column)


# ------------------------------------------------------------
# 23. CHECK DUPLICATES
# ------------------------------------------------------------

duplicates = (
    customer_features["customer_unique_id"]
    .duplicated()
    .sum()
)

print(
    "\nDuplicate customers:",
    duplicates
)


# ------------------------------------------------------------
# 24. SAVE CUSTOMER FEATURES
# ------------------------------------------------------------

output_file = (
    PROCESSED_DIR
    / "customer_features.csv"
)


customer_features.to_csv(
    output_file,
    index=False
)


# ------------------------------------------------------------
# 25. FINAL OUTPUT
# ------------------------------------------------------------

print("\n========== SUCCESS ==========")

print("Saved:")
print(output_file)

print("\nSTEP 4 COMPLETED SUCCESSFULLY!")