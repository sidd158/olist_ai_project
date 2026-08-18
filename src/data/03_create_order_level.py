from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# STEP 3 - CREATE ORDER-LEVEL ANALYTICAL DATASET
# ============================================================

print("STEP 3 STARTED")


# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DIR = BASE_DIR / "data" / "processed"

print("\nProcessed data folder:")
print(PROCESSED_DIR)


# ------------------------------------------------------------
# 2. CHECK REQUIRED FILES
# ------------------------------------------------------------

required_files = [
    "clean_customers.csv",
    "clean_orders.csv",
    "clean_order_items.csv",
    "clean_products.csv",
    "clean_payments.csv",
    "clean_reviews.csv",
    "clean_sellers.csv"
]

print("\nChecking required files...")

for file_name in required_files:

    file_path = PROCESSED_DIR / file_name

    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing file: {file_path}\n"
            "Run Step 2 first."
        )

    print("Found:", file_name)


# ------------------------------------------------------------
# 3. LOAD CLEAN DATA
# ------------------------------------------------------------

print("\nLoading cleaned datasets...")


customers = pd.read_csv(
    PROCESSED_DIR / "clean_customers.csv"
)

orders = pd.read_csv(
    PROCESSED_DIR / "clean_orders.csv"
)

order_items = pd.read_csv(
    PROCESSED_DIR / "clean_order_items.csv"
)

products = pd.read_csv(
    PROCESSED_DIR / "clean_products.csv"
)

payments = pd.read_csv(
    PROCESSED_DIR / "clean_payments.csv"
)

reviews = pd.read_csv(
    PROCESSED_DIR / "clean_reviews.csv"
)

sellers = pd.read_csv(
    PROCESSED_DIR / "clean_sellers.csv"
)

print("All datasets loaded successfully.")


# ------------------------------------------------------------
# 4. CONVERT ORDER DATES
# ------------------------------------------------------------

orders["order_purchase_timestamp"] = pd.to_datetime(
    orders["order_purchase_timestamp"],
    errors="coerce"
)

orders["order_approved_at"] = pd.to_datetime(
    orders["order_approved_at"],
    errors="coerce"
)

orders["order_delivered_customer_date"] = pd.to_datetime(
    orders["order_delivered_customer_date"],
    errors="coerce"
)

orders["order_estimated_delivery_date"] = pd.to_datetime(
    orders["order_estimated_delivery_date"],
    errors="coerce"
)


# ------------------------------------------------------------
# 5. AGGREGATE ORDER ITEMS
# ------------------------------------------------------------

print("\nCreating item summary...")

item_summary = (
    order_items
    .groupby("order_id")
    .agg(
        total_items=("order_item_id", "count"),
        total_product_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        unique_products=("product_id", "nunique"),
        unique_sellers=("seller_id", "nunique")
    )
    .reset_index()
)


# ------------------------------------------------------------
# 6. CALCULATE ORDER VALUE
# ------------------------------------------------------------

item_summary["order_value"] = (
    item_summary["total_product_price"]
    + item_summary["total_freight"]
)


# ------------------------------------------------------------
# 7. AGGREGATE PAYMENTS
# ------------------------------------------------------------

print("Creating payment summary...")

payment_summary = (
    payments
    .groupby("order_id")
    .agg(
        payment_value=("payment_value", "sum"),
        payment_installments=("payment_installments", "max")
    )
    .reset_index()
)


# ------------------------------------------------------------
# 8. PRIMARY PAYMENT TYPE
# ------------------------------------------------------------

payment_type_summary = (
    payments
    .groupby("order_id")["payment_type"]
    .agg(
        lambda x: (
            x.mode().iloc[0]
            if not x.mode().empty
            else "unknown"
        )
    )
    .reset_index()
)

payment_type_summary = payment_type_summary.rename(
    columns={
        "payment_type": "primary_payment_type"
    }
)


payment_summary = payment_summary.merge(
    payment_type_summary,
    on="order_id",
    how="left"
)


# ------------------------------------------------------------
# 9. AGGREGATE REVIEWS
# ------------------------------------------------------------

print("Creating review summary...")

review_summary = (
    reviews
    .groupby("order_id")
    .agg(
        average_review_score=("review_score", "mean"),
        has_review_text=("has_review_text", "max"),
        review_text_length=("review_text_length", "max")
    )
    .reset_index()
)


# ------------------------------------------------------------
# 10. CUSTOMER INFORMATION
# ------------------------------------------------------------

customer_data = customers[
    [
        "customer_id",
        "customer_unique_id",
        "customer_zip_code_prefix",
        "customer_city",
        "customer_state"
    ]
].copy()


# ------------------------------------------------------------
# 11. START ORDER-LEVEL DATASET
# ------------------------------------------------------------

print("\nBuilding order-level dataset...")


order_level = orders.merge(
    customer_data,
    on="customer_id",
    how="left"
)


# ------------------------------------------------------------
# 12. MERGE ITEM SUMMARY
# ------------------------------------------------------------

order_level = order_level.merge(
    item_summary,
    on="order_id",
    how="left"
)


# ------------------------------------------------------------
# 13. MERGE PAYMENT SUMMARY
# ------------------------------------------------------------

order_level = order_level.merge(
    payment_summary,
    on="order_id",
    how="left"
)


# ------------------------------------------------------------
# 14. MERGE REVIEW SUMMARY
# ------------------------------------------------------------

order_level = order_level.merge(
    review_summary,
    on="order_id",
    how="left"
)


# ------------------------------------------------------------
# 15. CREATE DELIVERY FEATURES
# ------------------------------------------------------------

order_level["delivery_days"] = (
    order_level["order_delivered_customer_date"]
    - order_level["order_purchase_timestamp"]
).dt.total_seconds() / (24 * 60 * 60)


order_level["estimated_delivery_days"] = (
    order_level["order_estimated_delivery_date"]
    - order_level["order_purchase_timestamp"]
).dt.total_seconds() / (24 * 60 * 60)


order_level["delivery_delay_days"] = (
    order_level["order_delivered_customer_date"]
    - order_level["order_estimated_delivery_date"]
).dt.total_seconds() / (24 * 60 * 60)


# ------------------------------------------------------------
# 16. DELIVERY STATUS FEATURES
# ------------------------------------------------------------

order_level["is_delivered"] = (
    order_level["order_status"] == "delivered"
).astype(int)


order_level["is_late"] = np.where(
    order_level["delivery_delay_days"].notna(),
    (
        order_level["delivery_delay_days"] > 0
    ).astype(int),
    np.nan
)


# ------------------------------------------------------------
# 17. CALENDAR FEATURES
# ------------------------------------------------------------

order_level["purchase_year"] = (
    order_level["order_purchase_timestamp"].dt.year
)

order_level["purchase_month"] = (
    order_level["order_purchase_timestamp"].dt.month
)

order_level["purchase_day_of_week"] = (
    order_level["order_purchase_timestamp"].dt.dayofweek
)


# ------------------------------------------------------------
# 18. FREIGHT RATIO
# ------------------------------------------------------------

order_level["freight_ratio"] = np.where(
    order_level["order_value"] > 0,
    order_level["total_freight"]
    / order_level["order_value"],
    0
)


# ------------------------------------------------------------
# 19. SELECT FINAL COLUMNS
# ------------------------------------------------------------

final_columns = [
    "order_id",
    "customer_id",
    "customer_unique_id",

    "customer_city",
    "customer_state",

    "order_status",

    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",

    "purchase_year",
    "purchase_month",
    "purchase_day_of_week",

    "delivery_days",
    "estimated_delivery_days",
    "delivery_delay_days",

    "is_delivered",
    "is_late",

    "total_items",
    "total_product_price",
    "total_freight",
    "order_value",

    "unique_products",
    "unique_sellers",

    "payment_value",
    "payment_installments",
    "primary_payment_type",

    "average_review_score",
    "has_review_text",
    "review_text_length",

    "freight_ratio"
]


order_level = order_level[final_columns]


# ------------------------------------------------------------
# 20. VALIDATION
# ------------------------------------------------------------

print("\n========== VALIDATION ==========")

print(
    "Total rows:",
    len(order_level)
)

print(
    "Unique orders:",
    order_level["order_id"].nunique()
)

print(
    "Total columns:",
    len(order_level.columns)
)

duplicate_orders = (
    order_level["order_id"].duplicated().sum()
)

print(
    "Duplicate order rows:",
    duplicate_orders
)


# ------------------------------------------------------------
# 21. CHECK REVENUE
# ------------------------------------------------------------

print("\n========== REVENUE CHECK ==========")

print(
    "Product revenue:",
    order_level["total_product_price"].sum()
)

print(
    "Freight revenue:",
    order_level["total_freight"].sum()
)

print(
    "Total order value:",
    order_level["order_value"].sum()
)

print(
    "Total payment value:",
    order_level["payment_value"].sum()
)


# ------------------------------------------------------------
# 22. SAVE FINAL DATASET
# ------------------------------------------------------------

output_file = PROCESSED_DIR / "order_level.csv"

order_level.to_csv(
    output_file,
    index=False
)


# ------------------------------------------------------------
# 23. FINAL MESSAGE
# ------------------------------------------------------------

print("\n========== SUCCESS ==========")

print("Saved file:")
print(output_file)

print("\nSTEP 3 COMPLETED SUCCESSFULLY!")