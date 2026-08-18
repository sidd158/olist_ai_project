from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# STEP 2 - CLEAN OLIST DATA
# ============================================================

print("STEP 2 STARTED")


# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("\nRaw data folder:")
print(RAW_DIR)

print("\nProcessed data folder:")
print(PROCESSED_DIR)


# ------------------------------------------------------------
# 2. LOAD RAW DATA
# ------------------------------------------------------------

print("\nLoading raw datasets...")


customers = pd.read_csv(
    RAW_DIR / "olist_customers_dataset.csv"
)

orders = pd.read_csv(
    RAW_DIR / "olist_orders_dataset.csv"
)

order_items = pd.read_csv(
    RAW_DIR / "olist_order_items_dataset.csv"
)

products = pd.read_csv(
    RAW_DIR / "olist_products_dataset.csv"
)

payments = pd.read_csv(
    RAW_DIR / "olist_order_payments_dataset.csv"
)

reviews = pd.read_csv(
    RAW_DIR / "olist_order_reviews_dataset.csv"
)

sellers = pd.read_csv(
    RAW_DIR / "olist_sellers_dataset.csv"
)

category_translation = pd.read_csv(
    RAW_DIR / "product_category_name_translation.csv"
)


print("Raw datasets loaded successfully.")


# ------------------------------------------------------------
# 3. CLEAN COLUMN NAMES
# ------------------------------------------------------------

def clean_column_names(df):

    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    return df


customers = clean_column_names(customers)
orders = clean_column_names(orders)
order_items = clean_column_names(order_items)
products = clean_column_names(products)
payments = clean_column_names(payments)
reviews = clean_column_names(reviews)
sellers = clean_column_names(sellers)
category_translation = clean_column_names(category_translation)


# ------------------------------------------------------------
# 4. FIX PRODUCT COLUMN NAMES
# ------------------------------------------------------------

products = products.rename(columns={
    "product_name_lenght": "product_name_length",
    "product_description_lenght": "product_description_length"
})


# ------------------------------------------------------------
# 5. CONVERT ORDER DATES
# ------------------------------------------------------------

order_date_columns = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]

for column in order_date_columns:

    orders[column] = pd.to_datetime(
        orders[column],
        errors="coerce"
    )


# ------------------------------------------------------------
# 6. CONVERT ORDER ITEM DATE
# ------------------------------------------------------------

order_items["shipping_limit_date"] = pd.to_datetime(
    order_items["shipping_limit_date"],
    errors="coerce"
)


# ------------------------------------------------------------
# 7. CONVERT REVIEW DATES
# ------------------------------------------------------------

reviews["review_creation_date"] = pd.to_datetime(
    reviews["review_creation_date"],
    errors="coerce"
)

reviews["review_answer_timestamp"] = pd.to_datetime(
    reviews["review_answer_timestamp"],
    errors="coerce"
)


# ------------------------------------------------------------
# 8. CLEAN TEXT COLUMNS
# ------------------------------------------------------------

customers["customer_city"] = (
    customers["customer_city"]
    .str.strip()
)

customers["customer_state"] = (
    customers["customer_state"]
    .str.strip()
)

sellers["seller_city"] = (
    sellers["seller_city"]
    .str.strip()
)

sellers["seller_state"] = (
    sellers["seller_state"]
    .str.strip()
)

products["product_category_name"] = (
    products["product_category_name"]
    .str.strip()
)


# ------------------------------------------------------------
# 9. TRANSLATE PRODUCT CATEGORIES
# ------------------------------------------------------------

products = products.merge(
    category_translation,
    on="product_category_name",
    how="left"
)

products["category"] = (
    products["product_category_name_english"]
    .fillna(products["product_category_name"])
)


# ------------------------------------------------------------
# 10. PRODUCT FEATURES
# ------------------------------------------------------------

products["product_volume_cm3"] = (
    products["product_length_cm"]
    * products["product_height_cm"]
    * products["product_width_cm"]
)


products["has_photos"] = (
    products["product_photos_qty"]
    .fillna(0)
    > 0
).astype(int)


products["has_description"] = (
    products["product_description_length"]
    .fillna(0)
    > 0
).astype(int)


# ------------------------------------------------------------
# 11. ORDER DELIVERY FEATURES
# ------------------------------------------------------------

orders["delivery_days"] = (
    orders["order_delivered_customer_date"]
    - orders["order_purchase_timestamp"]
).dt.total_seconds() / (24 * 60 * 60)


orders["estimated_delivery_days"] = (
    orders["order_estimated_delivery_date"]
    - orders["order_purchase_timestamp"]
).dt.total_seconds() / (24 * 60 * 60)


orders["delivery_delay_days"] = (
    orders["order_delivered_customer_date"]
    - orders["order_estimated_delivery_date"]
).dt.total_seconds() / (24 * 60 * 60)


# ------------------------------------------------------------
# 12. ORDER CALENDAR FEATURES
# ------------------------------------------------------------

orders["purchase_year"] = (
    orders["order_purchase_timestamp"].dt.year
)

orders["purchase_month"] = (
    orders["order_purchase_timestamp"].dt.month
)

orders["purchase_day"] = (
    orders["order_purchase_timestamp"].dt.day
)

orders["purchase_day_of_week"] = (
    orders["order_purchase_timestamp"].dt.dayofweek
)

orders["purchase_day_name"] = (
    orders["order_purchase_timestamp"].dt.day_name()
)


# ------------------------------------------------------------
# 13. REVIEW FEATURES
# ------------------------------------------------------------

reviews["has_review_text"] = (
    reviews["review_comment_message"]
    .fillna("")
    .str.strip()
    .ne("")
    .astype(int)
)


reviews["review_text_length"] = (
    reviews["review_comment_message"]
    .fillna("")
    .str.len()
)


# ------------------------------------------------------------
# 14. PRINT DATASET SIZES
# ------------------------------------------------------------

print("\n========== CLEANED DATA ==========")

print("Customers:", customers.shape)
print("Orders:", orders.shape)
print("Order items:", order_items.shape)
print("Products:", products.shape)
print("Payments:", payments.shape)
print("Reviews:", reviews.shape)
print("Sellers:", sellers.shape)


# ------------------------------------------------------------
# 15. SAVE CLEAN DATA
# ------------------------------------------------------------

print("\nSaving cleaned datasets...")


customers.to_csv(
    PROCESSED_DIR / "clean_customers.csv",
    index=False
)

orders.to_csv(
    PROCESSED_DIR / "clean_orders.csv",
    index=False
)

order_items.to_csv(
    PROCESSED_DIR / "clean_order_items.csv",
    index=False
)

products.to_csv(
    PROCESSED_DIR / "clean_products.csv",
    index=False
)

payments.to_csv(
    PROCESSED_DIR / "clean_payments.csv",
    index=False
)

reviews.to_csv(
    PROCESSED_DIR / "clean_reviews.csv",
    index=False
)

sellers.to_csv(
    PROCESSED_DIR / "clean_sellers.csv",
    index=False
)


# ------------------------------------------------------------
# 16. FINAL CHECK
# ------------------------------------------------------------

print("\n========== SAVED FILES ==========")

for file in PROCESSED_DIR.iterdir():

    print(file.name)


print("\nSTEP 2 COMPLETED SUCCESSFULLY!")