from pathlib import Path
import pandas as pd
import numpy as np

# Project directory
BASE_DIR = Path(__file__).resolve().parent

# Data directories
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

print("Raw data folder:")
print(RAW_DIR)

print("\nProcessed data folder:")
print(PROCESSED_DIR)

print("\nFiles inside raw folder:\n")

for file in RAW_DIR.iterdir():
    print(file.name)


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

print("\n========== DATASET SIZES ==========")

print("Customers:", customers.shape)
print("Orders:", orders.shape)
print("Order items:", order_items.shape)
print("Products:", products.shape)
print("Payments:", payments.shape)
print("Reviews:", reviews.shape)
print("Sellers:", sellers.shape)
print("Category translation:", category_translation.shape)

print("\n========== CUSTOMERS ==========")
print(customers.head())

print("\n========== ORDERS ==========")
print(orders.head())

print("\n========== ORDER ITEMS ==========")
print(order_items.head())

print("\n========== PRODUCTS ==========")
print(products.head())

print("\n========== PAYMENTS ==========")
print(payments.head())

print("\n========== REVIEWS ==========")
print(reviews.head())

print("\n========== COLUMNS ==========")

print("\nCUSTOMERS:")
print(customers.columns.tolist())

print("\nORDERS:")
print(orders.columns.tolist())

print("\nORDER ITEMS:")
print(order_items.columns.tolist())

print("\nPRODUCTS:")
print(products.columns.tolist())

print("\nPAYMENTS:")
print(payments.columns.tolist())

print("\nREVIEWS:")
print(reviews.columns.tolist())

print("\nSELLERS:")
print(sellers.columns.tolist())

def missing_summary(df):
    result = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percentage": (
            df.isna().mean() * 100
        ).round(2)
    })

    return result.sort_values(
        "missing_percentage",
        ascending=False
    )

print("\n========== CUSTOMER MISSING VALUES ==========")
print(missing_summary(customers))

print("\n========== ORDER MISSING VALUES ==========")
print(missing_summary(orders))

print("\n========== PRODUCT MISSING VALUES ==========")
print(missing_summary(products))

print("\n========== REVIEW MISSING VALUES ==========")
print(missing_summary(reviews))

print("\n========== DUPLICATES ==========")

print("Customers:", customers.duplicated().sum())
print("Orders:", orders.duplicated().sum())
print("Order items:", order_items.duplicated().sum())
print("Products:", products.duplicated().sum())
print("Payments:", payments.duplicated().sum())
print("Reviews:", reviews.duplicated().sum())
print("Sellers:", sellers.duplicated().sum())

print("\n========== UNIQUE IDS ==========")

print(
    "Unique customer_id:",
    customers["customer_id"].nunique()
)

print(
    "Unique customer_unique_id:",
    customers["customer_unique_id"].nunique()
)

print(
    "Unique order_id:",
    orders["order_id"].nunique()
)

print(
    "Unique product_id:",
    products["product_id"].nunique()
)

print(
    "Unique seller_id:",
    sellers["seller_id"].nunique()
)
