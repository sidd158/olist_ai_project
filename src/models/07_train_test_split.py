import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# STEP 7: TRAIN / TEST SPLIT
# ============================================================

print("========== STEP 7: TRAIN / TEST SPLIT ==========")


# ============================================================
# 1. LOAD ML DATA
# ============================================================

X = pd.read_csv(
    r"D:\olist_ai_project\data\processed\X_ml.csv"
)

y = pd.read_csv(
    r"D:\olist_ai_project\data\processed\y_ml.csv"
).squeeze()


print("\n========== DATA LOADED ==========")

print("X shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# 2. CHECK TARGET DISTRIBUTION
# ============================================================

print("\n========== TARGET DISTRIBUTION ==========")

print(y.value_counts())


print("\n========== TARGET PERCENTAGE ==========")

print(
    y.value_counts(normalize=True) * 100
)


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 4. DISPLAY DATASET SIZES
# ============================================================

print("\n========== TRAIN / TEST SPLIT ==========")

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# ============================================================
# 5. CHECK TRAIN TARGET
# ============================================================

print("\n========== TRAIN TARGET DISTRIBUTION ==========")

print(y_train.value_counts())


print("\n========== TRAIN TARGET PERCENTAGE ==========")

print(
    y_train.value_counts(normalize=True) * 100
)


# ============================================================
# 6. CHECK TEST TARGET
# ============================================================

print("\n========== TEST TARGET DISTRIBUTION ==========")

print(y_test.value_counts())


print("\n========== TEST TARGET PERCENTAGE ==========")

print(
    y_test.value_counts(normalize=True) * 100
)


# ============================================================
# 7. SAVE SPLIT DATA
# ============================================================

output_path = r"D:\olist_ai_project\data\processed"

X_train.to_csv(
    output_path + r"\X_train.csv",
    index=False
)

X_test.to_csv(
    output_path + r"\X_test.csv",
    index=False
)

y_train.to_csv(
    output_path + r"\y_train.csv",
    index=False
)

y_test.to_csv(
    output_path + r"\y_test.csv",
    index=False
)


# ============================================================
# SUCCESS
# ============================================================

print("\n========== SUCCESS ==========")

print("Saved:")
print(output_path + r"\X_train.csv")
print(output_path + r"\X_test.csv")
print(output_path + r"\y_train.csv")
print(output_path + r"\y_test.csv")

print("\nSTEP 7 COMPLETED SUCCESSFULLY!")