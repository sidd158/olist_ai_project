import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    average_precision_score
)


# ============================================================
# STEP 9: MODEL EVALUATION
# ============================================================

print("========== STEP 9: MODEL EVALUATION ==========")


# ============================================================
# 1. LOAD DATA
# ============================================================

data_path = r"D:\olist_ai_project\data\processed"

X_train = pd.read_csv(
    data_path + r"\X_train.csv"
)

X_test = pd.read_csv(
    data_path + r"\X_test.csv"
)

y_train = pd.read_csv(
    data_path + r"\y_train.csv"
).squeeze()

y_test = pd.read_csv(
    data_path + r"\y_test.csv"
).squeeze()


print("\nData loaded successfully!")


# ============================================================
# 2. TRAIN BASELINE MODEL
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
# 3. PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]


# ============================================================
# 4. CLASSIFICATION REPORT
# ============================================================

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# 5. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n========== CONFUSION MATRIX ==========")

print(cm)


# ============================================================
# 6. ROC-AUC
# ============================================================

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

print("\n========== ROC-AUC ==========")

print("ROC-AUC:", roc_auc)


# ============================================================
# 7. PR-AUC
# ============================================================

pr_auc = average_precision_score(
    y_test,
    y_prob
)

print("\n========== PR-AUC ==========")

print("PR-AUC:", pr_auc)


# ============================================================
# 8. CONFUSION MATRIX VISUALIZATION
# ============================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Active", "Inactive"]
)

disp.plot()

plt.title("Logistic Regression - Confusion Matrix")

plt.tight_layout()

plt.show()


# ============================================================
# SUCCESS
# ============================================================

print("\n========== STEP 9 COMPLETED SUCCESSFULLY ==========")