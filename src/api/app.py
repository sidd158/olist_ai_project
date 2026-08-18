from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
import joblib
import shap


# ============================================================
# OLIST CUSTOMER INACTIVITY PREDICTION API
# ============================================================

app = FastAPI(
    title="Olist Customer Inactivity Prediction API",
    description="ML API for predicting customer inactivity",
    version="2.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = (
    r"D:\olist_ai_project\models"
    r"\final_model.pkl"
)

DATA_PATH = (
    r"D:\olist_ai_project\data"
    r"\processed"
)

X_TEST_PATH = DATA_PATH + r"\X_test.csv"


# ============================================================
# LOAD FINAL ML MODEL
# ============================================================

print("Loading final ML model...")

try:
    model = joblib.load(MODEL_PATH)

except Exception as error:
    print("ERROR: Could not load final model.")
    print(error)
    raise

print("Final ML model loaded successfully!")


# ============================================================
# LOAD CUSTOMER FEATURES
# ============================================================

print("Loading customer features...")

try:
    X_test = pd.read_csv(X_TEST_PATH)

except Exception as error:
    print("ERROR: Could not load X_test.csv.")
    print(error)
    raise

print(
    f"Customer data loaded: "
    f"{X_test.shape[0]} customers, "
    f"{X_test.shape[1]} features"
)


# ============================================================
# CREATE SHAP EXPLAINER
# ============================================================

print("Creating SHAP explainer...")

try:

    shap_background = X_test.sample(
        min(50, len(X_test)),
        random_state=42
    )

    shap_explainer = shap.Explainer(
        model,
        shap_background
    )

except Exception as error:

    print("WARNING: SHAP explainer could not be created.")
    print(error)

    shap_explainer = None

print("SHAP explainer ready!")


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Olist Customer Inactivity Prediction API",
        "status": "running",
        "model": type(model).__name__,
        "features": X_test.shape[1],
        "customers": len(X_test)
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "shap_loaded": shap_explainer is not None,
        "features": X_test.shape[1],
        "customers": len(X_test)
    }


# ============================================================
# CUSTOMER PREDICTION
# ============================================================

@app.get("/predict/{customer_index}")
def predict_customer(customer_index: int):

    # --------------------------------------------------------
    # CHECK CUSTOMER INDEX
    # --------------------------------------------------------

    if (
        customer_index < 0
        or customer_index >= len(X_test)
    ):

        raise HTTPException(
            status_code=404,
            detail="Customer index not found"
        )


    # --------------------------------------------------------
    # GET CUSTOMER FEATURES
    # --------------------------------------------------------

    customer = X_test.iloc[
        customer_index:customer_index + 1
    ]


    # --------------------------------------------------------
    # ML PREDICTION
    # --------------------------------------------------------

    try:

        prediction = model.predict(
            customer
        )[0]

        probability = model.predict_proba(
            customer
        )[0][1]

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {error}"
        )


    # --------------------------------------------------------
    # RISK LEVEL
    # --------------------------------------------------------

    if probability >= 0.80:

        risk = "HIGH"

        recommendation = (
            "Send a personalized retention offer "
            "or re-engagement campaign."
        )

    elif probability >= 0.50:

        risk = "MEDIUM"

        recommendation = (
            "Send a personalized reminder "
            "or product recommendation."
        )

    else:

        risk = "LOW"

        recommendation = (
            "Continue regular customer engagement."
        )


    # --------------------------------------------------------
    # RETURN PREDICTION
    # --------------------------------------------------------

    return {

        "customer_index":
            customer_index,

        "inactive_prediction":
            int(prediction),

        "inactive_probability":
            round(
                float(probability),
                4
            ),

        "inactive_probability_percent":
            round(
                float(probability * 100),
                2
            ),

        "risk_level":
            risk,

        "recommendation":
            recommendation
    }


# ============================================================
# RISK STATISTICS
# ============================================================

@app.get("/stats")
def get_statistics():

    prediction_file = (
        DATA_PATH +
        r"\customer_predictions.csv"
    )

    try:

        predictions = pd.read_csv(
            prediction_file
        )

    except FileNotFoundError:

        raise HTTPException(
            status_code=404,
            detail=(
                "customer_predictions.csv not found. "
                "Run Step 18 first."
            )
        )


    total = len(predictions)

    if total == 0:

        raise HTTPException(
            status_code=404,
            detail="Prediction data is empty."
        )


    # --------------------------------------------------------
    # RISK COUNTS
    # --------------------------------------------------------

    high = int(
        (
            predictions["risk_level"] == "HIGH"
        ).sum()
    )

    medium = int(
        (
            predictions["risk_level"] == "MEDIUM"
        ).sum()
    )

    low = int(
        (
            predictions["risk_level"] == "LOW"
        ).sum()
    )


    # --------------------------------------------------------
    # RETURN STATISTICS
    # --------------------------------------------------------

    return {

        "total_customers":
            total,

        "high_risk":
            high,

        "medium_risk":
            medium,

        "low_risk":
            low,

        "high_percentage":
            round(
                high / total * 100,
                2
            ),

        "medium_percentage":
            round(
                medium / total * 100,
                2
            ),

        "low_percentage":
            round(
                low / total * 100,
                2
            )
    }


# ============================================================
# HIGH-RISK CUSTOMERS
# ============================================================

@app.get("/high-risk-customers")
def get_high_risk_customers():

    prediction_file = (
        DATA_PATH +
        r"\customer_predictions.csv"
    )

    try:

        predictions = pd.read_csv(
            prediction_file
        )

    except FileNotFoundError:

        raise HTTPException(
            status_code=404,
            detail="customer_predictions.csv not found."
        )


    # --------------------------------------------------------
    # FILTER HIGH-RISK CUSTOMERS
    # --------------------------------------------------------

    high_risk = predictions[
        predictions["risk_level"] == "HIGH"
    ].copy()


    # --------------------------------------------------------
    # SORT BY PROBABILITY
    # --------------------------------------------------------

    high_risk = high_risk.sort_values(
        "inactive_probability",
        ascending=False
    )


    # --------------------------------------------------------
    # TOP 20
    # --------------------------------------------------------

    high_risk = high_risk.head(20)


    customers = []


    for _, row in high_risk.iterrows():

        customers.append({

            "customer_index":
                int(
                    row["customer_index"]
                ),

            "inactive_probability":
                round(
                    float(
                        row[
                            "inactive_probability"
                        ]
                    ),
                    4
                ),

            "inactive_probability_percent":
                round(
                    float(
                        row[
                            "inactive_probability_percent"
                        ]
                    ),
                    2
                ),

            "risk_level":
                str(
                    row["risk_level"]
                )
        })


    return {

        "count":
            len(customers),

        "customers":
            customers
    }


# ============================================================
# STEP 31
# DYNAMIC SHAP EXPLANATION
# ============================================================

@app.get("/shap-explanation/{customer_index}")
def get_shap_explanation(
    customer_index: int
):

    # --------------------------------------------------------
    # CHECK CUSTOMER INDEX
    # --------------------------------------------------------

    if (
        customer_index < 0
        or customer_index >= len(X_test)
    ):

        raise HTTPException(
            status_code=404,
            detail="Customer index not found"
        )


    # --------------------------------------------------------
    # CHECK SHAP EXPLAINER
    # --------------------------------------------------------

    if shap_explainer is None:

        raise HTTPException(
            status_code=500,
            detail="SHAP explainer is not available."
        )


    # --------------------------------------------------------
    # GET CUSTOMER
    # --------------------------------------------------------

    customer = X_test.iloc[
        customer_index:customer_index + 1
    ]


    # --------------------------------------------------------
    # CALCULATE SHAP
    # --------------------------------------------------------

    try:

        shap_result = shap_explainer(
            customer
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                f"SHAP calculation failed: {error}"
            )
        )


    # --------------------------------------------------------
    # GET SHAP VALUES
    # --------------------------------------------------------

    values = shap_result.values


    # --------------------------------------------------------
    # HANDLE CLASSIFICATION OUTPUT
    # --------------------------------------------------------

    if len(values.shape) == 3:

        values = values[:, :, 1]


    customer_shap = values[0]


    # --------------------------------------------------------
    # CREATE EXPLANATION DATAFRAME
    # --------------------------------------------------------

    explanation = pd.DataFrame({

        "feature":
            customer.columns,

        "customer_value":
            customer.iloc[0].values,

        "shap_value":
            customer_shap
    })


    # --------------------------------------------------------
    # CALCULATE IMPORTANCE
    # --------------------------------------------------------

    explanation["importance"] = (
        explanation["shap_value"]
        .abs()
    )


    # --------------------------------------------------------
    # TOP 10 FEATURES
    # --------------------------------------------------------

    explanation = explanation.sort_values(
        "importance",
        ascending=False
    ).head(10)


    # --------------------------------------------------------
    # CREATE FACTORS
    # --------------------------------------------------------

    factors = []


    for _, row in explanation.iterrows():

        factors.append({

            "feature":
                str(
                    row["feature"]
                ),

            "customer_value":
                float(
                    row["customer_value"]
                ),

            "shap_value":
                round(
                    float(
                        row["shap_value"]
                    ),
                    6
                )
        })


    # --------------------------------------------------------
    # RETURN SHAP RESULT
    # --------------------------------------------------------

    return {

        "customer_index":
            customer_index,

        "factors":
            factors
    }


# ============================================================
# STEP 34
# AI BUSINESS SUMMARY
# ============================================================

@app.get("/business-summary")
def get_business_summary():

    prediction_file = (
        DATA_PATH +
        r"\customer_predictions.csv"
    )

    # --------------------------------------------------------
    # LOAD PREDICTIONS
    # --------------------------------------------------------

    try:

        predictions = pd.read_csv(
            prediction_file
        )

    except FileNotFoundError:

        raise HTTPException(
            status_code=404,
            detail="customer_predictions.csv not found."
        )


    # --------------------------------------------------------
    # CHECK DATA
    # --------------------------------------------------------

    total = len(predictions)

    if total == 0:

        raise HTTPException(
            status_code=404,
            detail="Prediction data is empty."
        )


    # --------------------------------------------------------
    # RISK COUNTS
    # --------------------------------------------------------

    high = int(
        (
            predictions["risk_level"] == "HIGH"
        ).sum()
    )

    medium = int(
        (
            predictions["risk_level"] == "MEDIUM"
        ).sum()
    )

    low = int(
        (
            predictions["risk_level"] == "LOW"
        ).sum()
    )


    # --------------------------------------------------------
    # CUSTOMERS AT RISK
    # --------------------------------------------------------

    at_risk = high + medium

    at_risk_percentage = (
        at_risk / total * 100
    )


    # --------------------------------------------------------
    # RISK RATIOS
    # --------------------------------------------------------

    high_ratio = high / total

    at_risk_ratio = at_risk / total


    # ========================================================
    # RETENTION PRIORITY
    # ========================================================

    if (
        high_ratio >= 0.20
        or at_risk_ratio >= 0.70
    ):

        priority = "CRITICAL"

        action = (
            "Launch an immediate customer retention "
            "campaign using personalized offers, "
            "reminders, and re-engagement strategies."
        )


    elif (
        high_ratio >= 0.10
        or at_risk_ratio >= 0.40
    ):

        priority = "HIGH"

        action = (
            "Prioritize at-risk customers with "
            "targeted retention offers and "
            "personalized communication."
        )


    elif at_risk_ratio >= 0.20:

        priority = "MEDIUM"

        action = (
            "Use targeted reminders and personalized "
            "product recommendations."
        )


    else:

        priority = "LOW"

        action = (
            "Continue regular customer engagement "
            "and monitor inactivity risk."
        )


    # ========================================================
    # BUSINESS SUMMARY
    # ========================================================

    return {

        "total_customers":
            total,

        "high_risk":
            high,

        "medium_risk":
            medium,

        "low_risk":
            low,

        "customers_at_risk":
            at_risk,

        "at_risk_percentage":
            round(
                at_risk_percentage,
                2
            ),

        "retention_priority":
            priority,

        "recommended_action":
            action
    }

    # ============================================================
# STEP 36: MODEL PERFORMANCE
# ============================================================

@app.get("/model-performance")
def get_model_performance():

    performance_file = (
        DATA_PATH +
        r"\model_performance.json"
    )

    try:

        import json

        with open(
            performance_file,
            "r"
        ) as file:

            performance = json.load(file)

    except FileNotFoundError:

        raise HTTPException(
            status_code=404,
            detail=(
                "model_performance.json not found. "
                "Run Step 35 first."
            )
        )

    return performance