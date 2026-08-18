import { useEffect, useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function App() {

  // ============================================================
  // STATES
  // ============================================================

  const [customerIndex, setCustomerIndex] = useState(0);

  const [result, setResult] = useState(null);

  const [explanation, setExplanation] = useState(null);

  const [highRiskCustomers, setHighRiskCustomers] = useState([]);

  const [businessSummary, setBusinessSummary] = useState(null);

  const [modelPerformance, setModelPerformance] = useState(null);

  const [loading, setLoading] = useState(false);

  const [shapLoading, setShapLoading] = useState(false);

  const [error, setError] = useState("");

  const [stats, setStats] = useState({
    total: 0,
    high: 0,
    medium: 0,
    low: 0,
    highPercentage: 0,
    mediumPercentage: 0,
    lowPercentage: 0
  });


  // ============================================================
  // LOAD STATISTICS
  // ============================================================

  useEffect(() => {

    const loadStats = async () => {

      try {

        const response = await fetch(
          `${API_URL}/stats`
        );

        if (!response.ok) {
          throw new Error(
            "Statistics API failed"
          );
        }

        const data = await response.json();

        setStats({
          total: data.total_customers,
          high: data.high_risk,
          medium: data.medium_risk,
          low: data.low_risk,
          highPercentage: data.high_percentage,
          mediumPercentage: data.medium_percentage,
          lowPercentage: data.low_percentage
        });

      } catch (error) {

        console.error(
          "Statistics error:",
          error
        );

      }

    };

    loadStats();

  }, []);


  // ============================================================
  // LOAD HIGH-RISK CUSTOMERS
  // ============================================================

  useEffect(() => {

    const loadHighRiskCustomers = async () => {

      try {

        const response = await fetch(
          `${API_URL}/high-risk-customers`
        );

        if (!response.ok) {
          throw new Error(
            "High-risk API failed"
          );
        }

        const data =
          await response.json();

        setHighRiskCustomers(
          data.customers || []
        );

      } catch (error) {

        console.error(
          "High-risk customer error:",
          error
        );

      }

    };

    loadHighRiskCustomers();

  }, []);


  // ============================================================
  // LOAD BUSINESS SUMMARY
  // ============================================================

  useEffect(() => {

    const loadBusinessSummary = async () => {

      try {

        const response = await fetch(
          `${API_URL}/business-summary`
        );

        if (!response.ok) {
          throw new Error(
            "Business summary API failed"
          );
        }

        const data =
          await response.json();

        setBusinessSummary(data);

      } catch (error) {

        console.error(
          "Business summary error:",
          error
        );

      }

    };

    loadBusinessSummary();

  }, []);


  // ============================================================
  // LOAD MODEL PERFORMANCE
  // ============================================================

  useEffect(() => {

    const loadModelPerformance = async () => {

      try {

        const response = await fetch(
          `${API_URL}/model-performance`
        );

        if (!response.ok) {
          throw new Error(
            "Model performance API failed"
          );
        }

        const data =
          await response.json();

        setModelPerformance(data);

      } catch (error) {

        console.error(
          "Model performance error:",
          error
        );

      }

    };

    loadModelPerformance();

  }, []);


  // ============================================================
  // PREDICT CUSTOMER + SHAP
  // ============================================================

  const predictCustomer = async () => {

    setLoading(true);

    setShapLoading(false);

    setError("");

    setResult(null);

    setExplanation(null);

    try {

      // --------------------------------------------------------
      // PREDICTION
      // --------------------------------------------------------

      const response = await fetch(
        `${API_URL}/predict/${customerIndex}`
      );

      if (!response.ok) {

        throw new Error(
          `Prediction API returned ${response.status}`
        );

      }

      const data =
        await response.json();

      setResult(data);

      setLoading(false);


      // --------------------------------------------------------
      // SHAP
      // --------------------------------------------------------

      setShapLoading(true);

      try {

        const shapResponse = await fetch(
          `${API_URL}/shap-explanation/${customerIndex}`
        );

        if (!shapResponse.ok) {

          throw new Error(
            `SHAP API returned ${shapResponse.status}`
          );

        }

        const shapData =
          await shapResponse.json();

        setExplanation(shapData);

      } catch (shapError) {

        console.error(
          "SHAP explanation error:",
          shapError
        );

        setExplanation(null);

      } finally {

        setShapLoading(false);

      }

    } catch (error) {

      console.error(
        "Prediction error:",
        error
      );

      setError(
        "Unable to connect to the prediction API. Make sure FastAPI is running."
      );

      setLoading(false);

      setShapLoading(false);

    }

  };


  // ============================================================
  // RISK CLASS
  // ============================================================

  const getRiskClass = (risk) => {

    if (risk === "HIGH") {
      return "high";
    }

    if (risk === "MEDIUM") {
      return "medium";
    }

    return "low";

  };


  // ============================================================
  // ANALYZE CUSTOMER FROM TABLE
  // ============================================================

  const analyzeCustomer = (index) => {

    setCustomerIndex(index);

    window.scrollTo({
      top: 0,
      behavior: "smooth"
    });

  };


  // ============================================================
  // RENDER
  // ============================================================

  return (

    <div className="dashboard">


      {/* ======================================================
          HEADER
      ====================================================== */}

      <header className="header">

        <div>

          <h1>
            Olist Customer AI
          </h1>

          <p>
            Customer Inactivity Prediction & Retention System
          </p>

        </div>


        <div className="status">

          <span></span>

          API Connected

        </div>

      </header>


      {/* ======================================================
          STATISTICS
      ====================================================== */}

      <section className="stats">


        <div className="stat-card">

          <span>
            Total Customers
          </span>

          <strong>
            {stats.total}
          </strong>

        </div>


        <div className="stat-card">

          <span>
            High Risk
          </span>

          <strong className="high-text">
            {stats.high}
          </strong>

        </div>


        <div className="stat-card">

          <span>
            Medium Risk
          </span>

          <strong className="medium-text">
            {stats.medium}
          </strong>

        </div>


        <div className="stat-card">

          <span>
            Low Risk
          </span>

          <strong className="low-text">
            {stats.low}
          </strong>

        </div>


      </section>


      {/* ======================================================
          MAIN
      ====================================================== */}

      <main className="main">


        {/* ====================================================
            AI BUSINESS SUMMARY
        ==================================================== */}

        {businessSummary && (

          <section className="panel">

            <h2>
              AI Business Summary
            </h2>

            <p className="description">
              Overall customer retention analysis
              generated from machine learning predictions.
            </p>


            <div className="business-summary">


              <div className="summary-card">

                <span>
                  Customers At Risk
                </span>

                <strong>
                  {
                    businessSummary.customers_at_risk
                  }
                </strong>

                <small>
                  {
                    businessSummary.at_risk_percentage
                  }%
                  {" "}of customers
                </small>

              </div>


              <div className="summary-card">

                <span>
                  High Risk
                </span>

                <strong className="high-text">
                  {
                    businessSummary.high_risk
                  }
                </strong>

              </div>


              <div className="summary-card">

                <span>
                  Medium Risk
                </span>

                <strong className="medium-text">
                  {
                    businessSummary.medium_risk
                  }
                </strong>

              </div>


              <div className="summary-card">

                <span>
                  Retention Priority
                </span>

                <strong
                  className={
                    businessSummary.retention_priority ===
                    "CRITICAL"
                      ? "high-text"
                      : businessSummary.retention_priority ===
                        "HIGH"
                      ? "medium-text"
                      : "low-text"
                  }
                >
                  {
                    businessSummary.retention_priority
                  }
                </strong>

              </div>


            </div>


            <div className="business-action">

              <h3>
                Recommended Business Action
              </h3>

              <p>
                {
                  businessSummary.recommended_action
                }
              </p>

            </div>

          </section>

        )}


        {/* ====================================================
            MODEL PERFORMANCE
        ==================================================== */}

        {modelPerformance && (

          <section className="panel">

            <h2>
              Final Model Performance
            </h2>

            <p className="description">
              Evaluation metrics of the final machine
              learning model on the test dataset.
            </p>


            <div className="performance-grid">


              {/* ACCURACY */}

              <div className="performance-card">

                <span>
                  Accuracy
                </span>

                <strong>
                  {
                    (
                      modelPerformance.accuracy *
                      100
                    ).toFixed(2)
                  }%
                </strong>

              </div>


              {/* PRECISION */}

              <div className="performance-card">

                <span>
                  Precision
                </span>

                <strong>
                  {
                    (
                      modelPerformance.precision *
                      100
                    ).toFixed(2)
                  }%
                </strong>

              </div>


              {/* RECALL */}

              <div className="performance-card">

                <span>
                  Recall
                </span>

                <strong>
                  {
                    (
                      modelPerformance.recall *
                      100
                    ).toFixed(2)
                  }%
                </strong>

              </div>


              {/* F1 */}

              <div className="performance-card">

                <span>
                  F1 Score
                </span>

                <strong>
                  {
                    (
                      modelPerformance.f1_score *
                      100
                    ).toFixed(2)
                  }%
                </strong>

              </div>


              {/* ROC AUC */}

              <div className="performance-card">

                <span>
                  ROC-AUC
                </span>

                <strong>
                  {
                    (
                      modelPerformance.roc_auc *
                      100
                    ).toFixed(2)
                  }%
                </strong>

              </div>


            </div>


            {/* ==================================================
                CONFUSION MATRIX
            ================================================== */}

            {modelPerformance.confusion_matrix &&
              modelPerformance.confusion_matrix.length >= 2 && (

              <div className="confusion-section">

                <h3>
                  Confusion Matrix
                </h3>


                <div className="confusion-matrix">


                  <div className="matrix-cell header-cell">
                    Actual / Predicted
                  </div>

                  <div className="matrix-cell header-cell">
                    Active
                  </div>

                  <div className="matrix-cell header-cell">
                    Inactive
                  </div>


                  <div className="matrix-cell header-cell">
                    Active
                  </div>

                  <div className="matrix-cell">
                    {
                      modelPerformance
                        .confusion_matrix[0][0]
                    }
                  </div>

                  <div className="matrix-cell">
                    {
                      modelPerformance
                        .confusion_matrix[0][1]
                    }
                  </div>


                  <div className="matrix-cell header-cell">
                    Inactive
                  </div>

                  <div className="matrix-cell">
                    {
                      modelPerformance
                        .confusion_matrix[1][0]
                    }
                  </div>

                  <div className="matrix-cell">
                    {
                      modelPerformance
                        .confusion_matrix[1][1]
                    }
                  </div>


                </div>

              </div>

            )}

          </section>

        )}


        {/* ====================================================
            CUSTOMER PREDICTION
        ==================================================== */}

        <section className="panel">

          <h2>
            Customer Prediction
          </h2>

          <p className="description">
            Enter a customer index to analyze inactivity risk.
          </p>


          <div className="input-area">

            <input
              type="number"
              min="0"
              max={
                stats.total > 0
                  ? stats.total - 1
                  : 16875
              }
              value={customerIndex}
              onChange={(e) =>
                setCustomerIndex(
                  Number(e.target.value)
                )
              }
            />


            <button
              onClick={predictCustomer}
              disabled={
                loading ||
                shapLoading
              }
            >

              {loading
                ? "Predicting..."
                : shapLoading
                ? "Generating SHAP..."
                : "Analyze Customer"}

            </button>

          </div>


          {error && (

            <div className="error">

              {error}

            </div>

          )}

        </section>


        {/* ====================================================
            CUSTOMER RESULT
        ==================================================== */}

        {result && (

          <section className="results">


            {/* RESULT HEADER */}

            <div className="result-header">

              <div>

                <span>
                  Customer
                </span>

                <h2>
                  #{result.customer_index}
                </h2>

              </div>


              <div
                className={
                  `risk-badge ${
                    getRiskClass(
                      result.risk_level
                    )
                  }`
                }
              >

                {result.risk_level} RISK

              </div>

            </div>


            {/* METRICS */}

            <div className="metrics">


              <div className="metric">

                <span>
                  Inactivity Probability
                </span>

                <strong>
                  {
                    result.inactive_probability_percent
                  }%
                </strong>


                <div className="progress">

                  <div
                    className={
                      `progress-bar ${
                        getRiskClass(
                          result.risk_level
                        )
                      }`
                    }
                    style={{
                      width:
                        `${Math.min(
                          result.inactive_probability_percent,
                          100
                        )}%`
                    }}
                  />

                </div>

              </div>


              <div className="metric">

                <span>
                  Prediction
                </span>

                <strong>
                  {
                    result.inactive_prediction === 1
                      ? "Likely Inactive"
                      : "Likely Active"
                  }
                </strong>

              </div>


            </div>


            {/* RECOMMENDATION */}

            <div className="recommendation">

              <h3>
                Recommended Action
              </h3>

              <p>
                {result.recommendation}
              </p>

            </div>


            {/* =================================================
                SHAP EXPLANATION
            ================================================= */}

            <div className="explanation">


              <div className="explanation-header">

                <div>

                  <h3>
                    Why Is This Customer at Risk?
                  </h3>

                  <p className="description">
                    Top individual factors influencing
                    this customer's prediction.
                  </p>

                </div>


                {shapLoading && (

                  <span className="shap-status">
                    Calculating SHAP...
                  </span>

                )}

              </div>


              {/* SHAP FACTORS */}

              {!shapLoading &&
                explanation &&
                explanation.factors &&
                explanation.factors.length > 0 && (

                <div className="factor-list">

                  {explanation.factors.map(
                    (factor, index) => {

                      const positive =
                        factor.shap_value > 0;

                      return (

                        <div
                          className="factor"
                          key={
                            `${factor.feature}-${index}`
                          }
                        >


                          <div className="factor-info">

                            <strong>
                              {factor.feature}
                            </strong>

                            <span>
                              Customer value:{" "}
                              {factor.customer_value}
                            </span>

                          </div>


                          <div className="factor-impact">

                            <span className="shap-value">

                              SHAP:{" "}
                              {factor.shap_value}

                            </span>


                            <span
                              className={
                                positive
                                  ? "risk-increase"
                                  : "risk-decrease"
                              }
                            >

                              {positive
                                ? "↑ Increases inactivity risk"
                                : "↓ Decreases inactivity risk"}

                            </span>

                          </div>


                        </div>

                      );

                    }
                  )}

                </div>

              )}


              {/* SHAP UNAVAILABLE */}

              {!shapLoading &&
                !explanation && (

                <div className="shap-warning">

                  <strong>
                    SHAP explanation unavailable
                  </strong>

                  <p>
                    The prediction was successful,
                    but an individual SHAP explanation
                    could not be generated.
                  </p>

                </div>

              )}


            </div>


          </section>

        )}


        {/* ====================================================
            HIGH-RISK CUSTOMERS
        ==================================================== */}

        <section className="panel">

          <h2>
            Top High-Risk Customers
          </h2>

          <p className="description">
            Customers with the highest probability
            of becoming inactive.
          </p>


          <div className="customer-table">


            <div className="table-header">

              <span>
                Customer
              </span>

              <span>
                Probability
              </span>

              <span>
                Risk
              </span>

              <span>
                Action
              </span>

            </div>


            {highRiskCustomers.length > 0 ? (

              highRiskCustomers.map(
                (customer) => (

                  <div
                    className="table-row"
                    key={
                      customer.customer_index
                    }
                  >


                    <span>
                      #{customer.customer_index}
                    </span>


                    <span>
                      {
                        customer.inactive_probability_percent
                      }%
                    </span>


                    <span>

                      <span className="risk-badge high">
                        HIGH
                      </span>

                    </span>


                    <button
                      className="view-button"
                      onClick={() =>
                        analyzeCustomer(
                          customer.customer_index
                        )
                      }
                    >
                      Analyze
                    </button>


                  </div>

                )

              )

            ) : (

              <div className="empty-message">

                No high-risk customers found.

              </div>

            )}


          </div>

        </section>


        {/* ====================================================
            RISK DISTRIBUTION
        ==================================================== */}

        <section className="panel">

          <h2>
            Risk Distribution
          </h2>


          {/* HIGH */}

          <div className="bar-row">

            <div className="bar-label">

              <span>
                High Risk
              </span>

              <strong>
                {stats.highPercentage}%
              </strong>

            </div>


            <div className="bar">

              <div
                className="bar-fill high"
                style={{
                  width:
                    `${stats.highPercentage}%`
                }}
              />

            </div>

          </div>


          {/* MEDIUM */}

          <div className="bar-row">

            <div className="bar-label">

              <span>
                Medium Risk
              </span>

              <strong>
                {stats.mediumPercentage}%
              </strong>

            </div>


            <div className="bar">

              <div
                className="bar-fill medium"
                style={{
                  width:
                    `${stats.mediumPercentage}%`
                }}
              />

            </div>

          </div>


          {/* LOW */}

          <div className="bar-row">

            <div className="bar-label">

              <span>
                Low Risk
              </span>

              <strong>
                {stats.lowPercentage}%
              </strong>

            </div>


            <div className="bar">

              <div
                className="bar-fill low"
                style={{
                  width:
                    `${stats.lowPercentage}%`
                }}
              />

            </div>

          </div>


        </section>


      </main>


    </div>

  );

}

export default App;