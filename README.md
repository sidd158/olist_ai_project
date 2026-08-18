# Olist Customer AI

## Customer Inactivity Prediction & Retention System

An end-to-end Machine Learning application that predicts customer inactivity risk and provides actionable customer-retention recommendations.

The project combines Machine Learning, SHAP explainability, FastAPI, and React to create an interactive customer-risk analytics dashboard.

---

## Features

- Customer inactivity prediction
- Customer risk classification
- High / Medium / Low risk segmentation
- Customer-level prediction API
- SHAP-based explainable AI
- High-risk customer identification
- Business retention recommendations
- Risk distribution analytics
- Model performance dashboard
- Confusion matrix
- FastAPI backend
- React frontend
- REST API integration
- Interactive customer analysis

---

## System Architecture

```text
                    Olist Dataset
                         |
                         v
                Data Preprocessing
                         |
                         v
                 Feature Engineering
                         |
                         v
                  ML Model Training
                         |
                         v
                   Final ML Model
                         |
              +----------+----------+
              |                     |
              v                     v
        Customer Prediction       SHAP
              |                     |
              +----------+----------+
                         |
                         v
                    FastAPI
                         |
                         v
                  React Dashboard
                         |
          +--------------+--------------+
          |              |              |
          v              v              v
      Prediction      Analytics       Explainability