# Bank Customer Churn Risk Analytics

A financial analytics project focused on predicting customer churn risk for a retail bank and supporting proactive, high-value retention planning.

## Business Objective

This project estimates the probability that a customer will churn and translates that risk into a commercially useful retention score. The goal is to help banks identify high-risk customers, prioritize interventions, and protect revenue streams.

## What is Included

- Predictive churn modeling for bank customers
- Customer-level churn probability dashboard
- Risk classification by low/medium/high risk
- Revenue-at-risk estimation
- Feature importance and explainability
- Scenario simulation for retention planning
- Financial decision support for bank stakeholders

## Project Structure

```text
Bank-Customer-Churn-Risk-Analytics/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── bank_customer_churn.csv
├── artifacts/
│   ├── churn_model.pkl
│   └── metrics.json
└── src/
    ├── __init__.py
    ├── generate_data.py
    ├── train_model.py
    └── churn_pipeline.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the App

```bash
python src/generate_data.py
python src/train_model.py
streamlit run app.py
```

## Advanced Features Added

- Financial impact analysis with revenue-at-risk metrics
- Risk threshold control for retention campaigns
- High-priority customer ranking for relationship managers
- Business scenario simulation for churn prevention
- XGBoost support when installed
- Explainable feature importance dashboard

## Key Use Cases

- Early warning for likely churners
- Detecting customers with high balance and high churn risk
- Identifying product and engagement patterns linked to churn
- Prioritizing retention spending on customers with maximum impact

## Deliverables

- Predictive churn model
- Dashboard for bankers and managers
- Executive-friendly retention insights
- Risk-based scenario planning output
