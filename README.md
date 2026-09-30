# Bank Customer Churn Risk Analytics

A practical financial analytics project focused on predicting customer churn risk for a retail bank and helping business teams prioritize retention actions.

## Project Objective

This project builds a churn prediction and risk-scoring system to identify customers most likely to leave, quantify the business impact, and support proactive retention strategies.

## Business Value

- Reduce customer attrition and protect revenue
- Rank customers by churn probability and business risk
- Support targeted retention campaigns
- Improve explainability for stakeholders and regulators
- Provide a dashboard for scenario analysis and risk monitoring

## Dataset

The project uses a banking customer dataset with the following fields:

- CustomerId
- Surname
- CreditScore
- Geography
- Gender
- Age
- Tenure
- Balance
- NumOfProducts
- HasCrCard
- IsActiveMember
- EstimatedSalary
- Exited (target variable)

A synthetic dataset is generated automatically if no local file is available.

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
    ├── churn_pipeline.py
    ├── generate_data.py
    └── train_model.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the Project

### 1) Generate dataset

```bash
python src/generate_data.py
```

### 2) Train the churn model

```bash
python src/train_model.py
```

### 3) Launch the Streamlit dashboard

```bash
streamlit run app.py
```

## Dashboard Features

- Customer churn risk calculator
- Probability distribution visualization
- Feature importance analysis
- What-if retention scenario simulator
- Business-oriented risk assessment and interpretation

## Model Strategy

- Data preprocessing and feature engineering
- Encoding of categorical variables
- Train-test split with class stratification
- Baseline and ensemble models
- Performance evaluation using precision, recall, F1-score, and ROC-AUC

## Recommended Output for Stakeholders

- Churn probability (0 to 1)
- Risk band: Low / Medium / High
- Ranked list of high-risk customers
- Key churn drivers identified from model explanations
- Scenario simulation for intervention planning

## Notes

This repository is designed to be easy to run locally and suitable for a financial analytics / banking risk project presentation.
