from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

from src.generate_data import generate_synthetic_data, load_data
from src.generate_data import get_feature_importance

DATA_PATH = Path("data/bank_customer_churn.csv")
MODEL_PATH = Path("artifacts/churn_model.pkl")

FEATURE_COLUMNS = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "BalanceToSalaryRatio",
    "ProductDensity",
    "EngagementProductInteraction",
    "AgeTenureInteraction",
]


def ensure_model_ready():
    if not DATA_PATH.exists():
        generate_synthetic_data(save_path=str(DATA_PATH))

    if not MODEL_PATH.exists():
        from src.train_model import train_and_save_model

        df = load_data(DATA_PATH)
        train_and_save_model(df)

    return MODEL_PATH.exists()


def risk_label(probability: float) -> str:
    if probability >= 0.60:
        return "High Risk"
    if probability >= 0.35:
        return "Medium Risk"
    return "Low Risk"


st.set_page_config(page_title="Bank Customer Churn Risk Dashboard", layout="wide")

ensure_model_ready()

with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

raw_df = load_data(DATA_PATH)
score_df = raw_df.copy()
score_df["PredictedChurnProbability"] = model.predict_proba(score_df[FEATURE_COLUMNS])[:, 1]
score_df["RevenueAtRisk"] = score_df["Balance"] * score_df["PredictedChurnProbability"] * 0.08
score_df["RiskBand"] = score_df["PredictedChurnProbability"].apply(risk_label)

st.title("Bank Customer Churn Risk Dashboard")
st.caption("Financial analytics view of retail customer churn, retention priority, and business impact")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", f"{len(raw_df):,}")
col2.metric("Churn Rate", f"{raw_df['Exited'].mean() * 100:.1f}%")
col3.metric("Average Balance", f"${raw_df['Balance'].mean():,.0f}")
col4.metric("Revenue at Risk", f"${score_df['RevenueAtRisk'].sum():,.0f}")

st.sidebar.header("Retention Strategy Controls")
threshold = st.sidebar.slider("Risk threshold for retention campaign", 0.10, 0.90, 0.40, 0.05)
priority_budget = st.sidebar.slider("Priority budget % of high-risk customers", 5, 30, 15, 5)

st.subheader("Customer Risk Calculator")
with st.form("customer_form"):
    left_col, right_col = st.columns(2)

    with left_col:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
        geography = st.selectbox("Geography", ["France", "Spain", "Germany"], index=0)
        gender = st.selectbox("Gender", ["Male", "Female"], index=0)
        age = st.number_input("Age", min_value=18, max_value=90, value=42)
        tenure = st.number_input("Tenure (Years)", min_value=0, max_value=20, value=5)
        balance = st.number_input("Balance", min_value=0, max_value=250000, value=120000, step=1000)

    with right_col:
        products = st.number_input("Number of Products", min_value=1, max_value=4, value=2)
        has_credit_card = st.checkbox("Has Credit Card", value=True)
        active_member = st.checkbox("Is Active Member", value=True)
        estimated_salary = st.number_input("Estimated Salary", min_value=10000, max_value=200000, value=85000, step=1000)
        balance_to_salary = st.number_input("Balance to Salary Ratio", min_value=0.0, max_value=10.0, value=1.5, step=0.1)
        product_density = st.number_input("Product Density", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
        engagement_product = st.number_input("Engagement-Product Interaction", min_value=-5.0, max_value=5.0, value=0.2, step=0.1)
        age_tenure = st.number_input("Age-Tenure Interaction", min_value=-50.0, max_value=50.0, value=5.0, step=0.5)

    submitted = st.form_submit_button("Calculate Risk")

if submitted:
    customer_df = pd.DataFrame([
        {
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": products,
            "HasCrCard": int(has_credit_card),
            "IsActiveMember": int(active_member),
            "EstimatedSalary": estimated_salary,
            "BalanceToSalaryRatio": balance_to_salary,
            "ProductDensity": product_density,
            "EngagementProductInteraction": engagement_product,
            "AgeTenureInteraction": age_tenure,
        }
    ])

    prob = model.predict_proba(customer_df[FEATURE_COLUMNS])[0, 1]
    label = risk_label(prob)

    st.subheader("Current Customer Risk")
    st.metric("Predicted Churn Probability", f"{prob * 100:.2f}%")
    st.metric("Risk Category", label)
    st.progress(int(prob * 100))

st.subheader("Financial Impact by Risk Band")
segment_summary = score_df.groupby("RiskBand").agg(
    Customers=("CustomerId", "count"),
    AvgRisk=("PredictedChurnProbability", "mean"),
    RevenueAtRisk=("RevenueAtRisk", "sum"),
).reset_index()
st.bar_chart(segment_summary.set_index("RiskBand")["RevenueAtRisk"])
st.dataframe(segment_summary, use_container_width=True)

st.subheader("Probability Distribution")
hist_col, summary_col = st.columns([2, 1])
with hist_col:
    st.bar_chart(score_df["PredictedChurnProbability"].value_counts().sort_index())
with summary_col:
    st.write(score_df["PredictedChurnProbability"].describe().round(3))

st.subheader("Feature Importance Dashboard")
feature_importance = get_feature_importance(model)
feature_chart = feature_importance.head(10).set_index("feature")["importance"]
st.bar_chart(feature_chart)
st.dataframe(feature_importance.head(15), use_container_width=True)

st.subheader("What-If Scenario Simulator")
scenario_col1, scenario_col2 = st.columns(2)
with scenario_col1:
    scenario_balance = st.slider("Adjust Balance", min_value=0, max_value=250000, value=120000, step=5000)
    scenario_products = st.slider("Adjust Number of Products", min_value=1, max_value=4, value=2)
    scenario_active = st.checkbox("Set Customer as Active", value=True)
with scenario_col2:
    scenario_salary = st.slider("Adjust Salary", min_value=10000, max_value=200000, value=85000, step=5000)
    scenario_age = st.slider("Adjust Age", min_value=18, max_value=90, value=42)
    scenario_tenure = st.slider("Adjust Tenure", min_value=0, max_value=20, value=5)

scenario_df = pd.DataFrame([
    {
        "CreditScore": 650,
        "Geography": "France",
        "Gender": "Female",
        "Age": scenario_age,
        "Tenure": scenario_tenure,
        "Balance": scenario_balance,
        "NumOfProducts": scenario_products,
        "HasCrCard": 1,
        "IsActiveMember": int(scenario_active),
        "EstimatedSalary": scenario_salary,
        "BalanceToSalaryRatio": scenario_balance / max(scenario_salary, 1),
        "ProductDensity": 0.7,
        "EngagementProductInteraction": 0.2,
        "AgeTenureInteraction": scenario_age * scenario_tenure,
    }
])

scenario_prob = model.predict_proba(scenario_df[FEATURE_COLUMNS])[0, 1]
scenario_label = risk_label(scenario_prob)
st.metric("Scenario Churn Probability", f"{scenario_prob * 100:.2f}%")
st.metric("Scenario Risk", scenario_label)

st.subheader("Retention Priority List")
high_risk = score_df[score_df["PredictedChurnProbability"] >= threshold].sort_values(
    ["PredictedChurnProbability", "RevenueAtRisk"], ascending=[False, False]
).head(12)

priority_columns = [
    "CustomerId",
    "Geography",
    "Age",
    "Balance",
    "NumOfProducts",
    "IsActiveMember",
    "PredictedChurnProbability",
    "RevenueAtRisk",
    "RiskBand",
]

st.dataframe(high_risk[priority_columns].round({"PredictedChurnProbability": 4, "RevenueAtRisk": 2}), use_container_width=True)

st.subheader("Business Strategy Summary")
eligible_customers = score_df[score_df["PredictedChurnProbability"] >= threshold]
retention_journey_cost = eligible_customers["RevenueAtRisk"].sum() * (priority_budget / 100)

st.metric("Customers in Retention Campaign", f"{len(eligible_customers):,}")
st.metric("Campaign Priority Budget", f"${retention_journey_cost:,.0f}")
st.metric("Projected Risk Coverage", f"{eligible_customers['PredictedChurnProbability'].mean() * 100:.1f}%")
