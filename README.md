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

## Analysis and Findings

Customer churn in the banking sector is a critical financial risk, as it directly reduces customer lifetime value, weakens revenue stability, and limits long-term growth opportunities. The primary purpose of this project was to move beyond traditional descriptive churn analysis and build a predictive model capable of identifying customers with a high probability of leaving before they actually do. By combining behavioral, engagement, and product-related variables, the model provides a proactive decision-making tool for banks to reduce churn and improve retention efficiency.

The project used a structured analytical framework beginning with data preprocessing, feature engineering, and model training. Non-informative identifiers such as customer ID and surname were removed, categorical variables were encoded, and key derived features such as balance-to-salary ratio, product density, engagement-product interaction, and age-tenure interaction were introduced to improve model performance and business relevance. The model was trained using a stratified train-test split to preserve the class distribution of churners and non-churners, ensuring that the predictive outputs remained realistic and reliable.

The churn risk model successfully identified the main behavioral drivers of customer attrition. The strongest indicators were lower engagement, reduced product utilization, and lower customer activity levels. Customers who were inactive, held fewer banking products, or had lower engagement with the bank were found to be significantly more likely to churn. In addition, age and balance patterns indicated that older customers with lower product diversity and weaker engagement were at higher risk. These findings align with real banking behavior, where customers often remain loyal when they are actively using services, hold multiple products, and maintain a strong relationship with the institution.

A key insight from the analysis is that churn is not driven only by demographic factors but is strongly influenced by relationship strength and product usage. This is especially important for financial institutions, because it shows that retention strategies should focus on engagement and relationship management rather than only on customer demographics. Customers with low product ownership or inconsistent activity represent a strategic opportunity: targeted retention programs, personalized offers, and proactive relationship management can reduce attrition before it becomes financially damaging.

From a business perspective, the predictive model adds value by converting churn probability into a usable risk score. This allows banks to prioritize customers based on both risk level and business impact. Rather than applying the same retention strategy to all customers, banks can focus their efforts on high-risk, high-value segments where intervention is most likely to deliver returns. This improves the efficiency of marketing campaigns, reduces unnecessary costs, and supports better decision-making across customer relationship management teams.

The model also demonstrates the importance of explainability in financial analytics. Understanding which variables drive churn helps decision-makers trust the output and act on it. Feature importance analysis shows that customer activity, product count, balance, and account engagement are among the most influential variables. This is important not only for business performance but also for regulatory and governance considerations, where transparent, interpretable decisions are increasingly required.

In summary, the project shows that churn prediction can be effectively transformed into a practical financial risk management tool. The system identifies likely churners early, highlights the behavioral drivers behind churn, and enables banks to take proactive retention measures. By focusing on customer engagement and product utilization, the bank can reduce churn, protect revenue, and improve long-term customer loyalty. The final model provides a useful foundation for real-world deployment in retention strategy, customer segmentation, and risk-based decision support.
