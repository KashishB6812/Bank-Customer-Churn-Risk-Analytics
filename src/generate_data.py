from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_data(n: int = 2500, save_path: str | Path = "data/bank_customer_churn.csv") -> pd.DataFrame:
    rng = np.random.default_rng(42)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    data = pd.DataFrame({
        "CustomerId": np.arange(1, n + 1),
        "Surname": [f"Customer{idx}" for idx in range(1, n + 1)],
        "CreditScore": rng.integers(350, 850, size=n),
        "Geography": rng.choice(["France", "Spain", "Germany"], size=n, p=[0.52, 0.28, 0.20]),
        "Gender": rng.choice(["Male", "Female"], size=n, p=[0.52, 0.48]),
        "Age": rng.integers(18, 70, size=n),
        "Tenure": rng.integers(0, 11, size=n),
        "Balance": rng.uniform(0, 200000, size=n),
        "NumOfProducts": rng.integers(1, 5, size=n),
        "HasCrCard": rng.integers(0, 2, size=n),
        "IsActiveMember": rng.integers(0, 2, size=n),
        "EstimatedSalary": rng.uniform(15000, 200000, size=n),
    })

    data["BalanceToSalaryRatio"] = data["Balance"] / (data["EstimatedSalary"] + 1)
    data["ProductDensity"] = data["NumOfProducts"] / (data["Tenure"] + 1)
    data["EngagementProductInteraction"] = data["NumOfProducts"] * data["IsActiveMember"]
    data["AgeTenureInteraction"] = data["Age"] * data["Tenure"]

    risk_components = (
        0.03 * data["Age"]
        + 0.00002 * data["Balance"]
        + 0.8 * data["BalanceToSalaryRatio"]
        + 0.8 * (1 - data["IsActiveMember"])
        + 0.3 * (data["NumOfProducts"] <= 1).astype(int)
        + 0.4 * (data["Age"] > 55).astype(int)
        - 0.6 * data["IsActiveMember"]
        - 0.5 * data["NumOfProducts"]
        + 0.3 * data["ProductDensity"]
        + 0.2 * data["AgeTenureInteraction"] / 100
    )

    unscaled_probability = 1 / (1 + np.exp(-(-6.5 + risk_components)))
    churn_probability = np.clip(unscaled_probability, 0.02, 0.95)
    data["Exited"] = rng.binomial(1, churn_probability).astype(int)

    data["Geography"] = data["Geography"].astype(str)
    data["Gender"] = data["Gender"].astype(str)
    data = data.round({
        "CreditScore": 0,
        "Balance": 2,
        "EstimatedSalary": 2,
        "BalanceToSalaryRatio": 4,
        "ProductDensity": 4,
        "EngagementProductInteraction": 2,
        "AgeTenureInteraction": 2,
    })

    data.to_csv(save_path, index=False)
    return data


def load_data(path: str | Path = "data/bank_customer_churn.csv") -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return generate_synthetic_data(save_path=path)
    return pd.read_csv(path)


def get_feature_list() -> tuple[list[str], list[str], list[str]]:
    drop_columns = ["CustomerId", "Surname", "Exited"]
    numeric_features = [
        "CreditScore",
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
    categorical_features = ["Geography", "Gender"]
    return drop_columns, numeric_features, categorical_features


def get_feature_importance(model) -> pd.DataFrame:
    final_model = model.named_steps["model"]
    transformed_columns = model.named_steps["preprocessor"].get_feature_names_out()

    if hasattr(final_model, "coef_"):
        importance = np.abs(final_model.coef_[0])
    elif hasattr(final_model, "feature_importances_"):
        importance = final_model.feature_importances_
    else:
        raise ValueError("Model does not expose coefficients or feature importances.")

    importance_df = pd.DataFrame({
        "feature": transformed_columns,
        "importance": importance,
    })

    return importance_df.sort_values("importance", ascending=False).reset_index(drop=True)


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
