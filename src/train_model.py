from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None

from src.generate_data import generate_synthetic_data, get_feature_list, load_data, save_json

MODEL_PATH = Path("artifacts/churn_model.pkl")
METRICS_PATH = Path("artifacts/metrics.json")


def build_preprocessor() -> ColumnTransformer:
    _, numeric_features, categorical_features = get_feature_list()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def get_model_candidates():
    candidates = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
        ("Random Forest", RandomForestClassifier(n_estimators=220, random_state=42, class_weight="balanced")),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=42)),
    ]

    if xgb is not None:
        candidates.append(("XGBoost", xgb.XGBClassifier(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric="logloss",
        )))

    return candidates


def train_and_save_model(df: pd.DataFrame) -> dict:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    drop_columns, _, _ = get_feature_list()
    feature_columns = [col for col in df.columns if col not in drop_columns]
    X = df[feature_columns]
    y = df["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    best_pipeline = None
    best_name = ""
    best_metrics = {}
    best_score = -1

    for model_name, estimator in get_model_candidates():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", estimator),
            ]
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob),
        }

        if metrics["roc_auc"] > best_score:
            best_score = metrics["roc_auc"]
            best_pipeline = pipeline
            best_name = model_name
            best_metrics = metrics

    if best_pipeline is None:
        raise ValueError("No model candidates were trained.")

    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(best_pipeline, model_file)

    metrics_payload = {
        "best_model": best_name,
        "roc_auc": round(best_metrics["roc_auc"], 4),
        "accuracy": round(best_metrics["accuracy"], 4),
        "precision": round(best_metrics["precision"], 4),
        "recall": round(best_metrics["recall"], 4),
        "f1": round(best_metrics["f1"], 4),
    }

    save_json(metrics_payload, METRICS_PATH)
    return metrics_payload


if __name__ == "__main__":
    data_path = Path("data/bank_customer_churn.csv")
    if not data_path.exists():
        generate_synthetic_data(save_path=data_path)

    df = load_data(data_path)
    print(train_and_save_model(df))
