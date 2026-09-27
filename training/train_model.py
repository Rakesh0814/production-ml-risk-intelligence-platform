from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "credit_risk.csv"
ARTIFACT_DIR = ROOT / "app" / "artifacts"

MODEL_PATH = ARTIFACT_DIR / "xgboost_model.json"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"
FEATURE_ORDER_PATH = ARTIFACT_DIR / "feature_order.json"
DEFAULTS_PATH = ARTIFACT_DIR / "feature_defaults.json"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} does not exist. Run the ETL step first."
        )

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["default_risk"])
    y = df["default_risk"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = XGBClassifier(
        n_estimators=350,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        reg_alpha=0.05,
        reg_lambda=1.2,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=4,
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    metrics = {
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "precision": round(
            float(precision_score(y_test, predictions, zero_division=0)), 4
        ),
        "recall": round(
            float(recall_score(y_test, predictions, zero_division=0)), 4
        ),
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)

    FEATURE_ORDER_PATH.write_text(
        json.dumps(list(X.columns), indent=2),
        encoding="utf-8",
    )

    defaults = {}
    for column in X.columns:
        if column in {"duration", "amount"}:
            defaults[column] = float(X_train[column].median())
        else:
            defaults[column] = float(X_train[column].mode().iloc[0])

    DEFAULTS_PATH.write_text(
        json.dumps(defaults, indent=2),
        encoding="utf-8",
    )

    metadata = {
        "model": "XGBoost",
        "dataset": "UCI South German Credit",
        "dataset_rows": int(len(df)),
        "features": int(X.shape[1]),
        "sensitive_features_excluded": [
            "personal_status_sex",
            "age",
            "foreign_worker",
        ],
        **metrics,
    }

    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nTraining complete")
    print(json.dumps(metadata, indent=2))
    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
