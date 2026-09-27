from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

from .config import (
    MODEL_PATH,
    METADATA_PATH,
    FEATURE_ORDER_PATH,
    DEFAULTS_PATH,
)

FEATURE_LABELS = {
    "status": "Checking account status",
    "duration": "Credit duration",
    "credit_history": "Credit history",
    "purpose": "Credit purpose",
    "amount": "Credit amount",
    "savings": "Savings level",
    "employment_duration": "Employment duration",
    "installment_rate": "Installment burden",
    "other_debtors": "Co-applicant / guarantor",
    "present_residence": "Residence duration",
    "property": "Property profile",
    "other_installment_plans": "Other installment plans",
    "housing": "Housing status",
    "number_credits": "Existing bank credits",
    "job": "Job category",
    "people_liable": "Dependants",
    "telephone": "Registered telephone",
}


class ModelService:
    def __init__(self):
        self.model = None
        self.explainer = None
        self.metadata = {}
        self.feature_order = []
        self.defaults = {}
        self.load()

    def load(self):
        if not MODEL_PATH.exists():
            return

        self.model = XGBClassifier()
        self.model.load_model(MODEL_PATH)

        if METADATA_PATH.exists():
            self.metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

        if FEATURE_ORDER_PATH.exists():
            self.feature_order = json.loads(
                FEATURE_ORDER_PATH.read_text(encoding="utf-8")
            )

        if DEFAULTS_PATH.exists():
            self.defaults = json.loads(DEFAULTS_PATH.read_text(encoding="utf-8"))

        self.explainer = shap.TreeExplainer(self.model)

    @property
    def loaded(self) -> bool:
        return self.model is not None and bool(self.feature_order)

    def predict(self, payload: dict):
        if not self.loaded:
            raise RuntimeError(
                "Trained model artifacts not found. Run: train_windows.bat"
            )

        row = dict(self.defaults)
        row.update(payload)

        frame = pd.DataFrame(
            [[row.get(feature, 0) for feature in self.feature_order]],
            columns=self.feature_order,
        )

        default_probability = float(self.model.predict_proba(frame)[0, 1])

        if default_probability < 0.30:
            risk_band = "LOW"
        elif default_probability < 0.60:
            risk_band = "MODERATE"
        else:
            risk_band = "HIGH"

        shap_values = self.explainer(frame)

        values = np.asarray(shap_values.values)
        if values.ndim == 3:
            values = values[:, :, -1]
        values = values[0]

        drivers = []
        for feature, shap_value, feature_value in zip(
            self.feature_order,
            values,
            frame.iloc[0].tolist(),
        ):
            drivers.append(
                {
                    "feature": feature,
                    "label": FEATURE_LABELS.get(feature, feature),
                    "value": float(feature_value),
                    "impact": float(shap_value),
                    "direction": (
                        "increases risk"
                        if shap_value > 0
                        else "reduces risk"
                    ),
                }
            )

        drivers.sort(key=lambda item: abs(item["impact"]), reverse=True)

        return {
            "default_probability": round(default_probability, 4),
            "repayment_probability": round(1.0 - default_probability, 4),
            "risk_band": risk_band,
            "decision_threshold": 0.50,
            "predicted_class": (
                "HIGHER DEFAULT RISK"
                if default_probability >= 0.50
                else "LOWER DEFAULT RISK"
            ),
            "top_drivers": drivers[:5],
            "model": self.metadata.get("model", "XGBoost"),
            "dataset": self.metadata.get("dataset", "South German Credit"),
            "roc_auc": self.metadata.get("roc_auc"),
            "f1": self.metadata.get("f1"),
            "accuracy": self.metadata.get("accuracy"),
        }


@lru_cache(maxsize=1)
def get_model_service() -> ModelService:
    return ModelService()
