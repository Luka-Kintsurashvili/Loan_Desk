"""Loads the trained company (business) risk model. Simpler than model_service.py:
the model's own feature names (revenue, total_debt, operating_cash_flow,
years_in_business, leverage, dscr, industry) are exactly the business case's field
names, so there's no chat-field-to-model-field remapping needed here.
"""
from __future__ import annotations

import joblib

import credit_model as cm

from . import config

PATH = config.HERE / "model" / "company_risk_model.joblib"

if not PATH.exists():
    raise RuntimeError("Train the company model first: python train_company_model.py")
B = joblib.load(PATH)


def score(rows: list[dict]) -> list[float]:
    """Calibrated default-probability predictions for a list of company feature rows."""
    return cm.score(B, rows)


def reasons(row: dict) -> dict:
    """Occlusion-based explanation of which fields raise/lower the predicted risk."""
    return cm.reasons(B, row)
