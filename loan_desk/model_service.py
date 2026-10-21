"""Loads the trained credit-risk model and maps its features to chat-facing field names.

credit_model.py (the trainer) is never imported for its training code here, only for its
PATH constant and its score()/reasons() inference helpers.
"""
from __future__ import annotations

import joblib

import credit_model as cm

if not cm.PATH.exists():
    raise RuntimeError("Train the model first: python credit_model.py data\\LC_loans_granting_model_dataset.csv")
B = joblib.load(cm.PATH)

F = set(B["features"])
_pick = lambda *names: next((n for n in names if n in F), None)
M_AMT, M_INC = _pick("loan_amnt", "amount_requested", "loan_amount"), _pick("annual_inc", "revenue", "income", "annual_income")
M_DTI, M_FICO, M_TERM = _pick("dti", "dti_n"), _pick("fico_range_low", "fico_n", "fico", "risk_score"), _pick("term")

# Chat-facing field names (what the agent's tools and the UI use), independent of the
# model's own internal column names above.
AMT, TERM, INC, DTI, FICO = "loan_amount", "term_months", "annual_income", "dti_pct", "credit_score"
ROLE = {f: k for f, k in ((M_AMT, AMT), (M_INC, INC), (M_DTI, DTI), (M_FICO, FICO), (M_TERM, TERM)) if f}
EXTRA = {"rate_pct": "interest rate the lender offers, percent", "collateral_value": "market value of collateral",
         "monthly_income": "income per month, if the user gives it monthly",
         "monthly_debt_payments": "existing recurring loan and credit card payments per month; the server turns it into DTI",
         AMT: "loan amount", TERM: "loan term in months, e.g. 24", INC: "income per year",
         DTI: "existing monthly debt payments as % of income, 18 means 18%", FICO: "credit score (FICO), only if known"}
ALLOWED = (F - set(ROLE)) | set(EXTRA)
HINTS = {"experience_c": "1 if the applicant has any credit history, 0 if none (not years)"}
LABELS = {"loan_amnt": "loan amount", "emp_length": "years at current job", "home_ownership": "renting, owning or mortgage",
          "home_ownership_n": "renting, owning or mortgage", "purpose": "purpose of the loan", "fico_range_low": "credit score (FICO)",
          "fico_n": "credit score (FICO)", "revenue": "annual income", "dti_n": "debt-to-income",
          "experience_c": "has any credit history (yes or no)", "verification_status": "is the income verified",
          "delinq_2yrs": "late payments (30+ days) in the last 2 years", "inq_last_6mths": "credit inquiries in the last 6 months",
          "open_acc": "number of open credit accounts", "pub_rec": "public records (bankruptcies, judgments)",
          "revol_bal": "credit card balance", "revol_util": "credit card utilization %", "total_acc": "total credit accounts ever opened",
          "mort_acc": "number of mortgage accounts", "application_type": "individual or joint application",
          "pub_rec_bankruptcies": "bankruptcies"}
CORE = [(AMT, "loan amount"), (TERM, "term in months"), (INC, "income (monthly or yearly)"), ("rate_pct", "interest rate offered"),
        (DTI, "existing monthly debt payments as % of income"), ("collateral_value", "collateral value, if the loan is secured")]
REASON = {"home_ownership_n": "housing status", "experience_c": "credit history", "revenue": "annual income"}


def describe(field: str) -> str:
    """One-line human description of a model input field, used to build the agent's system prompt."""
    if field in HINTS:
        return f"{field} ({HINTS[field]})"
    if field in B["num"]:
        return f"{field} (number, typical {B['baseline'][field]:.4g})"
    return f"{field} (one of: {', '.join(map(str, B['cat_values'][field][:12]))})"


FIELDS = ("ALLOWED KEYS:\n" + "\n".join("- " + describe(f) for f in B["features"] if f in ALLOWED) + "\n"
          + "\n".join(f"- {k} ({v})" for k, v in EXTRA.items()) + "\n")


def log_feature_mapping() -> None:
    """Print which chat fields were matched to which model features; startup diagnostics only."""
    print("Model inputs matched to chat fields:",
          {k: (f"{f} (typical {B['baseline'][f]:.4g})" if f in B["num"] else f) for f, k in ROLE.items()})
    print("Other model inputs:", [f for f in B["features"] if f not in ROLE])


def score(rows: list[dict]) -> list[float]:
    """Calibrated default-probability predictions for a list of model-feature rows."""
    return cm.score(B, rows)


def reasons(row: dict) -> dict:
    """Occlusion-based explanation of which fields raise/lower the predicted risk for one row."""
    return cm.reasons(B, row)


def named(rs: dict) -> dict:
    """Relabel occlusion-reason feature names from model-internal names to human labels."""
    return {k: [{**r, "feature": REASON.get(r["feature"]) or LABELS.get(r["feature"], r["feature"].replace("_", " "))} for r in v]
            for k, v in rs.items()}
