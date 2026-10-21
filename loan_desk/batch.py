"""Deterministic batch scoring: a CSV of applicants -> a verdict for each.

No LLM involved - this reuses the same merge()/evaluate() pipeline the chat agent's
tools call, just run directly over each CSV row. Useful for scoring many applicants at
once, and for proving the policy engine works independently of the chat agent.
"""
from __future__ import annotations

import csv
import io

from . import policy

MAX_ROWS = 200  # keep a single upload bounded


def score_csv(text: str) -> list[dict]:
    """Parse a CSV of applicants and return a verdict summary for each row.

    Expected columns match the chat agent's field names (loan_amount, annual_income,
    term_months, credit_score, dti_pct, rate_pct, collateral_value, monthly_income,
    monthly_debt_payments, experience_c, emp_length, purpose, home_ownership_n), plus an
    optional "name" column. Unknown columns are ignored, same as the chat agent does.

    Each result includes the row's merged `case` dict (not just the summary numbers) so
    the UI can load a specific applicant into the chat agent for a follow-up discussion.
    """
    reader = csv.DictReader(io.StringIO(text))
    results = []
    for i, row in enumerate(reader, start=1):
        if i > MAX_ROWS:
            break
        clean = {(k or "").strip(): (v or "").strip() for k, v in row.items() if k}
        name = clean.get("name") or clean.get("applicant") or f"Applicant {i}"
        facts = {k: v for k, v in clean.items() if k not in ("name", "applicant") and v}
        case = policy.merge({}, facts)
        d = policy.evaluate(case)
        result = {"row": i, "name": name, "verdict": d["verdict"], "case": case}
        if d["verdict"] == "need_info":
            result["missing"] = d["missing"]
        else:
            result["default_probability"] = d["request"]["pd"]
            result["monthly_payment"] = d["request"]["payment"]
            up = d["reasons"]["up"]
            result["top_risk_reason"] = up[0]["feature"] if up else None
        results.append(result)
    return results
