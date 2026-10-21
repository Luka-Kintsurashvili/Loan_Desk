"""Deterministic loan payment math, kept out of the language model entirely."""
from __future__ import annotations


def pmt(principal: float, annual_rate_pct: float, months: float) -> float:
    """Monthly amortized payment for a fixed-rate loan."""
    r = annual_rate_pct / 1200
    return principal / months if r == 0 else principal * r / (1 - (1 + r) ** -months)


def calculate_payment(amount, rate_pct, months) -> dict:
    """Tool entry point: exact monthly payment for an amount, annual rate percent, and term in months."""
    try:
        amount, rate_pct, months = float(amount), float(rate_pct), float(months)
    except (TypeError, ValueError):
        return {"error": "amount, rate_pct and months must all be numbers"}
    if months <= 0:
        return {"error": "months must be positive"}
    return {"amount": amount, "rate_pct": rate_pct, "months": months, "monthly_payment": round(pmt(amount, rate_pct, months), 2)}
