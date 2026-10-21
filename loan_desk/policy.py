"""Turns a loan application case into a policy verdict: deterministic checks plus the
model's calibrated default probability. The language model never computes any of this;
it only calls these functions (via tools.py) and reports what they return.
"""
from __future__ import annotations

import re

from . import config, loan_math
from . import model_service as ms

AMT, TERM, INC, DTI, FICO = ms.AMT, ms.TERM, ms.INC, ms.DTI, ms.FICO


def num(value) -> float:
    """Parse a user-typed number, understanding thousands separators and a trailing 'k'."""
    m = re.search(r"(-?\d+\.?\d*)\s*(k)?(?![a-zA-Z])", str(value).replace(",", ""), re.I)
    return float(m[1]) * (1000 if m[2] else 1) if m else 0.0


def ask(case: dict) -> tuple[list[str], list[str]]:
    """Which CORE fields are still missing, and which optional model inputs would sharpen the estimate."""
    core = [label for key, label in ms.CORE if case.get(key) in (None, "")]
    importance = ms.B.get("importance", {})
    rest = sorted((f for f in ms.B["features"] if f != "fico_range_high" and ms.ROLE.get(f, f) not in (AMT, TERM, INC, DTI)
                   and case.get(ms.ROLE.get(f, f)) in (None, "")), key=lambda f: -importance.get(f, 0))
    return core, [ms.LABELS.get(f, f.replace("_", " ")) for f in rest]


def to_model(case: dict, amount: float, income: float) -> dict:
    """Build a model-feature row from a chat-facing case dict."""
    row = {k: v for k, v in case.items() if k in ms.F and k not in ms.ROLE}
    for feature, value in ((ms.M_AMT, amount), (ms.M_INC, income), (ms.M_DTI, case.get(DTI)),
                            (ms.M_FICO, case.get(FICO)), (ms.M_TERM, case.get(TERM))):
        if feature and value not in (None, ""):
            row[feature] = value
    if ms.M_FICO == "fico_range_low" and "fico_range_high" in ms.F and row.get(ms.M_FICO):
        row["fico_range_high"] = row[ms.M_FICO] + 4
    return row


def var(row: dict, amount: float, term: float) -> dict:
    """A copy of a model-feature row with a different amount/term, for scoring alternatives."""
    v = dict(row)
    if ms.M_AMT:
        v[ms.M_AMT] = amount
    if ms.M_TERM:
        v[ms.M_TERM] = term
    return v


def merge(case: dict, updates: dict) -> dict:
    """Apply validated facts onto a case. Each field REPLACES its old value (no accumulation) -
    a corrected income overwrites the old one, it doesn't average with it."""
    case = dict(case)
    updates = dict(updates or {})
    if updates.get("monthly_income") not in (None, ""):
        updates[INC] = num(updates["monthly_income"]) * 12
    updates.pop("monthly_income", None)
    for k, v in updates.items():
        if k in ms.ALLOWED and v not in (None, ""):
            if k in ms.HINTS:
                case[k] = 1 if str(v).strip().lower() in ("yes", "true", "y") or num(v) > 0 else 0
            else:
                case[k] = num(v) if (k in ms.B["num"] or k in ms.EXTRA) else v
    if case.get("monthly_debt_payments") not in (None, "") and num(case.get(INC)) and case.get(DTI) in (None, ""):
        case[DTI] = round(100 * num(case["monthly_debt_payments"]) / (num(case[INC]) / 12), 1)
    if case.get("fico_range_low") and "fico_range_high" in ms.F and case.get("fico_range_high") in (None, ""):
        case["fico_range_high"] = case["fico_range_low"] + 4
    return case


def evaluate(case: dict) -> dict:
    """The single source of truth for a case's policy verdict: scores it, checks every
    policy limit, and (if it doesn't pass as requested) searches for amount/term variants
    that would. Pure function of `case` and policy.json - safe to call repeatedly, makes
    no network call, and is what the UI's decision panel is always rendered from.
    """
    amt, n, inc = num(case.get(AMT)), num(case.get(TERM)), num(case.get(INC))
    core, more = ask(case)
    d = {"currency": config.POLICY["currency"], "ask": core, "ask_more": more[:15], "ask_more_total": len(more),
         "model": ms.B["metrics"], "policy": config.POLICY,
         "missing": [k for k, v in ((AMT, amt), (TERM, n), (INC, inc)) if not v]}
    if d["missing"]:
        d["verdict"] = "need_info"
        return d
    rate = num(case.get("rate_pct"))
    d["rate_assumed"] = not rate
    d["out_of_range"] = amt * config.POLICY["usd_per_unit"] > 40000
    rate = rate or config.POLICY["default_rate_pct"]
    dti, coll = num(case.get(DTI)), num(case.get("collateral_value"))
    fico = num(case.get(FICO)) if FICO else 0

    def row(a, t, p):
        pay, mi = loan_math.pmt(a, rate, t), inc / 12
        checks = [("Default probability", p, config.POLICY["max_pd"], p <= config.POLICY["max_pd"]),
                  ("Payment / income", pay / mi, config.POLICY["max_pti"], pay / mi <= config.POLICY["max_pti"])]
        if case.get(DTI) not in (None, ""):
            x = dti / 100 + pay / mi
            checks.append(("DTI with this loan", x, config.POLICY["max_dti_after"], x <= config.POLICY["max_dti_after"]))
        if coll:
            x = a / coll
            checks.append(("Loan / collateral value", x, config.POLICY["max_ltv"], x <= config.POLICY["max_ltv"]))
        checks.append(("Annual income", inc, config.POLICY["min_income"], inc >= config.POLICY["min_income"]))
        if fico:
            checks.append(("FICO score", fico, config.POLICY["min_fico"], fico >= config.POLICY["min_fico"]))
        return {"amount": a, "term": int(t), "payment": round(pay, 2), "pd": round(p, 4), "ok": all(c[3] for c in checks),
                "checks": [dict(zip(("name", "value", "limit", "ok"), c)) for c in checks]}

    fx = config.POLICY["usd_per_unit"]
    model_row = to_model(case, amt * fx, inc * fx)  # the model was trained on USD
    main = row(amt, n, float(ms.score([model_row])[0]))
    candidates = [(round(amt * f, -2), t) for f in (1, .9, .8, .7, .6, .5) for t in sorted({int(n), 36, 60})
                  if not (f == 1 and t == int(n))]
    preds = ms.score([var(model_row, a * fx, t) for a, t in candidates])
    options = sorted((r for r in (row(a, t, float(p)) for (a, t), p in zip(candidates, preds)) if r["ok"]),
                      key=lambda r: (-r["amount"], r["pd"]))
    d.update(request=main, options=options[:3], reasons=ms.named(ms.reasons(model_row)),
             collateral_needed=round(amt / config.POLICY["max_ltv"]),
             verdict="meets_policy" if main["ok"] else ("conditions" if options else "refer"),
             completeness=round(sum(case.get(ms.ROLE.get(f, f)) not in (None, "") for f in ms.B["features"]) / len(ms.B["features"]), 2))
    return d
