"""Business-lending policy: turns a company case into a verdict, mirroring policy.py's
responsibilities for the personal-loan domain but for a different schema and checks
(debt service coverage, leverage) instead of FICO/DTI. Kept as a separate module rather
than unified with policy.py: the two domains' fields and rules differ enough that
forcing one function to handle both would hurt clarity more than it would save code.
"""
from __future__ import annotations

from . import company_model_service as cms
from . import config, loan_math
from .policy import num

# Must match train_company_model.py's DEBT_SERVICE_RATE exactly: the model was trained
# on dscr computed with this assumption, so inference has to use the same one or the
# model's learned relationship between dscr and risk would be systematically wrong.
DEBT_SERVICE_RATE = 0.20

AMT, TERM = "loan_amount", "term_months"
CORE = [(AMT, "loan amount requested"), (TERM, "term in months"), ("revenue", "annual revenue"),
        ("total_debt", "total existing debt"), ("operating_cash_flow", "annual operating cash flow")]
NUMERIC = {AMT, TERM, "rate_pct", "revenue", "total_debt", "operating_cash_flow", "years_in_business"}
ALLOWED = NUMERIC | {"company", "industry"}
LABELS = {"years_in_business": "years in business", "industry": "industry (retail, manufacturing, "
          "technology, construction, healthcare, hospitality, or professional_services)"}

FIELDS = (
    "BUSINESS ALLOWED KEYS:\n"
    "- company (the company's name, text)\n"
    "- loan_amount (number)\n"
    "- term_months (number)\n"
    "- rate_pct (interest rate the lender offers, percent, optional)\n"
    "- revenue (annual revenue, number)\n"
    "- total_debt (total existing debt/liabilities, number)\n"
    "- operating_cash_flow (annual operating cash flow, number, can be negative)\n"
    "- years_in_business (number, optional)\n"
    "- industry (optional, one of: retail, manufacturing, technology, construction, healthcare, hospitality, professional_services)\n"
)


def ask(case: dict) -> tuple[list[str], list[str]]:
    """Which CORE fields are still missing, and which optional fields would sharpen the estimate."""
    core = [label for key, label in CORE if case.get(key) in (None, "")]
    rest = [LABELS[f] for f in ("years_in_business", "industry") if case.get(f) in (None, "")]
    return core, rest


def merge(case: dict, updates: dict) -> dict:
    """Apply validated facts onto a business case. Same REPLACE (not accumulate)
    semantics as policy.merge(): a corrected revenue overwrites the old one."""
    case = dict(case)
    case["_segment"] = "business"
    for k, v in (updates or {}).items():
        if k in ALLOWED and v not in (None, ""):
            case[k] = num(v) if k in NUMERIC else v
    return case


def to_model(case: dict) -> dict:
    """Build a model-feature row. leverage/dscr are always derived here, never taken
    directly from the case - an officer or fetch tool only ever supplies the raw figures."""
    revenue, debt, ocf = num(case.get("revenue")), num(case.get("total_debt")), num(case.get("operating_cash_flow"))
    row = {"revenue": revenue, "total_debt": debt, "operating_cash_flow": ocf,
           "leverage": debt / revenue if revenue else 0.0,
           "dscr": ocf / max(debt * DEBT_SERVICE_RATE, 1)}
    if case.get("years_in_business") not in (None, ""):
        row["years_in_business"] = num(case["years_in_business"])
    if case.get("industry"):
        row["industry"] = case["industry"]
    return row


def evaluate(case: dict) -> dict:
    """The single source of truth for a company case's policy verdict. The company's
    own default-risk score depends only on its financials, never on the loan amount or
    term being requested (the model has no such input) - what DOES depend on the
    requested loan is whether the company can cover the added debt service, which is
    what the "with this loan" DSCR check reflects.
    """
    policy = config.BUSINESS_POLICY
    amt, term = num(case.get(AMT)), num(case.get(TERM))
    revenue, debt, ocf = num(case.get("revenue")), num(case.get("total_debt")), num(case.get("operating_cash_flow"))
    core, more = ask(case)
    # total_debt and operating_cash_flow can legitimately be 0 (a debt-free company, a
    # cash-flow-neutral one) - "missing" means absent, not falsy, so this checks presence
    # in the case dict directly rather than truthiness of the parsed number.
    d = {"currency": policy["currency"], "ask": core, "ask_more": more, "ask_more_total": len(more),
         "model": cms.B["metrics"], "policy": policy,
         "missing": [k for k, present in ((AMT, amt), (TERM, term), ("revenue", case.get("revenue") not in (None, "")),
                                           ("total_debt", case.get("total_debt") not in (None, "")),
                                           ("operating_cash_flow", case.get("operating_cash_flow") not in (None, "")))
                     if not present]}
    if d["missing"]:
        d["verdict"] = "need_info"
        return d
    rate = num(case.get("rate_pct"))
    d["rate_assumed"] = not rate
    rate = rate or policy["default_rate_pct"]
    d["out_of_range"] = revenue > policy["out_of_range_revenue"]

    model_row = to_model(case)
    pd = float(cms.score([model_row])[0])
    years = num(case.get("years_in_business")) if case.get("years_in_business") not in (None, "") else None
    existing_debt_service = debt * DEBT_SERVICE_RATE

    def row(a, t):
        pay = loan_math.pmt(a, rate, t)
        dscr_after = ocf / max(existing_debt_service + pay * 12, 1)
        checks = [
            ("Default probability", pd, policy["max_pd"], pd <= policy["max_pd"]),
            ("Debt service coverage with this loan", dscr_after, policy["min_dscr"], dscr_after >= policy["min_dscr"]),
            ("Leverage (debt / revenue)", model_row["leverage"], policy["max_leverage"], model_row["leverage"] <= policy["max_leverage"]),
            ("Annual revenue", revenue, policy["min_revenue"], revenue >= policy["min_revenue"]),
        ]
        if years is not None:
            checks.append(("Years in business", years, policy["min_years_in_business"], years >= policy["min_years_in_business"]))
        return {"amount": a, "term": int(t), "payment": round(pay, 2), "pd": round(pd, 4), "ok": all(c[3] for c in checks),
                "checks": [dict(zip(("name", "value", "limit", "ok"), c)) for c in checks]}

    main = row(amt, term)
    candidates = [(round(amt * f, -2), t) for f in (1, .9, .8, .7, .6, .5) for t in sorted({int(term), 36, 60})
                  if not (f == 1 and t == int(term))]
    options = sorted((r for r in (row(a, t) for a, t in candidates) if r["ok"]), key=lambda r: (-r["amount"], r["pd"]))
    d.update(request=main, options=options[:3], reasons=cms.reasons(model_row),
             verdict="meets_policy" if main["ok"] else ("conditions" if options else "refer"))
    return d
