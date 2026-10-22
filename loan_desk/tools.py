"""The agent's tool schema (OpenAI function-calling format) and tool dispatch.

Every number a tool returns comes from a deterministic function (policy.evaluate,
loan_math.calculate_payment, the trained model) - no tool ever asks the language model
to compute anything.
"""
from __future__ import annotations

import json

from . import business_policy as bp
from . import company_lookup, loan_math, policy, rag
from . import model_service as ms

TOOLS = [
    {"type": "function", "function": {
        "name": "update_case",
        "description": "Add or correct applicant facts stated by the user or read from an attached document/image.",
        "parameters": {"type": "object", "properties": {
            "facts": {"type": "object", "description": "key: value pairs using only the allowed keys listed in the system prompt",
                       "additionalProperties": True}},
            "required": ["facts"]}}},
    {"type": "function", "function": {
        "name": "score_applicant",
        "description": "Get the default-risk model's probability of default for the current case, and the top reasons it moves the risk up or down.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "check_policy",
        "description": "Check the current case against the lender's policy limits and get the verdict (need_info, meets_policy, conditions, refer).",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "find_conditions",
        "description": "Find alternative loan amount/term combinations for the current case that would fit the lender's policy.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "search_policy_docs",
        "description": "Search the lender's written policy for the section that explains a rule (income verification, collateral, exceptions, etc).",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "calculate_payment",
        "description": "Compute the exact monthly payment for a loan amount, annual interest rate percent, and term in months.",
        "parameters": {"type": "object", "properties": {
            "amount": {"type": "number"}, "rate_pct": {"type": "number"}, "months": {"type": "number"}},
            "required": ["amount", "rate_pct", "months"]}}},
    {"type": "function", "function": {
        "name": "fetch_company_financials",
        "description": "Look up a public (SEC-registered) company's real revenue, total debt, and operating cash flow from its "
                        "official SEC filings, and record them on the case. Only works for publicly traded companies - returns "
                        "found: false for a private company (ask the officer for figures instead via update_business_case).",
        "parameters": {"type": "object", "properties": {"company": {"type": "string"}}, "required": ["company"]}}},
    {"type": "function", "function": {
        "name": "update_business_case",
        "description": "Add or correct company facts stated by the user (for a private company, or to override a fetched figure).",
        "parameters": {"type": "object", "properties": {
            "facts": {"type": "object", "description": "key: value pairs using only the allowed business keys listed in the system prompt",
                       "additionalProperties": True}},
            "required": ["facts"]}}},
    {"type": "function", "function": {
        "name": "score_company_risk",
        "description": "Get the business-risk model's probability of default for the current company case, and the top reasons.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "check_business_policy",
        "description": "Check the current company case against the lender's business-lending policy (debt service coverage, "
                        "leverage, revenue, company age) and get the verdict.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "find_business_conditions",
        "description": "Find alternative loan amount/term combinations for the current company case that would fit the lender's "
                        "business policy.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "suggest_new_case",
        "description": "Call this (instead of mixing facts in) when the user is describing a different, new applicant, "
                        "customer, or company while the current case already holds someone else's facts - it lets the "
                        "interface offer a one-click way to start a fresh case for them.",
        "parameters": {"type": "object", "properties": {}}}},
]
TOOL_NAMES = {t["function"]["name"] for t in TOOLS}


def summarize(result, limit: int = 200) -> str:
    """Short JSON string for the UI's agent-steps trace; not the full tool result."""
    s = json.dumps(result, default=str)
    return s if len(s) <= limit else s[:limit] + "..."


def dispatch(name: str, args: dict, case: dict) -> tuple[dict, dict]:
    """Execute one tool call against the current case. Returns (possibly updated case, result dict)."""
    if name not in TOOL_NAMES:
        return case, {"error": f"unknown tool {name}"}
    if name == "update_case":
        facts = args.get("facts")
        if not isinstance(facts, dict):
            return case, {"error": "facts must be an object of key: value"}
        accepted = {k: v for k, v in facts.items() if k in ms.ALLOWED and v not in (None, "")}
        rejected = [k for k in facts if k not in ms.ALLOWED]
        result = {"updated": accepted}
        if rejected:
            result["rejected_unknown_keys"] = rejected
        return policy.merge(case, accepted), result
    if name == "score_applicant":
        d = policy.evaluate(case)
        if d["verdict"] == "need_info":
            return case, {"error": "missing required inputs", "missing": d["missing"]}
        return case, {"default_probability": d["request"]["pd"], "monthly_payment": d["request"]["payment"],
                       "rate_assumed": d.get("rate_assumed", False), "out_of_range": d.get("out_of_range", False),
                       "risk_reasons_up": d["reasons"]["up"], "risk_reasons_down": d["reasons"]["down"]}
    if name == "check_policy":
        d = policy.evaluate(case)
        if d["verdict"] == "need_info":
            return case, {"verdict": "need_info", "missing_required": d["ask"], "optional_missing": d["ask_more"][:10]}
        return case, {"verdict": d["verdict"], "checks": d["request"]["checks"], "completeness": d["completeness"]}
    if name == "find_conditions":
        d = policy.evaluate(case)
        if d["verdict"] == "need_info":
            return case, {"error": "missing required inputs", "missing": d["missing"]}
        if not d["options"]:
            note = "the requested amount and term already meet policy" if d["verdict"] == "meets_policy" else "no passing amount/term combination was found"
            return case, {"options": [], "note": note}
        return case, {"options": d["options"], "collateral_needed": d["collateral_needed"]}
    if name == "search_policy_docs":
        return case, rag.search_policy_docs(args.get("query", ""))
    if name == "calculate_payment":
        return case, loan_math.calculate_payment(args.get("amount"), args.get("rate_pct"), args.get("months"))
    if name == "fetch_company_financials":
        result = company_lookup.fetch_public_financials(str(args.get("company", "")))
        if not result.get("found"):
            return case, result
        facts = {k: result[k] for k in ("revenue", "total_debt", "operating_cash_flow")}
        facts["company"] = result["company"]
        return bp.merge(case, facts), result
    if name == "update_business_case":
        facts = args.get("facts")
        if not isinstance(facts, dict):
            return case, {"error": "facts must be an object of key: value"}
        accepted = {k: v for k, v in facts.items() if k in bp.ALLOWED and v not in (None, "")}
        rejected = [k for k in facts if k not in bp.ALLOWED]
        result = {"updated": accepted}
        if rejected:
            result["rejected_unknown_keys"] = rejected
        return bp.merge(case, accepted), result
    if name == "score_company_risk":
        d = bp.evaluate(case)
        if d["verdict"] == "need_info":
            return case, {"error": "missing required inputs", "missing": d["missing"]}
        return case, {"default_probability": d["request"]["pd"], "out_of_range": d.get("out_of_range", False),
                       "rate_assumed": d.get("rate_assumed", False),
                       "risk_reasons_up": d["reasons"]["up"], "risk_reasons_down": d["reasons"]["down"]}
    if name == "check_business_policy":
        d = bp.evaluate(case)
        if d["verdict"] == "need_info":
            return case, {"verdict": "need_info", "missing_required": d["ask"], "optional_missing": d["ask_more"]}
        return case, {"verdict": d["verdict"], "checks": d["request"]["checks"], "out_of_range": d.get("out_of_range", False)}
    if name == "find_business_conditions":
        d = bp.evaluate(case)
        if d["verdict"] == "need_info":
            return case, {"error": "missing required inputs", "missing": d["missing"]}
        if not d["options"]:
            note = "the requested amount and term already meet policy" if d["verdict"] == "meets_policy" else "no passing amount/term combination was found"
            return case, {"options": [], "note": note}
        return case, {"options": d["options"]}
    if name == "suggest_new_case":
        # Deterministic guard: the model sometimes pattern-matches on phrasing like
        # "a different applicant" even when the case is already empty (e.g. right after
        # the button it suggests last time resent that same phrasing as a fresh
        # message) - that's nothing to protect, so refuse the suggestion in code rather
        # than relying on the prompt alone to notice the case is empty.
        if not case:
            return case, {"error": "the current case is already empty - there is nothing to protect here. Record these "
                                    "facts normally with update_case or update_business_case instead of suggesting a new case."}
        return case, {"suggested": True}  # case deliberately untouched - the client, not this tool, starts the new one
    return case, {"error": f"unknown tool {name}"}  # unreachable given the TOOL_NAMES check above
