"""The agent's tool schema (OpenAI function-calling format) and tool dispatch.

Every number a tool returns comes from a deterministic function (policy.evaluate,
loan_math.calculate_payment, the trained model) - no tool ever asks the language model
to compute anything.
"""
from __future__ import annotations

import json

from . import loan_math, policy, rag
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
    return case, {"error": f"unknown tool {name}"}  # unreachable given the TOOL_NAMES check above
