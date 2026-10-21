"""Deterministic evaluation harness: runs every synthetic case in cases.json through the
real merge() -> evaluate() pipeline and the real RAG index, with no LLM call. This checks
that the agent's TOOLS are grounded correctly; it does not check how the model phrases
its reply (that needs a live LLM and is exercised manually, see CLAUDE.md's demo script).

Usage:
    python eval\\run_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loan_desk import policy, rag  # noqa: E402  (import after sys.path setup)

CASES_PATH = Path(__file__).resolve().parent / "cases.json"


def run_case(case: dict) -> dict:
    """Evaluate one case and return the row of results to report."""
    merged = policy.merge({}, case["facts"])
    decision = policy.evaluate(merged)
    verdict_ok = decision["verdict"] == case["expected_verdict"]

    row = {"id": case["id"], "category": case["category"], "expected": case["expected_verdict"],
           "actual": decision["verdict"], "verdict_ok": verdict_ok}

    if "expected_out_of_range" in case:
        row["out_of_range_ok"] = decision.get("out_of_range", False) == case["expected_out_of_range"]

    if "policy_query" in case:
        sections = [r["section"] for r in rag.search_policy_docs(case["policy_query"])["results"]]
        row["retrieval_ok"] = case["expected_policy_section"] in sections

    row["pass"] = all(v for k, v in row.items() if k.endswith("_ok"))
    return row


def main() -> int:
    cases = json.loads(CASES_PATH.read_text())
    rows = [run_case(c) for c in cases]

    id_w = max(len(r["id"]) for r in rows) + 2
    print(f"{'case':<{id_w}}{'category':<16}{'expected':<14}{'actual':<14}{'result'}")
    print("-" * (id_w + 16 + 14 + 14 + 8))
    for r in rows:
        mark = "PASS" if r["pass"] else "FAIL"
        print(f"{r['id']:<{id_w}}{r['category']:<16}{r['expected']:<14}{r['actual']:<14}{mark}")
        if not r["pass"]:
            extra = {k: v for k, v in r.items() if k.endswith("_ok") and not v}
            print(f"{'':<{id_w}}  -> failed checks: {extra}")

    passed = sum(r["pass"] for r in rows)
    print(f"\n{passed}/{len(rows)} cases passed ({passed / len(rows):.0%})")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
