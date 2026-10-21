from unittest.mock import patch

from loan_desk import tools

BASE_FACTS = {"loan_amount": 20000, "term_months": 36, "annual_income": 60000, "credit_score": 680, "dti_pct": 15}
STRONG_COMPANY = {"loan_amount": 2_000_000, "term_months": 36, "revenue": 1_180_552_000,
                   "total_debt": 215_933_000, "operating_cash_flow": 181_254_000,
                   "years_in_business": 30, "industry": "manufacturing"}


def test_update_case_accepts_allowed_keys_and_reports_unknown_ones():
    case, result = tools.dispatch("update_case", {"facts": dict(BASE_FACTS, bogus_key="x")}, {})
    assert case["loan_amount"] == 20000
    assert result["updated"] == BASE_FACTS
    assert result["rejected_unknown_keys"] == ["bogus_key"]


def test_update_case_rejects_non_object_facts():
    case, result = tools.dispatch("update_case", {"facts": "not a dict"}, {})
    assert "error" in result
    assert case == {}


def test_score_applicant_requires_the_core_fields():
    case, result = tools.dispatch("score_applicant", {}, {})
    assert "error" in result
    assert "missing" in result


def test_score_applicant_returns_a_probability_once_case_is_complete():
    case, _ = tools.dispatch("update_case", {"facts": BASE_FACTS}, {})
    _, result = tools.dispatch("score_applicant", {}, case)
    assert result["default_probability"] == 0.3805
    assert result["monthly_payment"] == 664.29


def test_check_policy_reports_the_refer_verdict():
    case, _ = tools.dispatch("update_case", {"facts": BASE_FACTS}, {})
    _, result = tools.dispatch("check_policy", {}, case)
    assert result["verdict"] == "refer"
    assert any(c["name"] == "Default probability" and not c["ok"] for c in result["checks"])


def test_find_conditions_reports_when_nothing_passes():
    case, _ = tools.dispatch("update_case", {"facts": BASE_FACTS}, {})
    _, result = tools.dispatch("find_conditions", {}, case)
    assert result["options"] == []
    assert "note" in result


def test_calculate_payment_matches_known_values():
    _, r1 = tools.dispatch("calculate_payment", {"amount": 15000, "rate_pct": 9.5, "months": 36}, {})
    assert r1["monthly_payment"] == 480.49
    _, r2 = tools.dispatch("calculate_payment", {"amount": 10000, "rate_pct": 24, "months": 24}, {})
    assert r2["monthly_payment"] == 528.71


def test_search_policy_docs_returns_a_cited_section():
    _, result = tools.dispatch("search_policy_docs", {"query": "collateral loan to value"}, {})
    assert result["results"][0]["section"] == "Collateral and secured loans"


def test_unknown_tool_name_is_rejected():
    case, result = tools.dispatch("delete_everything", {}, {"loan_amount": 1})
    assert result == {"error": "unknown tool delete_everything"}
    assert case == {"loan_amount": 1}  # case must be untouched


# --- business-lending tools -------------------------------------------------

def test_fetch_company_financials_auto_merges_found_data_into_the_case():
    fake = {"found": True, "company": "Acme Corp", "cik": "0000000001", "revenue": 1_000_000,
            "total_debt": 200_000, "operating_cash_flow": 150_000, "source": "SEC EDGAR"}
    with patch("loan_desk.tools.company_lookup.fetch_public_financials", return_value=fake):
        case, result = tools.dispatch("fetch_company_financials", {"company": "Acme Corp"}, {})
    assert result == fake
    assert case["revenue"] == 1_000_000 and case["total_debt"] == 200_000
    assert case["_segment"] == "business"


def test_fetch_company_financials_does_not_touch_case_when_not_found():
    fake = {"found": False, "reason": "not on file"}
    with patch("loan_desk.tools.company_lookup.fetch_public_financials", return_value=fake):
        case, result = tools.dispatch("fetch_company_financials", {"company": "Nope LLC"}, {})
    assert result == fake
    assert case == {}


def test_update_business_case_accepts_allowed_keys_and_reports_unknown_ones():
    case, result = tools.dispatch("update_business_case", {"facts": dict(STRONG_COMPANY, bogus_key="x")}, {})
    assert case["revenue"] == 1_180_552_000
    assert result["rejected_unknown_keys"] == ["bogus_key"]


def test_score_company_risk_requires_the_core_fields():
    case, result = tools.dispatch("score_company_risk", {}, {})
    assert "error" in result and "missing" in result


def test_score_company_risk_returns_a_probability_for_a_strong_company():
    case, _ = tools.dispatch("update_business_case", {"facts": STRONG_COMPANY}, {})
    _, result = tools.dispatch("score_company_risk", {}, case)
    assert result["default_probability"] < 0.05
    assert result["out_of_range"] is False


def test_check_business_policy_reports_meets_policy_for_a_strong_company():
    case, _ = tools.dispatch("update_business_case", {"facts": STRONG_COMPANY}, {})
    _, result = tools.dispatch("check_business_policy", {}, case)
    assert result["verdict"] == "meets_policy"


def test_find_business_conditions_returns_passing_alternatives_for_a_strong_company():
    case, _ = tools.dispatch("update_business_case", {"facts": STRONG_COMPANY}, {})
    _, result = tools.dispatch("find_business_conditions", {}, case)
    assert all(o["ok"] for o in result["options"])  # every returned option genuinely passes policy


def test_find_business_conditions_notes_when_nothing_passes():
    distressed = {"loan_amount": 5_000_000, "term_months": 60, "revenue": 1_500_000,
                  "total_debt": 6_000_000, "operating_cash_flow": -100_000}
    case, _ = tools.dispatch("update_business_case", {"facts": distressed}, {})
    _, result = tools.dispatch("find_business_conditions", {}, case)
    assert result["options"] == []
    assert "note" in result
