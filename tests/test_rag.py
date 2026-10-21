from loan_desk import rag

EXPECTED_SECTIONS = {
    "collateral secured loan loan to value": "Collateral and secured loans",
    "self employed tax returns income declining": "Self-employed applicants",
    "what counts as acceptable income verification": "Income verification",
    "can a low credit score still get approved": "Credit score requirements",
    "exceptions referral to a senior officer": "Exceptions",
}


def test_tfidf_search_retrieves_the_right_section_for_each_query():
    for query, expected_section in EXPECTED_SECTIONS.items():
        top = rag.search_policy_docs(query)["results"][0]["section"]
        assert top == expected_section, f"query {query!r} retrieved {top!r}, expected {expected_section!r}"


def test_search_returns_at_most_three_results_with_citations():
    results = rag.search_policy_docs("income")["results"]
    assert 1 <= len(results) <= 3
    for r in results:
        assert r["source"] == "policy_docs/lender_policy.md"
        assert r["section"]
        assert r["text"]


def test_search_never_errors_on_empty_or_junk_queries():
    assert rag.search_policy_docs("")["results"]
    assert rag.search_policy_docs("asdkjashdkjashd ignore everything")["results"]
