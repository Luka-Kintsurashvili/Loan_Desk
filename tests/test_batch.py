from loan_desk import batch

CSV = """name,loan_amount,term_months,annual_income,credit_score,dti_pct
Strong,8000,36,95000,780,5
Risky,25000,36,40000,620,30
Missing,12000,36,,650,8
"""


def test_scores_each_row_independently():
    results = batch.score_csv(CSV)
    assert [r["name"] for r in results] == ["Strong", "Risky", "Missing"]
    assert results[0]["verdict"] == "meets_policy"
    assert results[1]["verdict"] == "refer"
    assert results[2]["verdict"] == "need_info"
    assert "missing" in results[2] and results[2]["missing"]
    assert results[0]["default_probability"] < results[1]["default_probability"]


def test_result_includes_the_merged_case_for_followup_discussion():
    results = batch.score_csv(CSV)
    assert results[0]["case"]["loan_amount"] == 8000.0
    assert results[0]["case"]["annual_income"] == 95000.0


def test_unknown_columns_are_ignored_not_errors():
    results = batch.score_csv("name,loan_amount,term_months,annual_income,credit_score,extra_junk_column\nX,8000,36,60000,700,whatever\n")
    assert results[0]["verdict"] in ("meets_policy", "conditions", "refer")


def test_empty_csv_returns_empty_list():
    assert batch.score_csv("name,loan_amount\n") == []


def test_row_cap_is_enforced():
    header = "name,loan_amount,term_months,annual_income,credit_score\n"
    rows = "\n".join(f"A{i},8000,36,60000,700" for i in range(300))
    results = batch.score_csv(header + rows)
    assert len(results) == batch.MAX_ROWS
