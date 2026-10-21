from loan_desk import company_model_service as cms


def test_bundle_loads_with_expected_shape():
    for key in ("model", "iso", "num", "cat", "features", "baseline", "cat_values", "importance", "metrics"):
        assert key in cms.B
    assert cms.B["cat"] == ["industry"]
    assert set(cms.B["num"]) == {"revenue", "total_debt", "operating_cash_flow", "years_in_business", "leverage", "dscr"}


def test_distinguishes_a_healthy_company_from_a_distressed_one():
    healthy = {"revenue": 20_000_000, "total_debt": 8_000_000, "operating_cash_flow": 3_000_000,
               "years_in_business": 15, "leverage": 0.4, "dscr": 1.875, "industry": "healthcare"}
    distressed = {"revenue": 2_000_000, "total_debt": 6_000_000, "operating_cash_flow": -100_000,
                  "years_in_business": 1, "leverage": 3.0, "dscr": -0.08, "industry": "hospitality"}
    pd_healthy, pd_distressed = cms.score([healthy, distressed])
    assert pd_healthy < 0.05
    assert pd_distressed > 0.5


def test_reasons_names_leverage_as_a_top_driver_for_an_overleveraged_company():
    distressed = {"revenue": 2_000_000, "total_debt": 6_000_000, "operating_cash_flow": -100_000,
                  "years_in_business": 1, "leverage": 3.0, "dscr": -0.08, "industry": "hospitality"}
    r = cms.reasons(distressed)
    assert any(x["feature"] == "leverage" and x["effect_pts"] > 0 for x in r["up"])
