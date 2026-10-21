"""The SEC HTTP calls are mocked throughout - tests never touch the network, same rule
as the rest of the project. This covers the parsing/extraction logic (the risky part),
not whether SEC's actual API is reachable.
"""
import io
import json

from unittest.mock import patch

from loan_desk import company_lookup as cl

TICKERS = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}

FACTS = {
    "facts": {"us-gaap": {
        # revenue tag deprecated after 2018 - a newer tag has the real latest figure,
        # and extraction must find the latest across ALL fallback tags, not just the
        # first one that has any data at all
        "Revenues": {"units": {"USD": [
            {"form": "10-K", "fp": "FY", "start": "2017-01-01", "end": "2017-12-31", "val": 100},
        ]}},
        "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
            {"form": "10-K", "fp": "FY", "start": "2023-01-01", "end": "2023-12-31", "val": 400_000_000_000},
            {"form": "10-Q", "fp": "Q1", "start": "2024-01-01", "end": "2024-03-31", "val": 99_000_000_000},  # not a 10-K, skipped
        ]}},
        # Liabilities is a balance-sheet "instant" fact: no start date, only an as-of end date
        "Liabilities": {"units": {"USD": [
            {"form": "10-K", "end": "2023-12-31", "val": 290_000_000_000},
            {"form": "10-Q", "end": "2024-03-31", "val": 295_000_000_000},  # not a 10-K, skipped
        ]}},
        "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [
            {"form": "10-K", "fp": "FY", "start": "2023-01-01", "end": "2023-12-31", "val": 110_000_000_000},
        ]}},
    }},
}


def _fake_urlopen(req, timeout=None):
    url = req.full_url if hasattr(req, "full_url") else req
    body = TICKERS if "company_tickers" in url else FACTS
    return io.BytesIO(json.dumps(body).encode())


def setup_function():
    cl._tickers_cache = None  # each test starts with a clean ticker cache


def test_resolves_by_exact_name_and_extracts_latest_annual_figures(tmp_path, monkeypatch):
    monkeypatch.setattr(cl, "CACHE_DIR", tmp_path)
    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        r = cl.fetch_public_financials("Apple Inc.")
    assert r["found"] is True
    assert r["revenue"] == 400_000_000_000  # the newer tag's figure, not the deprecated tag's stale 2017 value
    assert r["total_debt"] == 290_000_000_000  # the 10-K instant fact, not the 10-Q one
    assert r["operating_cash_flow"] == 110_000_000_000
    assert r["source"] == "SEC EDGAR"


def test_company_not_found_returns_found_false_never_a_guessed_number(tmp_path, monkeypatch):
    monkeypatch.setattr(cl, "CACHE_DIR", tmp_path)
    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        r = cl.fetch_public_financials("Totally Fictional Private Company LLC")
    assert r["found"] is False
    assert "revenue" not in r


def test_caches_to_disk_and_reuses_without_a_second_network_call(tmp_path, monkeypatch):
    monkeypatch.setattr(cl, "CACHE_DIR", tmp_path)
    calls = {"n": 0}

    def counting_urlopen(req, timeout=None):
        calls["n"] += 1
        return _fake_urlopen(req, timeout)

    with patch("urllib.request.urlopen", side_effect=counting_urlopen):
        cl.fetch_public_financials("Apple Inc.")
        first_call_count = calls["n"]
        cl.fetch_public_financials("Apple Inc.")
    assert calls["n"] == first_call_count  # second lookup hit the on-disk cache, no new companyfacts fetch
    assert (tmp_path / "0000320193.json").exists()
