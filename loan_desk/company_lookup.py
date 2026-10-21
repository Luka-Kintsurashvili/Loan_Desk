"""Looks up real public-company financials from the SEC's official EDGAR API - not
scraping: data.sec.gov is a structured, free, government-published JSON API, which is
far more reliable for a live demo than parsing arbitrary web pages. Covers only
SEC-registered (publicly traded) companies; a private company returns found=False.

SEC requires a descriptive User-Agent on every request (their documented fair-use
policy) - this is not optional and is set on every call here.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

from . import config

logger = logging.getLogger(__name__)

UA = "Loan Desk student prototype (HIGGS AI Talent Club) contact@example.invalid"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
TIMEOUT = 8
CACHE_DIR = config.HERE / "company_cache"

# priority-ordered fallback tags: filers don't all use the same XBRL concept name
REVENUE_TAGS = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet"]
DEBT_TAGS = ["Liabilities"]
CASH_FLOW_TAGS = ["NetCashProvidedByUsedInOperatingActivities",
                   "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"]

_tickers_cache = None


def _get_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.load(r)


def _load_tickers() -> dict:
    """Company name (lowercased) -> {cik, ticker, title}. Fetched once per process."""
    global _tickers_cache
    if _tickers_cache is not None:
        return _tickers_cache
    data = _get_json(TICKERS_URL)
    _tickers_cache = {row["title"].lower(): row for row in data.values()}
    return _tickers_cache


def _resolve(company: str) -> dict | None:
    name = company.strip().lower()
    tickers = _load_tickers()
    if name in tickers:
        return tickers[name]
    matches = [row for title, row in tickers.items() if name in title or title in name]
    return matches[0] if matches else None


def _latest_annual(concept_units: dict, tags: list[str], instant: bool = False) -> tuple[float, str] | None:
    """Most recent 10-K figure across ALL matching tags. Returns (value, as-of date).

    Filers change which XBRL tag they report under over time (e.g. most switched off
    "Revenues" around 2018's ASC 606 change), so the first tag with *any* data isn't
    necessarily the one with the *latest* data - every candidate tag is pooled and the
    single most recent entry across all of them wins, not the first tag's own latest.

    Revenue and cash flow are "duration" facts (a start/end period, ~1 year for an
    annual figure); balance-sheet items like total liabilities are "instant" facts (a
    single as-of date, no start) - instant=True skips the duration-span check.
    """
    annual = []
    for tag in tags:
        entries = concept_units.get(tag, {}).get("units", {}).get("USD", [])
        annual += [e for e in entries if e.get("form") == "10-K" and e.get("end") and (
            instant or (e.get("fp") == "FY" and e.get("start")
                        and 300 <= (_days(e["end"]) - _days(e["start"])) <= 400))]
    if not annual:
        return None
    best = max(annual, key=lambda e: e["end"])
    return float(best["val"]), best["end"]


def _days(iso_date: str) -> int:
    from datetime import date
    y, m, d = map(int, iso_date.split("-"))
    return date(y, m, d).toordinal()


def _cache_path(cik: str) -> "config.HERE.__class__":
    return CACHE_DIR / f"{cik}.json"


def fetch_public_financials(company: str) -> dict:
    """Look up a public company's most recent annual revenue, total liabilities (debt
    proxy), and operating cash flow from its SEC filings. Returns found=False (never a
    guessed number) if the company isn't SEC-registered or the data isn't extractable.
    """
    try:
        match = _resolve(company)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        logger.warning("SEC ticker lookup failed: %s", e)
        return {"found": False, "reason": f"could not reach SEC EDGAR to look up the company name ({e})"}
    if not match:
        return {"found": False, "reason": f"'{company}' is not a company SEC has on file (it may be private, or the name doesn't match)"}

    cik = str(match["cik_str"]).zfill(10)
    cache_file = _cache_path(cik)
    if cache_file.exists():
        # The cache holds only the small EXTRACTED result (a few fields), not the raw
        # SEC filing - a company's full history of XBRL facts can run several megabytes,
        # almost all of it irrelevant to the three numbers this app actually needs.
        return json.loads(cache_file.read_text(encoding="utf-8"))

    try:
        facts = _get_json(FACTS_URL.format(cik=cik))
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        logger.warning("SEC companyfacts fetch failed for CIK %s: %s", cik, e)
        return {"found": False, "reason": f"found '{match['title']}' but could not reach SEC EDGAR for its filings ({e})"}

    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    revenue = _latest_annual(us_gaap, REVENUE_TAGS)
    debt = _latest_annual(us_gaap, DEBT_TAGS, instant=True)
    cash_flow = _latest_annual(us_gaap, CASH_FLOW_TAGS)
    if not (revenue and debt and cash_flow):
        missing = [n for n, v in (("revenue", revenue), ("total debt", debt), ("operating cash flow", cash_flow)) if not v]
        return {"found": False, "reason": f"found '{match['title']}' on SEC EDGAR but its filings are missing: {', '.join(missing)}"}

    result = {"found": True, "company": match["title"], "cik": cik, "ticker": match.get("ticker"),
              "revenue": revenue[0], "total_debt": debt[0], "operating_cash_flow": cash_flow[0],
              "fiscal_year_end": revenue[1], "source": "SEC EDGAR"}
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file.write_text(json.dumps(result), encoding="utf-8")
    return result
