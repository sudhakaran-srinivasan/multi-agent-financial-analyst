"""Definition of 'this works' for get_company_profile (pure logic only)."""
from datetime import date

import pytest

from company_profile import (
    InvalidTickerError,
    NotAStockError,
    build_profile,
    size_bucket,
)

TODAY = date(2026, 10, 7)

# Real NVDA values from our own yfinance dump.
NVDA_INFO = {
    "quoteType": "EQUITY", "longName": "NVIDIA Corporation",
    "sector": "Technology", "industry": "Semiconductors",
    "marketCap": 5776928145408, "trailingEps": 7.91,
    "totalRevenue": 302970011648,
}
NVDA_CAL = {"Earnings Date": [date(2026, 11, 17)]}


def test_nvda_full_profile():
    p = build_profile("nvda", NVDA_INFO, NVDA_CAL, TODAY)
    assert p.ticker == "NVDA"
    assert p.size_bucket == "mega"
    assert p.is_profitable is True
    assert p.is_pre_revenue is False
    assert p.days_to_earnings == 41
    assert p.warnings == []


def test_invalid_ticker_has_its_own_error():
    with pytest.raises(InvalidTickerError):
        build_profile("ZZZZZZ", {}, None, TODAY)


def test_etf_is_not_a_stock():
    with pytest.raises(NotAStockError):
        build_profile("SPY", {"quoteType": "ETF"}, None, TODAY)


def test_pre_revenue_biotech():
    info = {"quoteType": "EQUITY", "shortName": "Tiny Bio",
            "marketCap": 150e6, "trailingEps": -1.2, "totalRevenue": 0}
    p = build_profile("TBIO", info, None, TODAY)
    assert p.is_pre_revenue is True      # a real 0 is kept
    assert p.is_profitable is False
    assert p.size_bucket == "micro"


def test_missing_is_none_not_zero():
    info = {"quoteType": "EQUITY", "longName": "X"}
    p = build_profile("X", info, None, TODAY)
    assert p.market_cap is None and p.size_bucket is None
    assert p.is_profitable is None and p.is_pre_revenue is None
    assert "marketCap missing" in p.warnings


def test_bad_market_cap_becomes_none_with_warning():
    info = {"quoteType": "EQUITY", "longName": "X", "marketCap": -5}
    p = build_profile("X", info, None, TODAY)
    assert p.market_cap is None
    assert any("out of range" in w for w in p.warnings)


def test_past_earnings_date_is_ignored():
    cal = {"Earnings Date": [date(2026, 8, 1)]}
    p = build_profile("NVDA", NVDA_INFO, cal, TODAY)
    assert p.next_earnings_date is None and p.days_to_earnings is None


@pytest.mark.parametrize("cap, label", [
    (200e9, "mega"), (199e9, "large"), (10e9, "large"), (9e9, "mid"),
    (2e9, "mid"), (1e9, "small"), (300e6, "small"), (299e6, "micro"),
    (None, None),
])
def test_size_bucket_boundaries(cap, label):
    assert size_bucket(cap) == label