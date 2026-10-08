"""Definition of 'this works' for get_fundamentals (pure logic, offline)."""
import math

import pytest

from finagent.tools.fundamentals import build_fundamentals
from finagent.tools.registry import GROUPS, REGISTRY

# Real NVDA values from our own yfinance dump.
NVDA = {
    "marketCap": 5776928145408, "currentPrice": 239.24,
    "trailingPE": 30.24526, "forwardPE": 15.14251, "priceToBook": 25.228304,
    "priceToSalesTrailing12Months": 19.067657, "enterpriseToEbitda": 28.489,
    "pegRatio": 0.29, "trailingEps": 7.91, "grossMargins": 0.74674004,
    "operatingMargins": 0.66236997, "profitMargins": 0.63663,
    "returnOnEquity": 1.17211, "revenueGrowth": 1.059, "earningsGrowth": 1.278,
    "debtToEquity": 16.971, "currentRatio": 4.589,
    "freeCashflow": 41809874944, "totalCash": 62469001216,
    "totalDebt": 38860001280, "fiftyTwoWeekHigh": 243.37,
    "fiftyTwoWeekLow": 164.27, "fiftyDayAverage": 218.9194,
    "twoHundredDayAverage": 201.0205, "targetMeanPrice": 328.71695,
    "numberOfAnalystOpinions": 59, "dividendYield": 0.42,
}


def gap_for(result, field):
    return next((g for g in result["gaps"] if g["field"] == field), None)


def test_nvda_all_groups_filled_and_clean():
    r = build_fundamentals("nvda", NVDA)
    assert r["ticker"] == "NVDA"
    assert all(r[g] for g in GROUPS)
    assert r["valuation"]["trailingPE"] == pytest.approx(30.24526)
    assert r["gaps"] == []
    assert r["warnings"] == []


def test_debt_to_equity_is_normalized_to_ratio():
    r = build_fundamentals("NVDA", NVDA)
    assert r["balance_sheet"]["debtToEquity"] == pytest.approx(0.16971)


def test_dividend_yield_is_normalized_to_fraction():
    r = build_fundamentals("NVDA", NVDA)
    assert r["sentiment"]["dividendYield"] == pytest.approx(0.0042)


def test_derived_metrics_on_nvda():
    d = build_fundamentals("NVDA", NVDA)["derived"]
    assert d["net_cash"] == pytest.approx(62469001216 - 38860001280)
    assert d["fcf_yield"] == pytest.approx(41809874944 / 5776928145408)
    assert d["pe_recomputed"] == pytest.approx(239.24 / 7.91)


def test_loss_making_company_explains_missing_pe():
    biotech = {"marketCap": 150e6, "currentPrice": 4.0, "trailingEps": -1.2}
    r = build_fundamentals("TBIO", biotech)
    assert r["valuation"]["trailingPE"] is None
    assert gap_for(r, "trailingPE")["reason"] == "negative_earnings"
    assert gap_for(r, "forwardPE")["reason"] == "negative_earnings"
    assert r["derived"]["pe_recomputed"] is None
    assert gap_for(r, "pe_recomputed") is None  # not a data gap: expected


def test_hard_bound_violation_becomes_none_with_gap():
    r = build_fundamentals("X", {**NVDA, "grossMargins": 1.5})
    assert r["profitability"]["grossMargins"] is None
    assert gap_for(r, "grossMargins")["reason"] == "out_of_range"


def test_soft_bound_keeps_value_and_warns():
    r = build_fundamentals("X", {**NVDA, "revenueGrowth": 5.0})
    assert r["growth"]["revenueGrowth"] == 5.0
    assert any("revenueGrowth" in w for w in r["warnings"])
    assert gap_for(r, "revenueGrowth") is None


@pytest.mark.parametrize("bad", ["N/A", float("nan"), float("inf"), True])
def test_non_numbers_are_gaps(bad):
    r = build_fundamentals("X", {**NVDA, "trailingPE": bad})
    assert r["valuation"]["trailingPE"] is None
    assert gap_for(r, "trailingPE")["reason"] == "not_a_number"


def test_zero_is_kept_not_treated_as_missing():
    r = build_fundamentals("X", {**NVDA, "totalDebt": 0})
    assert r["balance_sheet"]["totalDebt"] == 0.0
    assert gap_for(r, "totalDebt") is None
    assert r["derived"]["net_cash"] == pytest.approx(62469001216)


def test_pe_cross_check_flags_inconsistent_data():
    r = build_fundamentals("X", {**NVDA, "trailingPE": 50.0})
    assert any("differs from price/EPS" in w for w in r["warnings"])


def test_empty_info_does_not_crash():
    r = build_fundamentals("ZZZ", {})
    assert len(r["gaps"]) >= len(REGISTRY)
    assert all(v is None for v in r["derived"].values())


def test_registry_integrity():
    keys = [s.key for s in REGISTRY]
    assert len(keys) == len(set(keys))            # no duplicate fields
    assert all(s.group in GROUPS for s in REGISTRY)
    for s in REGISTRY:                              # bounds are consistent
        if s.hard_min is not None and s.hard_max is not None:
            assert s.hard_min < s.hard_max
        if s.soft_max is not None and s.hard_max is not None:
            assert s.soft_max <= s.hard_max
        if s.soft_min is not None and s.hard_min is not None:
            assert s.soft_min >= s.hard_min