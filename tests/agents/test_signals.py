"""Offline tests for finagent.agent.signals (pure Python, no model).

Definition of "this works":
    1. Each signal gets the right direction on real NVDA and AADX numbers.
    2. A missing input gives no signal (never an invented one).
    3. A beat record is NOT called a strength (beats are the market norm).
    4. A thin record (fewer than 3 quarters) is neutral, even if it is a miss.
    5. Every signal names the field it came from.
"""

import pytest

from finagent.agent.signals import compute_signals

NVDA = {
    "get_fundamentals": {
        "valuation": {"trailingPE": 30.24526, "forwardPE": 15.14251},
        "profitability": {"profitMargins": 0.63663},
        "growth": {"revenueGrowth": 1.059},
        "balance_sheet": {"currentRatio": 4.589},
        "price": {"currentPrice": 239.24, "twoHundredDayAverage": 201.0205},
        "sentiment": {"targetMeanPrice": 328.71695, "numberOfAnalystOpinions": 59.0},
        "derived": {"net_cash": 23608999936.0, "fcf_yield": 0.02198},
    },
    "get_earnings_history": {
        "summary": {"n": 4, "beats": 4, "in_line": 0, "misses": 0, "beat_rate": 1.0},
    },
}

AADX = {
    "get_fundamentals": {
        "valuation": {"trailingPE": None, "forwardPE": 28.80503},
        "profitability": {"profitMargins": -0.30236998},
        "growth": {"revenueGrowth": 0.474},
        "balance_sheet": {"currentRatio": 2.509},
        "price": {"currentPrice": 11.45, "twoHundredDayAverage": 16.786556},
        "sentiment": {"targetMeanPrice": 25.125, "numberOfAnalystOpinions": 8.0},
        "derived": {"net_cash": -413432000.0, "fcf_yield": -0.06556797},
    },
    "get_earnings_history": {
        "summary": {"n": 1, "beats": 0, "in_line": 0, "misses": 1, "beat_rate": 0.0},
    },
}


def by_name(results):
    return {s["name"]: s for s in compute_signals(results)}


def test_nvda_signals():
    s = by_name(NVDA)
    assert s["forward_vs_trailing_pe"]["direction"] == "positive"
    assert s["profit_margin"]["direction"] == "positive"
    assert s["revenue_growth"]["direction"] == "positive"
    assert s["net_cash"]["direction"] == "positive"
    assert s["liquidity"]["direction"] == "positive"
    assert s["trend_vs_200d"]["direction"] == "positive"
    assert s["analyst_target"]["direction"] == "positive"
    # 2.2% FCF yield is positive cash flow but under the 3% bar.
    assert s["fcf_yield"]["direction"] == "neutral"


def test_four_of_four_beats_is_neutral_not_a_strength():
    signal = by_name(NVDA)["earnings_record"]
    assert signal["direction"] == "neutral"
    assert "norm" in signal["why"]


def test_aadx_signals_show_the_conflict():
    s = by_name(AADX)
    assert s["profit_margin"]["direction"] == "negative"
    assert s["net_cash"]["direction"] == "negative"
    assert s["fcf_yield"]["direction"] == "negative"
    assert s["trend_vs_200d"]["direction"] == "negative"
    assert s["revenue_growth"]["direction"] == "positive"
    assert s["analyst_target"]["direction"] == "positive"  # target is 2.2x price


def test_loss_maker_forward_pe_is_neutral_and_labelled_a_forecast():
    signal = by_name(AADX)["forward_vs_trailing_pe"]
    assert signal["direction"] == "neutral"
    assert "forecast" in signal["why"]


def test_one_missed_quarter_is_too_thin_to_judge():
    signal = by_name(AADX)["earnings_record"]
    assert signal["direction"] == "neutral"
    assert "too thin" in signal["why"]


def test_a_poor_record_over_enough_quarters_is_negative():
    results = {"get_earnings_history": {"summary": {
        "n": 4, "beats": 1, "in_line": 0, "misses": 3, "beat_rate": 0.25}}}
    assert by_name(results)["earnings_record"]["direction"] == "negative"


def test_thin_analyst_coverage_gives_no_target_signal():
    results = {"get_fundamentals": {
        "price": {"currentPrice": 10.0},
        "sentiment": {"targetMeanPrice": 20.0, "numberOfAnalystOpinions": 2.0}}}
    assert "analyst_target" not in by_name(results)


def test_missing_inputs_give_no_signal_not_an_invented_one():
    assert compute_signals({}) == []
    assert compute_signals({"get_fundamentals": {"derived": {"net_cash": None}}}) == []
    assert compute_signals({"get_news_insights": []}) == []


def test_liquidity_below_one_is_negative():
    results = {"get_fundamentals": {"balance_sheet": {"currentRatio": 0.8}}}
    assert by_name(results)["liquidity"]["direction"] == "negative"


@pytest.mark.parametrize("results", [NVDA, AADX])
def test_every_signal_names_its_source_and_a_reason(results):
    for signal in compute_signals(results):
        tool, *path = signal["source"].split(".")
        assert tool in results and path
        assert signal["why"]
        assert signal["direction"] in ("positive", "neutral", "negative")