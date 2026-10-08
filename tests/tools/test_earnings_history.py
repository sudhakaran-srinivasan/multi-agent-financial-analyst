"""Definition of 'this works' for get_earnings_history (offline)."""
import pytest

from finagent.tools.earnings_history import (
    build_earnings_history,
    get_earnings_history,
)

# Real NVDA rows from Finnhub company_earnings (free tier returns 4).
NVDA = [
    {"symbol": "NVDA", "estimate": 2.1384, "actual": 2.22, "period": "2026-09-30",
     "surprise": 0.0816, "surprisePercent": 3.8159, "year": 2027, "quarter": 2},
    {"symbol": "NVDA", "estimate": 1.7922, "actual": 1.87, "period": "2026-06-30",
     "surprise": 0.0778, "surprisePercent": 4.341, "year": 2027, "quarter": 1},
    {"symbol": "NVDA", "estimate": 1.5634, "actual": 1.62, "period": "2026-03-31",
     "surprise": 0.0566, "surprisePercent": 3.6203, "year": 2026, "quarter": 4},
    {"symbol": "NVDA", "estimate": 1.2746, "actual": 1.3, "period": "2025-12-31",
     "surprise": 0.0254, "surprisePercent": 1.9928, "year": 2026, "quarter": 3},
]


def row(estimate, actual, year=2027, quarter=1, **extra):
    return {"estimate": estimate, "actual": actual, "year": year,
            "quarter": quarter, **extra}


def gap_reasons(result):
    return {g["field"]: g["reason"] for g in result["gaps"]}


def test_nvda_real_rows():
    r = build_earnings_history("nvda", NVDA)
    s = r["summary"]
    assert (s["n"], s["beats"], s["in_line"], s["misses"]) == (4, 4, 0, 0)
    assert s["beat_rate"] == 1.0
    assert s["avg_surprise_pct"] == pytest.approx(3.4425, abs=0.01)
    assert s["latest_surprise_pct"] == pytest.approx(3.816, abs=0.01)
    assert r["gaps"] == [] and r["warnings"] == []


def test_rows_are_sorted_newest_first():
    r = build_earnings_history("NVDA", list(reversed(NVDA)))
    assert [(q["year"], q["quarter"]) for q in r["quarters"]] == [
        (2027, 2), (2027, 1), (2026, 4), (2026, 3)]


def test_miss_and_beat_classification():
    r = build_earnings_history("X", [row(1.60, 1.50, quarter=2), row(1.00, 1.10)])
    assert [q["result"] for q in r["quarters"]] == ["miss", "beat"]


def test_cent_rounding_makes_tiny_gap_in_line():
    # raw actual > estimate, but both round to $2.14: FactSet-style in line
    r = build_earnings_history("X", [row(2.1384, 2.14)])
    assert r["quarters"][0]["result"] == "in_line"


def test_half_up_rounding_beats_float_round():
    # float round(2.675, 2) == 2.67 would call this a beat; half-up gives 2.68
    r = build_earnings_history("X", [row(2.675, 2.68)])
    assert r["quarters"][0]["result"] == "in_line"


def test_negative_estimate_surprise_sign():
    q = build_earnings_history("X", [row(-1.0, -0.5)])["quarters"][0]
    assert q["result"] == "beat"
    assert q["surprise_pct"] == pytest.approx(50.0)


def test_zero_estimate_has_no_percent_but_is_classified():
    r = build_earnings_history("X", [row(0.0, 0.05)])
    assert r["quarters"][0]["result"] == "beat"
    assert r["quarters"][0]["surprise_pct"] is None
    assert gap_reasons(r)["FY2027Q1.surprise_pct"] == "zero_estimate"


@pytest.mark.parametrize("rows", [[], None, "oops"])
def test_empty_or_bad_input_is_no_history(rows):
    r = build_earnings_history("IPO", rows)
    assert r["summary"] is None and r["quarters"] == []
    assert gap_reasons(r) == {"earnings_history": "no_history"}


def test_thin_history_warns_but_summarizes():
    r = build_earnings_history("NEW", NVDA[:2])
    assert r["summary"]["n"] == 2
    assert any("thin history" in w for w in r["warnings"])


def test_missing_actual_is_a_gap_and_excluded_from_summary():
    r = build_earnings_history("X", [row(1.0, None, quarter=2), row(1.0, 1.2)])
    assert gap_reasons(r)["FY2027Q2.actual"] == "missing"
    assert r["summary"]["n"] == 1


def test_non_number_is_gap():
    r = build_earnings_history("X", [row("N/A", 1.0)])
    assert gap_reasons(r)["FY2027Q1.estimate"] == "not_a_number"
    assert r["summary"] is None


def test_finnhub_percent_disagreement_warns():
    r = build_earnings_history("X", [row(2.0, 2.2, surprisePercent=25.0)])
    assert any("differs from recomputed" in w for w in r["warnings"])


def test_fetch_failure_degrades_gracefully():
    class Boom:
        def company_earnings(self, *a, **k):
            raise ConnectionError("network down")

    r = get_earnings_history("NVDA", client=Boom())
    assert r["summary"] is None
    assert gap_reasons(r) == {"earnings_history": "fetch_failed"}
    assert any("ConnectionError" in w for w in r["warnings"])


def test_injected_client_is_used():
    class Fake:
        def company_earnings(self, symbol, limit):
            assert (symbol, limit) == ("NVDA", 4)
            return NVDA

    assert get_earnings_history("nvda", client=Fake())["summary"]["n"] == 4


def test_missing_api_key_is_a_setup_error(monkeypatch):
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        get_earnings_history("NVDA")