"""Signals: plain-Python reads of the collected facts, before any model writes.

Agent Design (rubric: Agent Functions, tool use and reasoning support).
    Why signals?  A bare list of 40 numbers invites the model to cherry-pick or
    invent. A signal is one small, checkable judgment ("price is below its
    200-day average -> negative") computed by Python from tool results. The
    model then has to ACCOUNT for every signal, including the ones that
    disagree, and the evaluator can re-check each one.

    Each signal has a direction (positive / neutral / negative), the value it
    was based on, the exact field it came from (``source``) and a one-line
    ``why`` that says what it means for an investor.

    Thresholds are conventions, not laws. They are named constants marked
    ``# AUTHOR:`` so the project owner can defend or change each one.

AI use: see the AI note in README.md.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

Direction = Literal["positive", "neutral", "negative"]


class Signal(TypedDict):
    name: str
    direction: Direction
    value: float | str | None
    source: str  # tool.group.field, so the number can be verified
    why: str  # one line: what it means for an investor


# AUTHOR: review every threshold below; these are common conventions.
GOOD_MARGIN = 0.10  # profit margin at or above 10% is healthy
GOOD_GROWTH = 0.10  # revenue growth at or above 10% a year
GOOD_FCF_YIELD = 0.03  # free cash flow yield at or above 3%
SAFE_CURRENT_RATIO = 1.5  # short-term assets cover bills 1.5x or better
STRESS_CURRENT_RATIO = 1.0  # below 1 = liquidity strain
FORWARD_PE_GAP = 0.10  # forward vs trailing P/E must differ by 10% to count
TARGET_UPSIDE = 0.15  # analyst target 15% above price counts as upside
MIN_ANALYSTS = 5  # fewer opinions than this: the target is too thin to use
MIN_QUARTERS = 3  # fewer quarters than this: the beat record is too thin
# Roughly 78% of S&P 500 companies beat estimates (FactSet 5-year average),
# so a high beat rate is the NORM, not a strength.
BEAT_BASELINE_NOTE = "beating estimates is the norm (about 78% of S&P 500 companies)"


def _num(value: Any) -> float | None:
    """A real number or None (bool is not a number here)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _group(results: dict[str, Any], group: str) -> dict[str, Any]:
    fundamentals = results.get("get_fundamentals")
    if not isinstance(fundamentals, dict):
        return {}
    section = fundamentals.get(group)
    return section if isinstance(section, dict) else {}


def _signal(name: str, direction: Direction, value: float | str | None,
            source: str, why: str) -> Signal:
    return {"name": name, "direction": direction, "value": value,
            "source": source, "why": why}


def _threshold_signal(name: str, value: float | None, source: str, *,
                      good: float | None = None, bad: float | None = None,
                      good_why: str, neutral_why: str, bad_why: str
                      ) -> Signal | None:
    """Positive at/above ``good``, negative below ``bad``, otherwise neutral."""
    if value is None:
        return None
    if good is not None and value >= good:
        return _signal(name, "positive", value, source, good_why)
    if bad is not None and value < bad:
        return _signal(name, "negative", value, source, bad_why)
    return _signal(name, "neutral", value, source, neutral_why)


def compute_signals(results: dict[str, Any]) -> list[Signal]:
    """Turn tool results into signals. Missing inputs simply give no signal.

    A missing input is already reported as a gap by its tool, so skipping the
    signal avoids inventing a judgment from nothing.
    """
    valuation = _group(results, "valuation")
    profitability = _group(results, "profitability")
    growth = _group(results, "growth")
    balance = _group(results, "balance_sheet")
    price_group = _group(results, "price")
    sentiment = _group(results, "sentiment")
    fundamentals = results.get("get_fundamentals")
    derived = fundamentals.get("derived", {}) if isinstance(fundamentals, dict) else {}

    signals: list[Signal | None] = []

    # -- valuation: what is the market expecting? --
    trailing = _num(valuation.get("trailingPE"))
    forward = _num(valuation.get("forwardPE"))
    src = "get_fundamentals.valuation.forwardPE"
    if trailing and forward and trailing > 0 and forward > 0:
        if forward < trailing * (1 - FORWARD_PE_GAP):
            signals.append(_signal(
                "forward_vs_trailing_pe", "positive", forward, src,
                f"forward P/E {forward:.1f} is below trailing {trailing:.1f}: "
                "analysts expect earnings to grow"))
        elif forward > trailing * (1 + FORWARD_PE_GAP):
            signals.append(_signal(
                "forward_vs_trailing_pe", "negative", forward, src,
                f"forward P/E {forward:.1f} is above trailing {trailing:.1f}: "
                "analysts expect earnings to fall"))
        else:
            signals.append(_signal(
                "forward_vs_trailing_pe", "neutral", forward, src,
                "forward and trailing P/E are close: no big change expected"))
    elif forward and forward > 0 and trailing is None:
        signals.append(_signal(
            "forward_vs_trailing_pe", "neutral", forward, src,
            f"trailing earnings are negative; analysts expect a profit at "
            f"{forward:.1f}x forward earnings (a forecast, not a result)"))

    # -- profitability and growth: is the business earning and getting bigger? --
    signals.append(_threshold_signal(
        "profit_margin", _num(profitability.get("profitMargins")),
        "get_fundamentals.profitability.profitMargins",
        good=GOOD_MARGIN, bad=0.0,
        good_why="net profit margin is healthy (10% or more)",
        neutral_why="profitable, but with a thin margin (under 10%)",
        bad_why="the company loses money on each dollar of sales"))
    signals.append(_threshold_signal(
        "revenue_growth", _num(growth.get("revenueGrowth")),
        "get_fundamentals.growth.revenueGrowth",
        good=GOOD_GROWTH, bad=0.0,
        good_why="sales are growing 10% a year or faster",
        neutral_why="sales growth is slow (under 10%)",
        bad_why="sales are shrinking"))

    # -- financial strength: can it survive and fund itself? --
    signals.append(_threshold_signal(
        "net_cash", _num(derived.get("net_cash")),
        "get_fundamentals.derived.net_cash", good=0.0, bad=0.0,
        good_why="cash exceeds debt (a net cash position)",
        neutral_why="cash and debt are about equal",
        bad_why="debt exceeds cash (net debt)"))
    signals.append(_threshold_signal(
        "fcf_yield", _num(derived.get("fcf_yield")),
        "get_fundamentals.derived.fcf_yield", good=GOOD_FCF_YIELD, bad=0.0,
        good_why="free cash flow is 3% or more of market value",
        neutral_why="free cash flow is positive but under 3% of market value",
        bad_why="the business burns cash after capital spending"))
    signals.append(_threshold_signal(
        "liquidity", _num(balance.get("currentRatio")),
        "get_fundamentals.balance_sheet.currentRatio",
        good=SAFE_CURRENT_RATIO, bad=STRESS_CURRENT_RATIO,
        good_why="short-term assets comfortably cover short-term bills",
        neutral_why="short-term assets cover bills, with a modest cushion",
        bad_why="short-term assets do not cover short-term bills"))

    # -- trend: what has the market been doing with the price? --
    price = _num(price_group.get("currentPrice"))
    avg200 = _num(price_group.get("twoHundredDayAverage"))
    if price is not None and avg200 is not None:
        above = price >= avg200
        signals.append(_signal(
            "trend_vs_200d", "positive" if above else "negative", price,
            "get_fundamentals.price.currentPrice",
            f"price {price:.2f} is {'above' if above else 'below'} its "
            f"200-day average {avg200:.2f} (the long-term trend)"))

    # -- sentiment: opinions, labelled as opinions --
    target = _num(sentiment.get("targetMeanPrice"))
    analysts = _num(sentiment.get("numberOfAnalystOpinions"))
    if (price and target is not None and analysts is not None
            and analysts >= MIN_ANALYSTS):
        upside = target / price - 1
        direction: Direction = ("positive" if upside >= TARGET_UPSIDE
                                else "negative" if upside <= 0 else "neutral")
        signals.append(_signal(
            "analyst_target", direction, target,
            "get_fundamentals.sentiment.targetMeanPrice",
            f"analyst average target {target:.2f} is {upside:+.0%} from the "
            f"price ({int(analysts)} analysts; an opinion, not a fact)"))

    # -- earnings record: a beat is the baseline, a thin record proves little --
    history = results.get("get_earnings_history")
    summary = history.get("summary") if isinstance(history, dict) else None
    if isinstance(summary, dict) and summary.get("n"):
        n, beats, misses = summary["n"], summary["beats"], summary["misses"]
        source = "get_earnings_history.summary.beat_rate"
        rate = _num(summary.get("beat_rate"))
        if n < MIN_QUARTERS:
            signals.append(_signal(
                "earnings_record", "neutral", rate, source,
                f"only {n} quarter(s) of history ({beats} beat, {misses} "
                "miss): too thin to draw a conclusion"))
        elif rate is not None and rate <= 0.5:
            signals.append(_signal(
                "earnings_record", "negative", rate, source,
                f"beat estimates in only {beats} of {n} quarters, which is "
                "below the market norm"))
        else:
            signals.append(_signal(
                "earnings_record", "neutral", rate, source,
                f"beat estimates in {beats} of {n} quarters; "
                f"{BEAT_BASELINE_NOTE}, so this is not a strength"))

    return [s for s in signals if s is not None]
