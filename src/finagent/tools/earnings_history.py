"""get_earnings_history: did the company beat analyst EPS estimates lately?

AGENT DESIGN (rubric: Agent Design / Functions)
    Component : get_earnings_history  (Track A numeric tool; the model may call it)
    Consumes  : Finnhub `company_earnings` rows (actual vs. estimate EPS)
    Produces  : EarningsHistory (quarters + summary + gaps + warnings)
    Why       : a company's beat record shows how much to trust the forecasts
                behind forward P/E; it feeds synthesis and reflection.

DESIGN REASONING (author-owned decision: follow the FactSet convention)
    Beat / in-line / miss
        FactSet Earnings Insight (May 2026) classifies a report by comparing
        actual EPS with the MEAN analyst estimate: above = positive surprise
        (beat), equal = in line, below = negative surprise (miss). There is no
        percentage band for "in line". We follow that published convention.
        (We could not verify LSEG/I-B-E-S's convention, so we make no claim
        about it.)
    Cent rounding
        EPS is quoted to the cent. Finnhub's estimate has 4 decimals (2.1384),
        so raw equality would almost never happen. We round both sides to cents
        (ROUND_HALF_UP via Decimal, not float round(), which can misround
        values such as 2.675) before comparing.
    Magnitude is kept separately
        surprise_pct = (actual - estimate) / |estimate| * 100. Percent, not
        dollars, because a $0.08 beat means different things on $2 and $20 EPS.
        |estimate| in the denominator keeps the sign meaningful when
        estimates are negative (loss-making companies).
    A beat is the baseline, not a signal
        FactSet reports ~78% of S&P 500 companies beat (5-year average) and a
        5-year average aggregate surprise of ~7.3%. So the agent should read
        "4 of 4 beats" as consistent with the norm, not as exceptional. Our
        simple average of a few percentages is NOT the same measure as
        FactSet's aggregate, so do not compare them directly.
    Why no trend statistic
        The free tier returns at most 4 quarters; a trend from 4 points is
        weak evidence, so we report beat rate and average surprise only.
    Data caveats found in real NVDA output
        - `period` is calendar-aligned and cannot be a true report date, so we
          label rows with the company's own fiscal `year` and `quarter`.
        - Finnhub's four actuals sum to ~7.01 while yfinance trailingEps is
          7.91 (different definitions/timing). One-source rule: never mix them.

AI DISCLOSURE: 
"""
from __future__ import annotations

import math
import os
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, TypedDict

from finagent.tools.fundamentals import Gap

# AUTHOR: free tier returns at most 4 quarters; fewer than this = thin history.
EXPECTED_QUARTERS = 4
# AUTHOR: how far our recomputed surprise may differ from Finnhub's own value
# (in percentage points) before we flag the row.
SURPRISE_TOLERANCE_PTS = 0.5


class Quarter(TypedDict):
    year: int | None  # company fiscal year (NVDA: FY2027)
    quarter: int | None
    estimate: float | None
    actual: float | None
    surprise_pct: float | None
    result: str | None  # "beat" | "in_line" | "miss" | None (not classifiable)


class EarningsSummary(TypedDict):
    n: int  # quarters we could classify
    beats: int
    in_line: int
    misses: int
    beat_rate: float  # beats / n
    avg_surprise_pct: float | None
    latest_surprise_pct: float | None


class EarningsHistory(TypedDict):
    ticker: str
    quarters: list[Quarter]  # newest first
    summary: EarningsSummary | None
    gaps: list[Gap]
    warnings: list[str]


def _num(value: object) -> tuple[float | None, str | None]:
    """Return (number, gap_reason). A real 0 is kept; missing is None."""
    if value is None:
        return None, "missing"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, "not_a_number"
    if not math.isfinite(value):  # NaN / inf
        return None, "not_a_number"
    return float(value), None


def _cents(x: float) -> int:
    """Round to whole cents, half-up, avoiding binary-float surprises."""
    return int((Decimal(str(x)) * 100).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def _classify(actual: float, estimate: float) -> str:
    a, e = _cents(actual), _cents(estimate)
    if a > e:
        return "beat"
    if a < e:
        return "miss"
    return "in_line"


def build_earnings_history(ticker: str, rows: Any) -> EarningsHistory:
    """Pure logic (no network): validate rows, classify, summarize."""
    ticker = ticker.strip().upper()
    gaps: list[Gap] = []
    warnings: list[str] = []
    quarters: list[Quarter] = []

    if not isinstance(rows, list) or not rows:
        gaps.append({"field": "earnings_history", "reason": "no_history"})
        return EarningsHistory(ticker=ticker, quarters=[], summary=None,
                               gaps=gaps, warnings=warnings)

    # Newest first by the company's own fiscal labels; rows lacking them go last.
    def sort_key(row: dict) -> tuple[int, int]:
        return (row.get("year") or 0, row.get("quarter") or 0)

    for i, row in enumerate(sorted((r for r in rows if isinstance(r, dict)),
                                   key=sort_key, reverse=True)):
        year, qtr = row.get("year"), row.get("quarter")
        label = f"FY{year}Q{qtr}" if year and qtr else f"row{i}"
        estimate, est_reason = _num(row.get("estimate"))
        actual, act_reason = _num(row.get("actual"))
        for name, reason in (("estimate", est_reason), ("actual", act_reason)):
            if reason is not None:
                gaps.append({"field": f"{label}.{name}", "reason": reason})

        result = surprise = None
        if estimate is not None and actual is not None:
            result = _classify(actual, estimate)
            if estimate == 0:  # division undefined
                gaps.append({"field": f"{label}.surprise_pct", "reason": "zero_estimate"})
            else:
                surprise = (actual - estimate) / abs(estimate) * 100
                reported, _ = _num(row.get("surprisePercent"))
                if reported is not None and abs(reported - surprise) > SURPRISE_TOLERANCE_PTS:
                    warnings.append(
                        f"{label}: Finnhub surprisePercent {reported:.2f} differs from "
                        f"recomputed {surprise:.2f}: data may be inconsistent")

        quarters.append(Quarter(year=year, quarter=qtr, estimate=estimate,
                                actual=actual, surprise_pct=surprise, result=result))

    classified = [q for q in quarters if q["result"] is not None]
    if not classified:
        gaps.append({"field": "earnings_history", "reason": "no_history"})
        return EarningsHistory(ticker=ticker, quarters=quarters, summary=None,
                               gaps=gaps, warnings=warnings)

    n = len(classified)
    if n < EXPECTED_QUARTERS:
        warnings.append(f"thin history: only {n} classifiable quarter(s) "
                        f"(expected {EXPECTED_QUARTERS})")
    surprises = [q["surprise_pct"] for q in classified if q["surprise_pct"] is not None]
    beats = sum(q["result"] == "beat" for q in classified)
    summary = EarningsSummary(
        n=n,
        beats=beats,
        in_line=sum(q["result"] == "in_line" for q in classified),
        misses=sum(q["result"] == "miss" for q in classified),
        beat_rate=beats / n,
        avg_surprise_pct=sum(surprises) / len(surprises) if surprises else None,
        latest_surprise_pct=classified[0]["surprise_pct"],
    )
    return EarningsHistory(ticker=ticker, quarters=quarters, summary=summary,
                           gaps=gaps, warnings=warnings)


def get_earnings_history(ticker: str, client: Any = None,
                         limit: int = EXPECTED_QUARTERS) -> EarningsHistory:
    """Fetch from Finnhub and build the history.

    `client` can be injected for tests. Earnings are SECONDARY data: if the
    call fails we degrade gracefully (a gap + warning) instead of crashing the
    run. A missing API key is a setup error, so that one still raises.
    """
    if client is None:
        import finnhub  # local import keeps build_earnings_history offline-testable

        key = os.environ.get("FINNHUB_API_KEY")
        if not key:
            raise RuntimeError("FINNHUB_API_KEY is not set (see .env.example)")
        client = finnhub.Client(api_key=key)
    try:
        rows = client.company_earnings(ticker.strip().upper(), limit=limit)
    except Exception as exc:  # tool boundary: report, don't crash the agent loop
        result = build_earnings_history(ticker, [])
        result["gaps"] = [{"field": "earnings_history", "reason": "fetch_failed"}]
        result["warnings"].append(f"Finnhub call failed: {type(exc).__name__}: {exc}")
        return result
    return build_earnings_history(ticker, rows)