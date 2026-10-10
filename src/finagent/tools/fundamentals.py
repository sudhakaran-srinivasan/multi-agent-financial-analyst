"""get_fundamentals: validated, grouped financial metrics for one ticker.

AGENT DESIGN (rubric: Agent Design / Functions)
    Component : get_fundamentals  (Track A numeric tool; the model may call it)
    Consumes  : yfinance `info` dict (raw) + the metric registry
    Produces  : Fundamentals (six groups + derived metrics + gaps + warnings)
    Why       : Python validates and computes; the LLM only interprets.
                `gaps` tells the agent WHY a number is absent so it can
                reason (re-fetch, lower confidence, or state a limitation).

AI DISCLOSURE:  # AUTHOR:.
"""
from __future__ import annotations

import math
from typing import TypedDict

from finagent.tools.registry import GROUPS, REGISTRY, FieldSpec


class Gap(TypedDict):
    """Why a value is None. The agent reads `reason` to decide what to do."""

    field: str
    reason: str  # missing | not_a_number | out_of_range |
    #              negative_earnings | inputs_missing


class Fundamentals(TypedDict):
    ticker: str
    valuation: dict[str, float | None]
    profitability: dict[str, float | None]
    growth: dict[str, float | None]
    balance_sheet: dict[str, float | None]
    price: dict[str, float | None]
    sentiment: dict[str, float | None]
    derived: dict[str, float | None]
    gaps: list[Gap]
    warnings: list[str]


# Ratios that are meaningless when the company loses money.
EARNINGS_DEPENDENT = {"trailingPE", "forwardPE", "pegRatio"}
# AUTHOR: how far Yahoo's P/E may differ from price / EPS before we flag it.
PE_MISMATCH_TOLERANCE = 0.05


def _clean(spec: FieldSpec, raw: object) -> tuple[float | None, str | None, str | None]:
    """Validate and normalize one value.

    Returns (value, gap_reason, warning). A real 0 is kept; missing is None.
    """
    if raw is None:
        return None, "missing", None
    # bool is a subclass of int, so exclude it before the numeric check.
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None, "not_a_number", None
    if not math.isfinite(raw):  # NaN / inf
        return None, "not_a_number", None

    value = raw * spec.scale

    if spec.hard_min is not None:
        too_low = value <= spec.hard_min if spec.min_exclusive else value < spec.hard_min
        if too_low:
            return None, "out_of_range", None
    if spec.hard_max is not None and value > spec.hard_max:
        return None, "out_of_range", None

    warning = None
    if spec.soft_min is not None and value < spec.soft_min:
        warning = f"{spec.key}={value:g} is unusually low (below {spec.soft_min:g})"
    if spec.soft_max is not None and value > spec.soft_max:
        warning = f"{spec.key}={value:g} is unusually high (above {spec.soft_max:g})"
    return value, None, warning


def _derive(g: dict[str, dict[str, float | None]], market_cap: float | None,
            gaps: list[Gap], warnings: list[str]) -> dict[str, float | None]:
    """Cross-check metrics computed from cleaned inputs (never raw ones)."""
    cash = g["balance_sheet"].get("totalCash")
    debt = g["balance_sheet"].get("totalDebt")
    fcf = g["balance_sheet"].get("freeCashflow")
    price = g["price"].get("currentPrice")
    eps = g["profitability"].get("trailingEps")
    pe = g["valuation"].get("trailingPE")

    net_cash = cash - debt if cash is not None and debt is not None else None
    fcf_yield = fcf / market_cap if fcf is not None and market_cap else None
    pe_recomputed = price / eps if price is not None and eps and eps > 0 else None

    derived = {"net_cash": net_cash, "fcf_yield": fcf_yield,
               "pe_recomputed": pe_recomputed}
    for name, value in derived.items():
        if value is None and not (name == "pe_recomputed" and eps is not None and eps <= 0):
            gaps.append({"field": name, "reason": "inputs_missing"})

    if pe is not None and pe_recomputed is not None:
        gap_pct = abs(pe - pe_recomputed) / pe
        if gap_pct > PE_MISMATCH_TOLERANCE:
            warnings.append(
                f"trailingPE {pe:.1f} differs from price/EPS {pe_recomputed:.1f} "
                f"by {gap_pct:.0%}: data may be stale or inconsistent")
    return derived


def extract_ttm_fcf(frame: object) -> float | None:
    """Pull trailing-twelve-month free cash flow from a yfinance cash-flow table.

    ``Ticker.ttm_cashflow`` is a table: rows are line items, columns are
    period-end dates. We read the "Free Cash Flow" row and take the newest
    date. Anything unexpected (no table, no row, only NaN) returns None, so
    the caller can fall back instead of crashing.
    """
    try:
        row = frame.loc["Free Cash Flow"]  # type: ignore[attr-defined]
        values = row.dropna().sort_index()
        value = float(values.iloc[-1])
    except (AttributeError, KeyError, IndexError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def build_fundamentals(ticker: str, info: dict,
                       ttm_fcf: float | None = None) -> Fundamentals:
    """Pure logic (no network): clean every registry field and assemble groups.

    ``ttm_fcf`` is trailing-twelve-month free cash flow from the cash-flow
    statement. It replaces Yahoo's ``info["freeCashflow"]``, which for NVDA
    ($41.8B) matched no reported period while the statement said $127.0B.
    Without it we fall back to the summary value and warn that it is unverified.
    """
    groups: dict[str, dict[str, float | None]] = {name: {} for name in GROUPS}
    gaps: list[Gap] = []
    warnings: list[str] = []

    if ttm_fcf is not None:
        info = {**info, "freeCashflow": ttm_fcf}  # one source: the statement
    elif info.get("freeCashflow") is not None:
        warnings.append(
            "freeCashflow comes from Yahoo's summary, not the cash-flow "
            "statement; it can disagree with reported figures (unverified)")

    eps_raw = info.get("trailingEps")
    loses_money = isinstance(eps_raw, (int, float)) and not isinstance(eps_raw, bool) \
        and eps_raw <= 0

    for spec in REGISTRY:
        value, reason, warning = _clean(spec, info.get(spec.key))
        groups[spec.group][spec.key] = value
        if reason is not None:
            if spec.key in EARNINGS_DEPENDENT and loses_money:
                reason = "negative_earnings"  # not a data problem: not meaningful
            gaps.append({"field": spec.key, "reason": reason})
        if warning is not None:
            warnings.append(warning)

    market_cap = info.get("marketCap")
    market_cap = float(market_cap) if isinstance(market_cap, (int, float)) \
        and not isinstance(market_cap, bool) and market_cap > 0 else None
    derived = _derive(groups, market_cap, gaps, warnings)

    return Fundamentals(
        ticker=ticker.strip().upper(),
        valuation=groups["valuation"],
        profitability=groups["profitability"],
        growth=groups["growth"],
        balance_sheet=groups["balance_sheet"],
        price=groups["price"],
        sentiment=groups["sentiment"],
        derived=derived,
        gaps=gaps,
        warnings=warnings,
    )


def get_fundamentals(ticker: str, info: dict | None = None,
                     ttm_fcf: float | None = None) -> Fundamentals:
    """Fetch (unless `info` is supplied) and build Fundamentals.

    Pass an already-fetched `info` (and `ttm_fcf`) to avoid extra Yahoo calls.
    """
    if info is None:
        import yfinance as yf  # local import keeps build_fundamentals offline-testable

        stock = yf.Ticker(ticker.strip().upper())
        info = stock.info
        try:
            ttm_fcf = extract_ttm_fcf(stock.ttm_cashflow)
        except Exception:  # noqa: BLE001  - optional upgrade; never fail the tool
            ttm_fcf = None
    return build_fundamentals(ticker, info, ttm_fcf)