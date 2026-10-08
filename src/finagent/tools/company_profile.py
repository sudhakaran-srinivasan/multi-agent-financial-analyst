"""Company profile: validated facts about one ticker, built before planning.

AGENT DESIGN (rubric: Agent Design / Functions)
    Component : get_company_profile  (Track A, step 1 of the research run)
    Consumes  : yfinance `info` dict + `calendar` dict (raw, messy, 189+ fields)
    Produces  : Profile (small, validated, typed) -> read by the planner
    Why       : the planner must plan from verified facts, never from the
                model's guess about what kind of company this is.

AI DISCLOSURE: first draft generated with Claude (Anthropic) and reviewed /
modified by the author. Author-owned decisions are marked  # AUTHOR:.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


# --------------------------------------------------------------------------
# Errors: the "core data failure" tier -> stop and report, never guess.
# --------------------------------------------------------------------------
class ProfileError(Exception):
    """Base class so callers can catch every profile failure at once."""


class InvalidTickerError(ProfileError):
    """Yahoo returned nothing usable (yfinance does NOT raise on bad tickers)."""


class NotAStockError(ProfileError):
    """The ticker exists but is an ETF, index, crypto, etc."""


# --------------------------------------------------------------------------
# The contract (Profile). Required fields first, optional (= None) after.
# --------------------------------------------------------------------------
@dataclass
class Profile:
    """Built by get_company_profile(); consumed by the planner."""

    # Required: without these the ticker is not usable at all.
    ticker: str
    name: str
    quote_type: str
    # Optional: None means "unknown", never "zero".
    sector: str | None = None
    industry: str | None = None
    market_cap: float | None = None  # raw USD (not millions)
    size_bucket: str | None = None  # derived from market_cap
    is_profitable: bool | None = None  # derived from trailing EPS
    is_pre_revenue: bool | None = None  # derived from total revenue
    next_earnings_date: date | None = None
    days_to_earnings: int | None = None  # derived
    # Soft problems (tier 3: degrade gracefully, flag the gap).
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Size buckets. A table + loop is easier to read and change than if/elif.
# --------------------------------------------------------------------------
# AUTHOR: confirm or change these cut-offs (USD). Common analyst convention.
SIZE_BUCKETS: list[tuple[float, str]] = [
    (200e9, "mega"),
    (10e9, "large"),
    (2e9, "mid"),
    (300e6, "small"),
]  # anything below the last floor is "micro"


def size_bucket(market_cap: float | None) -> str | None:
    """Label a market cap. Unknown stays unknown (None), not 'micro'."""
    if market_cap is None:
        return None
    for floor, label in SIZE_BUCKETS:
        if market_cap >= floor:
            return label
    return "micro"


# --------------------------------------------------------------------------
# Validation helper: the first, tiny version of the metric registry.
# --------------------------------------------------------------------------
def _number(
    info: dict,
    key: str,
    warnings: list[str],
    *,
    minimum: float | None = None,
    strict: bool = False,
) -> float | None:
    """Return info[key] as a float if valid, else None (and say why).

    minimum / strict encode the HARD bound: strict=True means value > minimum,
    otherwise value >= minimum. A real 0 is kept; a missing value is None.
    """
    value = info.get(key)
    if value is None:
        warnings.append(f"{key} missing")
        return None
    # bool is a subclass of int in Python, so exclude it explicitly.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        warnings.append(f"{key}={value!r} is not a number; treated as missing")
        return None
    if minimum is not None:
        too_low = value <= minimum if strict else value < minimum
        if too_low:
            warnings.append(f"{key}={value!r} out of range; treated as missing")
            return None
    return float(value)


def _next_earnings(calendar: dict | None, today: date) -> date | None:
    """First earnings date that is today or later; None if none/unknown."""
    dates = (calendar or {}).get("Earnings Date") or []
    upcoming = sorted(d for d in dates if isinstance(d, date) and d >= today)
    return upcoming[0] if upcoming else None


# --------------------------------------------------------------------------
# Pure logic (no network) -> easy to test.
# --------------------------------------------------------------------------
def build_profile(
    ticker: str,
    info: dict,
    calendar: dict | None = None,
    today: date | None = None,
) -> Profile:
    """Validate raw yfinance data and build a Profile, or raise ProfileError."""
    ticker = ticker.strip().upper()
    today = today or date.today()

    quote_type = info.get("quoteType")
    if quote_type is None:  # `is None`, not falsiness: see the 0-vs-missing rule
        raise InvalidTickerError(f"{ticker}: no data returned; check the symbol")
    if quote_type != "EQUITY":
        raise NotAStockError(f"{ticker} is a {quote_type}, not a single stock")

    warnings: list[str] = []
    name = info.get("longName") or info.get("shortName") or ticker

    market_cap = _number(info, "marketCap", warnings, minimum=0, strict=True)
    eps = _number(info, "trailingEps", warnings)  # negative EPS is valid
    revenue = _number(info, "totalRevenue", warnings, minimum=0)  # 0 is valid

    earnings_date = _next_earnings(calendar, today)
    days_to = (earnings_date - today).days if earnings_date else None

    return Profile(
        ticker=ticker,
        name=name,
        quote_type=quote_type,
        sector=info.get("sector"),
        industry=info.get("industry"),
        market_cap=market_cap,
        size_bucket=size_bucket(market_cap),
        is_profitable=None if eps is None else eps > 0,
        # Known revenue of exactly 0 -> True; unknown revenue -> None.
        is_pre_revenue=None if revenue is None else revenue == 0,
        next_earnings_date=earnings_date,
        days_to_earnings=days_to,
        warnings=warnings,
    )


# --------------------------------------------------------------------------
# Thin network wrapper.
# --------------------------------------------------------------------------
def get_company_profile(ticker: str) -> Profile:
    """Fetch from yfinance and build the Profile."""
    import yfinance as yf  # local import keeps build_profile testable offline

    t = yf.Ticker(ticker.strip().upper())
    # TODO(tier 1): retry transient network failures before giving up.
    return build_profile(ticker, t.info, t.calendar)