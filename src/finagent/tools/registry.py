"""Metric registry: what each yfinance field means and what values are valid.

AGENT DESIGN (rubric: Agent Design / Functions)
    Component : metric registry (data, not logic), used by get_fundamentals
    Why       : domain knowledge (units, sane ranges, why a metric matters)
                is written ONCE here as data, so it is inspectable, testable
                and identical on every run. No LLM is involved in validation.

Two kinds of bounds (see FieldSpec):
    hard  -> impossible values = bad data -> value becomes None + a gap
    soft  -> unusual but possible -> value kept + a warning

AI DISCLOSURE: 
"""
from __future__ import annotations

from dataclasses import dataclass

GROUPS = (
    "valuation",
    "profitability",
    "growth",
    "balance_sheet",
    "price",
    "sentiment",
)


@dataclass(frozen=True)
class FieldSpec:
    """One row of the registry."""

    key: str  # yfinance `info` key (kept as-is for traceability)
    group: str  # one of GROUPS
    unit: str  # human-readable unit AFTER normalization
    why: str  # why an analyst cares (also feeds the LLM's interpretation)
    forecast: bool = False  # True = analyst opinion/forecast, not a fact
    scale: float = 1.0  # multiply raw value by this to normalize
    hard_min: float | None = None
    hard_max: float | None = None
    min_exclusive: bool = False  # True: value must be > hard_min, not >=
    soft_min: float | None = None
    soft_max: float | None = None


# AUTHOR: review the soft/hard bounds; they encode judgement, not facts.
REGISTRY: tuple[FieldSpec, ...] = (
    # ---- valuation: "how expensive is it?" --------------------------------
    FieldSpec("trailingPE", "valuation", "x (price / last-12-month EPS)",
              "Price paid per $1 of actual yearly profit.",
              hard_min=0, min_exclusive=True, hard_max=5000, soft_max=100),
    FieldSpec("forwardPE", "valuation", "x (price / expected EPS)",
              "Same, on EXPECTED earnings; far below trailing = growth priced in.",
              forecast=True,
              hard_min=0, min_exclusive=True, hard_max=5000, soft_max=100),
    FieldSpec("priceToBook", "valuation", "x",
              "Price vs. accounting net worth; useful for banks/asset-heavy firms.",
              hard_min=0, min_exclusive=True, hard_max=1000, soft_max=50),
    FieldSpec("priceToSalesTrailing12Months", "valuation", "x",
              "Price per $1 of revenue; works when profits are tiny or negative.",
              hard_min=0, min_exclusive=True, hard_max=1000, soft_max=50),
    FieldSpec("enterpriseToEbitda", "valuation", "x",
              "Whole-business value vs. operating profit; fair across debt levels.",
              hard_min=0, min_exclusive=True, hard_max=2000, soft_max=60),
    FieldSpec("pegRatio", "valuation", "x",
              "P/E divided by growth. Derived from a growth FORECAST: treat with care.",
              forecast=True,
              hard_min=0, min_exclusive=True, hard_max=100, soft_max=5),
    # ---- profitability: "does it make money on what it sells?" ------------
    FieldSpec("trailingEps", "profitability", "USD per share",
              "Profit per share over the last 12 months. Negative = losing money.",
              hard_min=-1000, hard_max=100000),
    FieldSpec("grossMargins", "profitability", "fraction (0.75 = 75%)",
              "Share of revenue left after direct costs; pricing power.",
              hard_min=-10, hard_max=1, soft_min=0),
    FieldSpec("operatingMargins", "profitability", "fraction",
              "Share of revenue left after running costs.",
              hard_min=-1000, hard_max=1, soft_min=-1),
    FieldSpec("profitMargins", "profitability", "fraction",
              "Net profit as a share of revenue.",
              hard_min=-1000, hard_max=1, soft_min=-1),
    FieldSpec("returnOnEquity", "profitability", "fraction",
              "Profit generated per $1 of shareholder equity; inflated by buybacks.",
              hard_min=-100, hard_max=100, soft_max=1.5),
    # ---- growth: "is it getting bigger?" ----------------------------------
    FieldSpec("revenueGrowth", "growth", "fraction YoY (1.0 = +100%)",
              "Sales growth versus the same period last year.",
              hard_min=-1, hard_max=1000, soft_max=3),
    FieldSpec("earningsGrowth", "growth", "fraction YoY",
              "Profit growth versus the same period last year.",
              hard_min=-100, hard_max=1000, soft_max=5),
    # ---- balance sheet and cash: "can it survive and fund itself?" --------
    FieldSpec("debtToEquity", "balance_sheet", "ratio (0.17 = debt is 17% of equity)",
              "Leverage: how much of the business is funded by borrowing.",
              scale=0.01,  # Yahoo reports a percent-style number (16.97)
              hard_min=-50, hard_max=50, soft_min=0, soft_max=3),
    FieldSpec("currentRatio", "balance_sheet", "x",
              "Short-term assets vs. short-term bills; below 1 = liquidity strain.",
              hard_min=0, hard_max=1000, soft_min=1, soft_max=20),
    FieldSpec("freeCashflow", "balance_sheet", "USD",
              "Cash left after capital spending: the most 'real' profit measure."),
    FieldSpec("totalCash", "balance_sheet", "USD",
              "Cash and short-term investments on hand.", hard_min=0),
    FieldSpec("totalDebt", "balance_sheet", "USD",
              "All borrowings.", hard_min=0),
    # ---- price and trend --------------------------------------------------
    FieldSpec("currentPrice", "price", "USD", "Latest share price.",
              hard_min=0, min_exclusive=True),
    FieldSpec("fiftyTwoWeekHigh", "price", "USD", "Highest price in 52 weeks.",
              hard_min=0, min_exclusive=True),
    FieldSpec("fiftyTwoWeekLow", "price", "USD", "Lowest price in 52 weeks.",
              hard_min=0, min_exclusive=True),
    FieldSpec("fiftyDayAverage", "price", "USD",
              "Average close, last 50 days: short-term trend.",
              hard_min=0, min_exclusive=True),
    FieldSpec("twoHundredDayAverage", "price", "USD",
              "Average close, last 200 days: long-term trend.",
              hard_min=0, min_exclusive=True),
    # ---- sentiment: opinions, not facts -----------------------------------
    FieldSpec("targetMeanPrice", "sentiment", "USD",
              "Average analyst price target (an opinion).", forecast=True,
              hard_min=0, min_exclusive=True),
    FieldSpec("numberOfAnalystOpinions", "sentiment", "count",
              "How many analysts cover it; few = thin, less reliable consensus.",
              hard_min=0),
    FieldSpec("dividendYield", "sentiment", "fraction (0.0042 = 0.42%)",
              "Yearly dividend as a share of price.",
              scale=0.01,  # yfinance 1.7.0 reports percent (0.42); see canary test
              hard_min=0, hard_max=0.25, soft_max=0.10),
)