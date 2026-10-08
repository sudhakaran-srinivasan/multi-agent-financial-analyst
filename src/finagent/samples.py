"""Illustrative mock values for each contract, so teammates can build and test
against realistic shapes before the real components exist.

These are MOCKS to show the SHAPE. They are not investment analysis.
"""
from __future__ import annotations

from finagent.contracts import Analysis, Evaluation, NewsInsight, Plan

SAMPLE_NEWS_INSIGHT: NewsInsight = {
    "category": "earnings",
    "summary": "NVIDIA reported quarterly EPS above analyst estimates.",
    "sentiment": "positive",
    "source_url": "https://example.com/nvda-earnings",
}

SAMPLE_PLAN: Plan = {
    "ticker": "NVDA",
    "steps": [
        {"tool": "get_fundamentals", "reason": "profitable mega-cap: check valuation"},
        {"tool": "get_earnings_history", "reason": "earnings due within 60 days"},
        {"tool": "news_insights", "reason": "high news volume"},
    ],
    "priority_note": "Earnings are close, so weigh the beat record heavily.",
}

SAMPLE_ANALYSIS: Analysis = {
    "ticker": "NVDA",
    "stance": "bullish",
    "confidence": "medium",
    "thesis": (
        "Forward P/E is roughly half of trailing P/E, so the market expects "
        "earnings to roughly double; the company beat estimates in 4 of 4 "
        "recent quarters."
    ),
    "evidence_used": [
        {"claim": "Trailing P/E", "value": 30.2,
         "source": "get_fundamentals.valuation.trailingPE"},
        {"claim": "Forward P/E", "value": 15.1,
         "source": "get_fundamentals.valuation.forwardPE"},
        {"claim": "Beats in last 4 quarters", "value": 4,
         "source": "get_earnings_history.summary.beats"},
    ],
    "risks": ["Forward P/E depends on a forecast that may be missed."],
    "sources": ["https://example.com/nvda-earnings"],
    "data_gaps": [],
}

SAMPLE_EVALUATION: Evaluation = {
    "verdict": "pass",
    "overall_score": 4.2,
    "scores": [
        {"criterion": "evidence_support", "score": 4, "comment": "Cites numbers."},
        {"criterion": "numeric_accuracy", "score": 5, "comment": "Matches tools."},
        {"criterion": "stance_consistency", "score": 4, "comment": "Follows."},
        {"criterion": "risk_coverage", "score": 4, "comment": "One risk only."},
        {"criterion": "confidence_calibration", "score": 4, "comment": "Fair."},
    ],
    "feedback": ["Add a second risk, such as valuation compression."],
    "rounds_used": 1,
    "final_analysis": SAMPLE_ANALYSIS,
}