"""Shared data shapes ("contracts") between components and between teammates.

AGENT DESIGN (rubric: Agent Design / Functions)
    Why        : each contract is the agreement between two pieces of code.
                 Everyone codes against the same file, so a mismatch shows up
                 immediately instead of at integration time.
    Who agrees : the three that cross a track boundary need team agreement
                 before changing: NewsInsight (A <-> B), Analysis (A <-> C),
                 Evaluation (C <-> notebook). Plan is internal to Track A.
    Profile    : lives in finagent.tools.company_profile (Track A only).

    Flow:  ticker -> Profile -> Plan -> tool loop -> Analysis -> Evaluation
                                   ^ NewsInsight list comes from Jeff's pipeline

TypedDicts are plain dicts at runtime (json.dumps works); Literal types list
the only allowed values. Python does not enforce them at runtime, so
tests/test_contracts.py checks the samples against these definitions.

AI DISCLOSURE: 
"""
from __future__ import annotations

from typing import Literal, TypedDict

# ---- allowed values -------------------------------------------------------
NewsCategory = Literal["earnings", "general", "market"]
# Three labels, not a score: LLMs label consistently but invent false
# precision with numbers; labels aggregate easily and map to +1/0/-1 later.
Sentiment = Literal["positive", "neutral", "negative"]
Stance = Literal["bullish", "neutral", "bearish"]
Confidence = Literal["low", "medium", "high"]
Verdict = Literal["pass", "revise"]


# ---- NewsInsight: Jeff's pipeline -> agent core (crosses A <-> B) ----------
class NewsInsight(TypedDict):
    """One analyzed news article. Jeff's pipeline returns list[NewsInsight]."""

    category: NewsCategory  # set by Classify, used by the router
    summary: str  # written by the routed specialist
    sentiment: Sentiment
    source_url: str  # so the final analysis can cite it


# ---- Plan: planner -> tool loop (internal to Track A) ----------------------
class PlanStep(TypedDict):
    tool: str  # must match a registered tool name
    reason: str  # why THIS ticker needs it (e.g. "earnings in 12 days")


class Plan(TypedDict):
    ticker: str
    steps: list[PlanStep]
    priority_note: str  # one line on what the plan emphasizes and why


# ---- Analysis: agent core -> evaluator (crosses A <-> C) -------------------
class EvidenceItem(TypedDict):
    """One number or fact the thesis relies on, so a reviewer can verify it."""

    claim: str  # e.g. "Forward P/E is about half of trailing P/E"
    value: float | str | None  # the number or fact itself
    source: str  # tool + field, e.g. "get_fundamentals.valuation.forwardPE"


class Analysis(TypedDict):
    ticker: str
    stance: Stance
    confidence: Confidence
    thesis: str  # the reasoning, tied to evidence_used
    evidence_used: list[EvidenceItem]
    risks: list[str]
    sources: list[str]  # news URLs and data sources cited
    data_gaps: list[str]  # what was missing, so confidence can be judged


# ---- Evaluation: evaluator -> notebook (crosses C <-> notebook) -------------
class CriterionScore(TypedDict):
    criterion: str  # one of RUBRIC_CRITERIA
    score: int  # 1 (poor) to 5 (excellent), given by the LLM judge
    comment: str  # why, so the optimizer knows what to fix


class Evaluation(TypedDict):
    verdict: Verdict  # decided by PYTHON from the scores, not by the LLM
    overall_score: float  # mean of criterion scores
    scores: list[CriterionScore]
    feedback: list[str]  # concrete fixes for the optimizer
    rounds_used: int  # capped at MAX_ROUNDS
    final_analysis: Analysis  # the analysis after the last revision