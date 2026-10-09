"""Shared data shapes ("contracts") between components and between teammates.

AGENT DESIGN (rubric: Agent Design / Functions)
    Why        : each contract is the agreement between two pieces of code.
                 Everyone codes against the same file, so a mismatch shows up
                 immediately instead of at integration time.
    Who agrees : the three that cross a track boundary need team agreement
                 before changing: NewsInsight (A <-> B), Analysis (A <-> C),
                 Evaluation (C <-> notebook). Plan is internal to Track A.
    Profile    : lives in finagent.tools.company_profile (Track A only).
    LoopResult : internal to Track A, but its `results` field (the raw tool
                 outputs) is the fact base the evaluator checks numbers against.

    Flow:  ticker -> Profile -> Plan -> tool loop -> LoopResult
                                   |        -> Analysis -> Evaluation
                                   ^ NewsInsight list comes from Jeff's pipeline

TypedDicts are plain dicts at runtime (json.dumps works); Literal types list
the only allowed values. Python does not enforce them at runtime, so
tests/test_contracts.py checks the samples against these definitions.

AI DISCLOSURE: 
"""
from __future__ import annotations

from typing import Any, Literal, TypedDict

# ---- allowed values -------------------------------------------------------
# Why the tool loop stopped: the model said it was finished, the step cap was
# hit, or a core-data error (bad ticker, not a stock) ended the run.
StopReason = Literal["done", "step_cap", "fatal_error"]
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


# ---- LoopResult: tool loop -> synthesis, reflection, evaluator (Track A) ----
class TraceStep(TypedDict):
    """One tool call, kept as the audit trail (the notebook shows these)."""

    step: int  # 0 = the pre-flight profile check; 1..N = model turns
    tool: str
    arguments: dict[str, Any]
    ok: bool
    error: str | None  # the text the model saw when ok is False


class LoopResult(TypedDict):
    """Everything the loop hands over when it stops.

    Why these fields (each has a reader):
        results      -> synthesis writes from it; the evaluator checks
                        numbers against it. Python-computed facts, never
                        the model's prose.
        trace        -> reflection (what was tried and failed) and notebook.
        stop_reason  -> reflection: step_cap means "maybe incomplete".
    """

    ticker: str
    stop_reason: StopReason
    steps_used: int  # model turns used; the pre-flight check does not count
    profile: dict[str, Any] | None  # None when the pre-flight check failed
    results: dict[str, Any]  # tool name -> latest successful structured output
    trace: list[TraceStep]  # every call in order, including failed ones
    fatal_error: str | None  # set only when stop_reason == "fatal_error"


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