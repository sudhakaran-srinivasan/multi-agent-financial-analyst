"""Scoring rules for the evaluator-optimizer loop.

AGENT DESIGN (rubric: Workflow Patterns - evaluator-optimizer)
    The LLM judge gives per-criterion scores. PYTHON decides the verdict, so
    the decision is deterministic, testable and explainable.

    Analogy: a factory inspector uses a checklist, and one critical defect
    fails the part even if the average is great (the FLOOR rule). A teacher
    grades each rubric line, then totals (the AVERAGE rule).

    Proposal for April (Track C) to confirm or change.

AI DISCLOSURE: 
"""
from __future__ import annotations

from finagent.contracts import CriterionScore, Verdict

# AUTHOR/April: what the judge scores. Each maps to a question about the draft.
RUBRIC_CRITERIA: tuple[str, ...] = (
    "evidence_support",  # is every claim backed by a specific number?
    "numeric_accuracy",  # do cited numbers match the data the tools returned?
    "stance_consistency",  # does the stance follow from the evidence?
    "risk_coverage",  # are the real risks and data gaps stated?
    "confidence_calibration",  # is confidence honest given gaps and evidence?
)
# AUTHOR/April: pass needs a high average AND no critical defect.
PASS_AVERAGE = 4.0
CRITERION_FLOOR = 3  # any score below this forces "revise"
MAX_ROUNDS = 2  # hard cap on revise loops, set in OUR code


def decide_verdict(
    scores: list[CriterionScore],
    pass_average: float = PASS_AVERAGE,
    floor: int = CRITERION_FLOOR,
) -> tuple[Verdict, float]:
    """Return (verdict, overall_score). No scores means we cannot approve."""
    if not scores:
        return "revise", 0.0
    values = [s["score"] for s in scores]
    overall = sum(values) / len(values)
    if overall >= pass_average and min(values) >= floor:
        return "pass", overall
    return "revise", overall