"""Evaluator-optimizer workflow: a judge scores the analysis, a reviser fixes it.

AGENT DESIGN (rubric: Workflow Patterns - evaluator-optimizer)
    Component : evaluate_and_optimize  (judge -> Python verdict -> revise, loop)
    Consumes  : a finished SynthesisOutcome plus the raw tool results
    Produces  : EvaluationRun (final outcome, every round, verdict, scores)
    Why       : one writing pass can leave the stance weaker than its evidence
                (for example the AADX run that ignored the news guidance). A
                second reader with a checklist catches that, and the writer
                gets a concrete list of fixes.

    The loop, in order:

        draft ---> JUDGE (LLM scores 6 criteria, 1-5)
                      |
                   PYTHON verdict: mean >= 4.0, no score < 3, no critical issue
                      |
              pass ---+--- revise --> REVISER (synthesis again, with the notes)
                                         |
                                      judge again (at most MAX_ROUNDS revisions)

    Who does what (AUTHOR: the split is the design):
        LLM judge     scores and comments; it never decides pass or fail
        Python        validates the scores, decides the verdict, stops the loop,
                      picks the best draft
        Reviser       is synthesis itself, so a revision passes the same Python
                      checks as a first draft (confidence ceiling, evidence
                      check, Python-built gaps and sources). The optimizer
                      cannot invent evidence to raise its own score.

    Failure policy:
        * An unusable judge reply is retried (MAX_JUDGE_ATTEMPTS), then the
          loop stops and keeps the best draft so far, marked not evaluated.
        * A revision that falls back to the rule-based analysis is not used.
        * After MAX_ROUNDS revisions that still fail, the BEST draft is
          returned and clearly marked as not passed. Never a silent pass.

KNOWN LIMITS (for the notebook, Evaluation and Iteration)
    * By default the same model writes and judges (config.JUDGE_MODEL), which
      tends to be lenient on its own writing.
    * The judge is not calibrated against human ratings yet.
    * A judge can miss an invented non-numeric claim; the number check in
      synthesis catches numbers only.

AI DISCLOSURE: first draft generated with Claude (Anthropic), combining the
team's two evaluator drafts (Sudha's rules, April's structured judge output,
retry and critical-issue ideas). Author-owned decisions are marked  # AUTHOR:.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from finagent import config
from finagent.agent.synthesis import SynthesisOutcome, synthesize
from finagent.contracts import (
    Analysis,
    CriterionScore,
    Evaluation,
    LoopResult,
    Reflection,
    Verdict,
)
from finagent.workflows.evaluation_rubric import build_rubric_text
from finagent.workflows.evaluation_rules import (
    MAX_ROUNDS,
    RUBRIC_CRITERIA,
    decide_verdict,
)

# AUTHOR/April: a judge reply that is not valid JSON, or misses a criterion,
# is asked for again this many times before the loop gives up.
MAX_JUDGE_ATTEMPTS = 3
# Scores below this earn a note for the reviser (4 or 5 is good enough).
NOTE_BELOW_SCORE = 4
MAX_FEEDBACK_ITEMS = 8
MAX_NOTE_CHARS = 300

JUDGE_OUTPUT_FORMAT = """\
Reply with JSON only, no other text, in exactly this shape:
{"scores": {"<criterion>": {"score": <whole number 1-5>, "comment": "<why>"}, \
...one entry for EVERY criterion above...},
 "strengths": ["..."], "issues": ["..."], "recommendations": ["..."],
 "critical_issues": ["..."]}
All criteria are REQUIRED. Scores are whole numbers. "recommendations" are \
concrete fixes the writer can make using only the data given."""


class JudgeError(ValueError):
    """The judge's reply was unusable. Retried, then the loop stops."""


@dataclass
class Judgment:
    """The judge's validated output (before Python decides the verdict)."""

    scores: list[CriterionScore]
    strengths: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    critical_issues: list[str] = field(default_factory=list)


@dataclass
class Round:
    """One evaluated draft: the analysis, the judgment, Python's verdict."""

    label: str  # "draft", "revision 1", ...
    outcome: SynthesisOutcome
    judgment: Judgment
    verdict: Verdict
    overall: float


@dataclass
class EvaluationRun:
    """Everything the loop did, for the report and the notebook table."""

    final: SynthesisOutcome  # the best draft (may be the first one)
    rounds: list[Round] = field(default_factory=list)
    revisions: int = 0  # how many revisions were written
    passed: bool = False
    stopped_because: str = ""  # human-readable reason the loop ended
    notes: list[str] = field(default_factory=list)

    @property
    def best(self) -> Round | None:
        return _best_round(self.rounds)

    @property
    def evaluated(self) -> bool:
        return bool(self.rounds)


Judge = Callable[[Analysis, dict[str, Any], list[Any]], Judgment]
Reviser = Callable[[list[str], Analysis], SynthesisOutcome]


# ---- parsing and validating the judge ----------------------------------------
def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [v.strip() for v in value if isinstance(v, str) and v.strip()]


def parse_judgment(text: str) -> Judgment:
    """Turn the judge's reply into a validated Judgment, or raise JudgeError.

    Strict on purpose: a missing criterion or a score outside 1-5 means the
    judge did not follow the rubric, and a quietly "fixed" score would be a
    number we invented. bool is rejected because True == 1 in Python.
    """
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise JudgeError("no JSON object found in the judge's reply")
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise JudgeError(f"invalid JSON: {exc}") from exc
    raw = data.get("scores") if isinstance(data, dict) else None
    if not isinstance(raw, dict):
        raise JudgeError("reply has no 'scores' object")

    scores: list[CriterionScore] = []
    for criterion in RUBRIC_CRITERIA:
        entry = raw.get(criterion)
        if not isinstance(entry, dict):
            raise JudgeError(f"missing score for {criterion}")
        value = entry.get("score")
        if isinstance(value, bool) or not isinstance(value, int):
            raise JudgeError(f"{criterion} score must be a whole number, "
                             f"got {value!r}")
        if not 1 <= value <= 5:
            raise JudgeError(f"{criterion} score must be 1 to 5, got {value}")
        comment = entry.get("comment")
        scores.append(CriterionScore(
            criterion=criterion, score=value,
            comment=comment.strip() if isinstance(comment, str) else ""))
    return Judgment(
        scores=scores, strengths=_strings(data.get("strengths")),
        issues=_strings(data.get("issues")),
        recommendations=_strings(data.get("recommendations")),
        critical_issues=_strings(data.get("critical_issues")))


def build_judge_messages(analysis: Analysis, results: dict[str, Any],
                         signals: list[Any]) -> list[dict[str, str]]:
    """System prompt (rubric + format) and user message (the data to judge).

    The analysis and tool data go in the USER message only: they contain text
    from news articles, which must never be able to give the judge orders.
    """
    system = (build_rubric_text() + "\n\n" + JUDGE_OUTPUT_FORMAT)
    user = ("Judge this analysis against the raw data.\n\n"
            + json.dumps({"analysis": analysis, "raw_tool_results": results,
                          "signals_computed_by_python": signals},
                         indent=2, default=str))
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


def make_llm_judge(client: Any) -> Judge:
    """The default judge: one LLM call, retried when the reply is unusable."""

    def judge(analysis: Analysis, results: dict[str, Any],
              signals: list[Any]) -> Judgment:
        messages = build_judge_messages(analysis, results, signals)
        last_error = "no attempt made"
        for attempt in range(1, MAX_JUDGE_ATTEMPTS + 1):
            if attempt > 1:
                messages = messages[:2] + [{
                    "role": "user",
                    "content": f"Your previous reply was unusable ({last_error}). "
                               "Return ALL criteria as the JSON shape described."}]
            response = client.chat.completions.create(
                model=config.JUDGE_MODEL, messages=messages,
                max_tokens=config.MAX_TOKENS, temperature=config.TEMPERATURE)
            try:
                return parse_judgment(response.choices[0].message.content or "")
            except JudgeError as exc:
                last_error = str(exc)
        raise JudgeError(f"judge failed {MAX_JUDGE_ATTEMPTS} times: {last_error}")

    return judge


# ---- the verdict and the feedback (Python) -----------------------------------
def build_feedback(judgment: Judgment) -> list[str]:
    """Concrete notes for the reviser: critical issues first, then weak scores,
    then the judge's recommendations. Deduplicated and capped."""
    notes = [f"CRITICAL: {issue}" for issue in judgment.critical_issues]
    for score in judgment.scores:
        if score["score"] < NOTE_BELOW_SCORE:
            detail = f": {score['comment']}" if score["comment"] else ""
            notes.append(f"{score['criterion']} scored {score['score']}{detail}")
    notes += judgment.recommendations
    unique = list(dict.fromkeys(n[:MAX_NOTE_CHARS] for n in notes))
    return unique[:MAX_FEEDBACK_ITEMS]


def _rank(round_: Round) -> tuple[int, int, float]:
    """Higher is better: a pass, then fewer critical issues, then the score."""
    return (int(round_.verdict == "pass"),
            -len(round_.judgment.critical_issues), round_.overall)


def _best_round(rounds: list[Round]) -> Round | None:
    """The best draft. On a tie the EARLIER one wins, so a revision has to be
    strictly better to replace the original."""
    best: Round | None = None
    for round_ in rounds:
        if best is None or _rank(round_) > _rank(best):
            best = round_
    return best


def _evaluate(label: str, outcome: SynthesisOutcome, results: dict[str, Any],
              judge: Judge) -> Round:
    judgment = judge(outcome.analysis, results, outcome.signals)
    verdict, overall = decide_verdict(
        judgment.scores, critical_issues=judgment.critical_issues)
    return Round(label, outcome, judgment, verdict, overall)


# ---- the loop ----------------------------------------------------------------
def evaluate_and_optimize(
    draft: SynthesisOutcome, results: dict[str, Any], judge: Judge,
    reviser: Reviser, max_rounds: int = MAX_ROUNDS,
) -> EvaluationRun:
    """Judge the draft; while it fails, revise and judge again (max_rounds).

    Never raises for judge or reviser problems: the run records why it stopped
    and returns the best draft it has. ``judge`` and ``reviser`` are injected
    so the loop is tested with no model and no network.
    """
    run = EvaluationRun(final=draft)
    current = draft
    for index in range(max_rounds + 1):
        label = "draft" if index == 0 else f"revision {index}"
        try:
            round_ = _evaluate(label, current, results, judge)
        except Exception as exc:  # noqa: BLE001  - recorded, loop stops safely
            run.stopped_because = (
                f"judge failed on the {label}: {type(exc).__name__}: {exc}")
            break
        run.rounds.append(round_)
        if round_.verdict == "pass":
            run.stopped_because = f"the {label} passed"
            break
        if index == max_rounds:
            run.stopped_because = (
                f"still failing after {max_rounds} revision(s); "
                "kept the best draft")
            break
        feedback = build_feedback(round_.judgment)
        try:
            revised = reviser(feedback, current.analysis)
        except Exception as exc:  # noqa: BLE001  - recorded, loop stops safely
            run.stopped_because = (
                f"reviser failed: {type(exc).__name__}: {exc}")
            break
        if revised.fallback_reason:
            run.stopped_because = ("the revision fell back to the rule-based "
                                   f"analysis ({revised.fallback_reason}); "
                                   "kept the best earlier draft")
            break
        run.revisions += 1
        current = revised

    best = run.best
    if best is not None:
        run.final = best.outcome
        run.passed = best.verdict == "pass"
    else:
        run.notes.append("not evaluated: " + run.stopped_because)
    if best is not None and not run.passed:
        run.notes.append("the final analysis did NOT pass evaluation")
    return run


def to_evaluation(run: EvaluationRun) -> Evaluation | None:
    """The contract shape (contracts.Evaluation); None when never evaluated."""
    best = run.best
    if best is None:
        return None
    return Evaluation(
        verdict=best.verdict, overall_score=best.overall,
        scores=best.judgment.scores, feedback=build_feedback(best.judgment),
        rounds_used=run.revisions, final_analysis=run.final.analysis)


def evaluate_run(loop_result: LoopResult, outcome: SynthesisOutcome,
                 reflection: Reflection | None = None,
                 client: Any = None) -> EvaluationRun:
    """Entry-point wiring: the real judge, and synthesis as the reviser.

    Builds the model client when none is given (a missing API key then fails
    here, with a message that names it).
    """
    if client is None:
        from finagent.agent.loop import make_client

        client = make_client()

    def reviser(feedback: list[str], previous: Analysis) -> SynthesisOutcome:
        return synthesize(loop_result, client, reflection,
                          feedback=feedback, previous=previous)

    return evaluate_and_optimize(outcome, loop_result["results"],
                                 make_llm_judge(client), reviser)
