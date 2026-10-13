"""The research pipeline: one function that chains every stage for a ticker.

AGENT DESIGN (rubric: Agent Design / Functions)
    Component : run_pipeline  (the conductor; it contains no analysis logic)
    Consumes  : a ticker
    Produces  : PipelineResult (loop result, reflection, analysis, stage errors)
    Why       : the stages used to be wired together inside the command-line
                file, mixed with argument parsing and printing. Pulling the
                wiring out gives three benefits:
                  1. the notebook, the CLI and the tests all call ONE function;
                  2. each stage can fail on its own without losing the work
                     already done (a crash in synthesis still returns the
                     data the loop collected);
                  3. memory and the evaluator slot in as new stages without
                     touching the others.

    Stages, in order (a stage in [brackets] is planned, not built yet):

        run_research_loop   profile -> plan -> tool loop        (model decides)
        reflect             Python retries what a retry can fix (code decides)
        [memory read]       prior view of this ticker, as context only
        synthesize          signals + thesis + Python checks    (model + Python)
        [evaluate]          evaluator-optimizer loop, max 2 rounds
        [memory write]      store the final analysis with a timestamp

    Failure policy (AUTHOR: decision, see "Partial results" below):
        * A SETUP problem (missing API key) raises, because nothing can run.
        * A FATAL data problem (bad ticker) is a normal result: the loop
          returns stop_reason "fatal_error" and later stages are skipped.
        * A crash in a LATER stage (reflection, synthesis) is caught, recorded
          in `stage_errors`, and the pipeline returns what it has so far.

AI DISCLOSURE: first draft generated with Claude (Anthropic), reviewed and
modified by the author. Author-owned decisions are marked  # AUTHOR:.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from finagent.agent.loop import run_research_loop
from finagent.agent.reflection import reflect
from finagent.agent.synthesis import SynthesisOutcome, analyze_run
from finagent.contracts import LoopResult, Reflection

# Names of the stages that can record an error. A fixed list keeps reports
# and tests from depending on free-form strings.
STAGE_REFLECT = "reflect"
STAGE_SYNTHESIZE = "synthesize"


@dataclass
class StageError:
    """A later stage crashed; the pipeline carried on without it."""

    stage: str
    error: str  # "ExceptionType: message"


@dataclass
class PipelineResult:
    """Everything one run produced. Memory and the evaluator will read this."""

    ticker: str
    loop: LoopResult  # after reflection (retries applied), if reflection ran
    reflection: Reflection | None = None  # None: skipped or crashed
    outcome: SynthesisOutcome | None = None  # None: not asked, skipped or crashed
    stage_errors: list[StageError] = field(default_factory=list)

    @property
    def fatal(self) -> bool:
        """True when the data loop could not start (bad ticker, not a stock)."""
        return self.loop["stop_reason"] == "fatal_error"


def _describe(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def run_pipeline(
    ticker: str,
    *,
    analyze: bool = True,
    runner: Callable[[str], LoopResult] = run_research_loop,
    reflector: Callable[[LoopResult], tuple[LoopResult, Reflection]] = reflect,
    analyst: Callable[..., SynthesisOutcome] = analyze_run,
) -> PipelineResult:
    """Run every stage for ``ticker`` and return one PipelineResult.

    ``runner``, ``reflector`` and ``analyst`` are injectable (dependency
    injection) so tests run with no model and no network.

    Raises RuntimeError for a setup problem such as a missing API key, because
    no stage can run without it.
    """
    loop_result = runner(ticker)  # setup problems propagate to the caller
    result = PipelineResult(ticker=loop_result["ticker"], loop=loop_result)
    if result.fatal:
        return result  # nothing to fix up or analyze

    # Stage: reflection. A crash here is not worth losing the collected data.
    try:
        result.loop, result.reflection = reflector(loop_result)
    except Exception as exc:  # noqa: BLE001  - stage boundary, recorded below
        result.stage_errors.append(StageError(STAGE_REFLECT, _describe(exc)))

    # [memory read goes here: the prior view is passed to synthesis as context]

    # Stage: synthesis.
    if analyze:
        try:
            result.outcome = analyst(result.loop, reflection=result.reflection)
        except Exception as exc:  # noqa: BLE001  - stage boundary, recorded below
            result.stage_errors.append(StageError(STAGE_SYNTHESIZE, _describe(exc)))

    # [evaluate goes here: revise up to MAX_ROUNDS, then keep the best draft]
    # [memory write goes here: store result.outcome.analysis with a timestamp]
    return result


def stage_error_payload(errors: list[StageError]) -> list[dict[str, Any]]:
    """Stage errors as plain dicts (JSON-friendly)."""
    return [{"stage": e.stage, "error": e.error} for e in errors]