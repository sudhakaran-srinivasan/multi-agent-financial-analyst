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
        memory read         prior view of this ticker, as context only
        synthesize          signals + thesis + Python checks    (model + Python)
        [evaluate]          evaluator-optimizer loop, max 2 rounds
        memory write        append the final analysis with a timestamp

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
from pathlib import Path
from typing import Any

from finagent.agent.loop import run_research_loop
from finagent.agent.reflection import reflect
from finagent.agent.synthesis import SynthesisOutcome, analyze_run
from finagent.contracts import LoopResult, Reflection
from finagent.memory import (
    RunDiff,
    RunRecord,
    diff_runs,
    load_prior,
    record_from_outcome,
    save_run,
)

# Names of the stages that can record an error. A fixed list keeps reports
# and tests from depending on free-form strings.
STAGE_REFLECT = "reflect"
STAGE_SYNTHESIZE = "synthesize"
STAGE_MEMORY = "memory"


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
    prior: RunRecord | None = None  # last run of this ticker within the cutoff
    changes: RunDiff | None = None  # what changed since `prior`
    saved: bool = False  # True once this run was appended to memory

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
    memory_path: Path | None = None,
) -> PipelineResult:
    """Run every stage for ``ticker`` and return one PipelineResult.

    ``runner``, ``reflector`` and ``analyst`` are injectable (dependency
    injection) so tests run with no model and no network.

    ``memory_path`` turns memory on: the prior run is loaded before synthesis,
    and the finished analysis is appended afterwards. None (the default) means
    no file is read or written.

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

    # Stage: memory read. Context only; a failure here never blocks the run.
    if memory_path is not None and analyze:
        try:
            result.prior = load_prior(result.ticker, memory_path)
        except Exception as exc:  # noqa: BLE001  - stage boundary, recorded below
            result.stage_errors.append(StageError(STAGE_MEMORY, _describe(exc)))

    # Stage: synthesis.
    if analyze:
        try:
            result.outcome = analyst(result.loop, reflection=result.reflection)
        except Exception as exc:  # noqa: BLE001  - stage boundary, recorded below
            result.stage_errors.append(StageError(STAGE_SYNTHESIZE, _describe(exc)))

    # [evaluate goes here: revise up to MAX_ROUNDS, then keep the best draft]

    # Stage: memory write. A fallback analysis (the model's reply was rejected)
    # is not saved: it would teach the next run something we did not trust.
    outcome = result.outcome
    if memory_path is not None and outcome is not None and not outcome.fallback_reason:
        try:
            record = record_from_outcome(outcome)
            if result.prior is not None:
                result.changes = diff_runs(result.prior, record)
            save_run(record, memory_path)
            result.saved = True
        except Exception as exc:  # noqa: BLE001  - stage boundary, recorded below
            result.stage_errors.append(StageError(STAGE_MEMORY, _describe(exc)))
    return result


def stage_error_payload(errors: list[StageError]) -> list[dict[str, Any]]:
    """Stage errors as plain dicts (JSON-friendly)."""
    return [{"stage": e.stage, "error": e.error} for e in errors]