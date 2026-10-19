"""Command-line entry point: run the research loop for one ticker.

    python -m finagent.run_agent NVDA
    python -m finagent.run_agent NVDA --json     # full LoopResult as JSON
    python -m finagent.run_agent NVDA --analyze  # also write the analysis

Why a separate file? The stages are chained in finagent.pipeline, a library
function the notebook and the tests call too. This file is the thin "front
door" for a human at a terminal: it parses the command line, calls the
pipeline, prints a readable report, and returns an exit code.

Exit codes: 0 = finished (done or step_cap), 1 = fatal_error (bad ticker,
not a stock, or the model API failed), 2 = setup problem (for example a
missing API key).

AI use: see the AI note in README.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from finagent.agent.loop import run_research_loop
from finagent.agent.reflection import reflect
from finagent.agent.signals import Signal
from finagent.agent.synthesis import DISCLAIMER, SynthesisOutcome, analyze_run
from finagent.contracts import Analysis, LoopResult, Plan, Reflection
from finagent.memory import DEFAULT_PATH, format_diff
from finagent.pipeline import (
    PipelineResult,
    StageError,
    run_pipeline,
    stage_error_payload,
)
from finagent.workflows.evaluator_optimizer import (
    EvaluationRun,
    evaluate_run,
    to_evaluation,
)

_MAX_ERROR_CHARS = 100


def _profile_line(profile: dict[str, Any] | None) -> str:
    """One readable line about the company, skipping unknown (None) parts."""
    if profile is None:
        return "Profile: none (the pre-flight check failed)"
    parts = [f"{profile.get('name')} ({profile.get('quote_type')})"]
    if profile.get("size_bucket"):
        parts.append(f"{profile['size_bucket']}-cap")
    if profile.get("is_profitable") is not None:
        parts.append("profitable" if profile["is_profitable"] else "not profitable")
    if profile.get("days_to_earnings") is not None:
        parts.append(f"earnings in {profile['days_to_earnings']} days")
    return "Profile: " + ", ".join(parts)


def _plan_lines(plan: Plan | None) -> list[str]:
    """The plan as readable lines; empty when the run stopped before planning."""
    if plan is None:
        return []
    lines = ["", "Plan:"]
    lines += [f"  {i}. {step['tool']}: {step['reason']}"
              for i, step in enumerate(plan["steps"], start=1)]
    lines.append(f"  Emphasis: {plan['priority_note']}")
    return lines


def _reflection_lines(reflection: Reflection | None) -> list[str]:
    """Reflection as readable lines; empty when reflection did not run."""
    if reflection is None:
        return []
    lines = ["", f"Reflection: {reflection['summary']}"]
    for item in reflection["retried"]:
        detail = f" ({item['error']})" if item["error"] else ""
        lines.append(f"  retried {item['tool']} [{item['why']}]: "
                     f"{item['outcome']}{detail}"[:160])
    return lines


def _result_line(name: str, data: Any) -> str:
    """Summarize one tool result: gaps and warnings, or an item count."""
    if isinstance(data, dict):
        gaps = data.get("gaps", [])
        shown = ", ".join(
            f"{g.get('field')}: {g.get('reason')}" if isinstance(g, dict) else str(g)
            for g in gaps[:4]
        )
        more = f", +{len(gaps) - 4} more" if len(gaps) > 4 else ""
        line = f"gaps: {len(gaps)}" + (f" [{shown}{more}]" if gaps else "")
        if "warnings" in data:
            line += f"   warnings: {len(data['warnings'])}"
        return f"  {name:<22} {line}"
    if isinstance(data, list):
        return f"  {name:<22} {len(data)} items"
    return f"  {name:<22} ok"


def format_report(result: LoopResult) -> str:
    """Turn a LoopResult into a short, human-readable report (pure function)."""
    lines = [
        (f"{result['ticker']}  |  stop: {result['stop_reason']}"
         f"  |  model steps: {result['steps_used']}"),
        _profile_line(result["profile"]),
    ]
    lines += _plan_lines(result["plan"])
    lines += [
        "",
        "Trace (step 0 = pre-flight check):",
    ]
    for item in result["trace"]:
        status = "ok" if item["ok"] else "ERROR: " + (item["error"] or "")[:_MAX_ERROR_CHARS]
        lines.append(f"  {item['step']:>2}  {item['tool']:<22} {status}")

    lines += ["", "Results collected:"]
    if result["results"]:
        lines += [_result_line(n, d) for n, d in result["results"].items()]
    else:
        lines.append("  (none)")
    if result["fatal_error"]:
        lines += ["", f"FATAL: {result['fatal_error']}"]
    return "\n".join(lines)


_MARKS = {"positive": "[+]", "neutral": "[=]", "negative": "[-]"}


def _format_value(value: float | str | None) -> str:
    """Compact display for a signal value: 1.27B, 0.0452, n/a."""
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    magnitude = abs(value)
    for size, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if magnitude >= size:
            return f"{value / size:.1f}{suffix}"
    return f"{value:,.4g}"


def format_scorecard(signals: list[Signal]) -> str:
    """The signals as a table: what was measured, its value, how to read it.

    Output layer only (the Analysis contract is untouched). The "Read" column
    is the direction Python computed from documented thresholds, not a letter
    grade: we do not invent a proprietary scale.
    """
    if not signals:
        return "  (no signals)"
    name_w = max(len("Signal"), *(len(s["name"]) for s in signals))
    value_w = max(len("Value"), *(len(_format_value(s["value"])) for s in signals))
    lines = [f"  {'Signal':<{name_w}}  {'Value':>{value_w}}  Read  Why",
             f"  {'-' * name_w}  {'-' * value_w}  ----  ---"]
    lines += [
        f"  {s['name']:<{name_w}}  {_format_value(s['value']):>{value_w}}  "
        f"{_MARKS[s['direction']]}  {s['why']}"
        for s in signals
    ]
    return "\n".join(lines)


def format_analysis(outcome: SynthesisOutcome) -> str:
    """The analysis as readable text (pure function), ending with the disclaimer."""
    analysis: Analysis = outcome.analysis
    lines = [
        "",
        (f"ANALYSIS  |  stance: {analysis['stance']}"
         f"  |  confidence: {analysis['confidence']}"),
        "",
        analysis["thesis"],
        "",
        "Scorecard (computed by Python; [+] positive, [=] neutral, [-] negative):",
        format_scorecard(outcome.signals),
    ]
    lines += ["", "Evidence:"]
    lines += [f"  - {e['claim']} = {e['value']}  <{e['source']}>"
              for e in analysis["evidence_used"]] or ["  (none)"]
    lines += ["", "Risks:"] + [f"  - {r}" for r in analysis["risks"]]
    lines += ["", "Data gaps:"]
    lines += [f"  - {g}" for g in analysis["data_gaps"]] or ["  (none)"]
    if analysis["sources"]:
        lines += ["", "Sources:"] + [f"  - {s}" for s in analysis["sources"]]
    if outcome.fallback_reason:
        lines += ["", f"NOTE: model analysis not used ({outcome.fallback_reason})"]
    lines += [f"NOTE: {note}" for note in outcome.notes]
    lines += ["", DISCLAIMER]
    return "\n".join(lines)


def _stage_error_lines(errors: list[StageError]) -> list[str]:
    """Stage failures as readable lines; empty when every stage ran."""
    return ["", "STAGE ERRORS (the run continued without these):"] + [
        f"  {e.stage}: {e.error}" for e in errors] if errors else []


def _evaluation_lines(run: EvaluationRun | None) -> list[str]:
    """The evaluator's work as readable lines; empty when it did not run."""
    if run is None:
        return []
    if not run.evaluated:
        return ["", "Evaluation: not evaluated (" + run.stopped_because + ")"]
    best = run.best
    status = "PASSED" if run.passed else "DID NOT PASS"
    lines = ["", (f"Evaluation: {status}  |  score {best.overall:.2f}/5"
                  f"  |  revisions: {run.revisions}"
                  f"  |  {run.stopped_because}")]
    lines += [f"  {r.label:<11} {r.verdict:<6} {r.overall:.2f}  "
              + " ".join(f"{s['score']}" for s in r.judgment.scores)
              for r in run.rounds]
    lines.append("  (scores in order: " + ", ".join(
        s["criterion"] for s in best.judgment.scores) + ")")
    lines += [f"  critical: {c}" for c in best.judgment.critical_issues]
    return lines


def _memory_lines(run: PipelineResult) -> list[str]:
    """What memory did this run: compared with a prior run, or first of its kind."""
    if run.changes is not None:
        return [""] + format_diff(run.changes)
    if run.saved:
        return ["", "Memory: first run for this ticker within 30 days; saved."]
    return []


def main(
    argv: Sequence[str] | None = None,
    runner: Callable[[str], LoopResult] = run_research_loop,
    analyst: Callable[..., SynthesisOutcome] = analyze_run,
    reflector: Callable[[LoopResult], tuple[LoopResult, Reflection]] = reflect,
    memory_path: Path = DEFAULT_PATH,
    evaluator: Callable[..., EvaluationRun] = evaluate_run,
) -> int:
    """Parse arguments, run the pipeline, print the report, return an exit code.

    ``runner``, ``reflector`` and ``analyst`` are injectable so tests need no
    model and no network.
    """
    parser = argparse.ArgumentParser(
        prog="python -m finagent.run_agent",
        description="Run the research agent's data-gathering loop for a ticker.",
    )
    parser.add_argument("ticker", help="stock symbol, for example NVDA")
    parser.add_argument("--json", action="store_true",
                        help="print the full result as JSON")
    parser.add_argument("--analyze", action="store_true",
                        help="after the data loop, write the analysis")
    parser.add_argument("--evaluate", action="store_true",
                        help="judge the analysis and let it be revised (needs --analyze)")
    parser.add_argument("--no-memory", action="store_true",
                        help="do not read or write the run memory file")
    args = parser.parse_args(argv)

    try:
        run = run_pipeline(args.ticker, analyze=args.analyze, runner=runner,
                           reflector=reflector, analyst=analyst,
                           memory_path=None if args.no_memory else memory_path,
                           evaluator=evaluator if args.evaluate and args.analyze
                           else None)
    except RuntimeError as exc:  # for example a missing API key
        print(f"Setup problem: {exc}", file=sys.stderr)
        return 2

    if args.json:
        payload: dict[str, Any] = {"loop": run.loop} if args.analyze else dict(run.loop)
        payload["reflection"] = run.reflection
        payload["stage_errors"] = stage_error_payload(run.stage_errors)
        payload["evaluation"] = to_evaluation(run.evaluation) if run.evaluation else None
        payload["memory"] = {"saved": run.saved, "prior": run.prior,
                             "changes": run.changes}
        if run.outcome is not None:
            payload.update(
                analysis=run.outcome.analysis, signals=run.outcome.signals,
                analysis_fallback=run.outcome.fallback_reason,
                analysis_notes=run.outcome.notes, disclaimer=DISCLAIMER)
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(format_report(run.loop))
        print("\n".join(_reflection_lines(run.reflection)))
        if run.outcome is not None:
            print(format_analysis(run.outcome))
            print("\n".join(_evaluation_lines(run.evaluation)))
            print("\n".join(_memory_lines(run)))
        print("\n".join(_stage_error_lines(run.stage_errors)))
    return 1 if run.fatal else 0


if __name__ == "__main__":
    # Runs only when started as a script (python -m finagent.run_agent),
    # never when this file is imported, for example by the tests.
    raise SystemExit(main())
