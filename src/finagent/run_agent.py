"""Command-line entry point: run the research loop for one ticker.

    python -m finagent.run_agent NVDA
    python -m finagent.run_agent NVDA --json     # full LoopResult as JSON
    python -m finagent.run_agent NVDA --analyze  # also write the analysis

Why a separate file? The loop is a library function other code (the notebook,
synthesis, tests) calls. This file is the thin "front door" for a human at a
terminal: it parses the command line, calls the loop, prints a readable
report, and returns an exit code.

Exit codes: 0 = finished (done or step_cap), 1 = fatal_error (bad ticker,
not a stock, or the model API failed), 2 = setup problem (for example a
missing API key).

AI DISCLOSURE: structure drafted with Claude (Anthropic).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from typing import Any

from finagent.agent.loop import run_research_loop
from finagent.agent.reflection import reflect
from finagent.agent.signals import Signal
from finagent.agent.synthesis import DISCLAIMER, SynthesisOutcome, analyze_run
from finagent.contracts import Analysis, LoopResult, Plan, Reflection

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


def _signal_line(signal: Signal) -> str:
    marks = {"positive": "+", "neutral": "=", "negative": "-"}
    return f"  [{marks[signal['direction']]}] {signal['name']}: {signal['why']}"


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
        "Signals computed by Python ([+] positive, [=] neutral, [-] negative):",
    ]
    lines += [_signal_line(s) for s in outcome.signals] or ["  (none)"]
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


def main(
    argv: Sequence[str] | None = None,
    runner: Callable[[str], LoopResult] = run_research_loop,
    analyst: Callable[..., SynthesisOutcome] = analyze_run,
    reflector: Callable[[LoopResult], tuple[LoopResult, Reflection]] = reflect,
) -> int:
    """Parse arguments, run the loop, print the report, return an exit code.

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
    args = parser.parse_args(argv)

    try:
        result = runner(args.ticker)
    except RuntimeError as exc:  # for example a missing API key
        print(f"Setup problem: {exc}", file=sys.stderr)
        return 2

    fatal = result["stop_reason"] == "fatal_error"
    reflection: Reflection | None = None
    if not fatal:  # fix what a retry can fix before anything is written
        result, reflection = reflector(result)
    outcome = analyst(result, reflection=reflection) if args.analyze and not fatal \
        else None

    if args.json:
        payload: dict[str, Any] = {"loop": result} if args.analyze else dict(result)
        payload["reflection"] = reflection
        if outcome is not None:
            payload.update(
                analysis=outcome.analysis, signals=outcome.signals,
                analysis_fallback=outcome.fallback_reason, analysis_notes=outcome.notes,
                disclaimer=DISCLAIMER)
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(format_report(result))
        print("\n".join(_reflection_lines(reflection)))
        if outcome is not None:
            print(format_analysis(outcome))
    return 1 if fatal else 0


if __name__ == "__main__":
    # Runs only when started as a script (python -m finagent.run_agent),
    # never when this file is imported, for example by the tests.
    raise SystemExit(main())