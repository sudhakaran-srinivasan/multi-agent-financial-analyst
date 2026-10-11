"""Reflection: the agent checks its own run and fixes what a retry can fix.

Agent Design (rubric: Agent Functions, "self-reflection").
    The tool loop can end with holes: a tool failed, the step cap cut the run
    short, or a tool "succeeded" but reported a failed fetch. Before any
    analysis is written, reflection looks at the run and asks one question per
    hole: could trying again help?

        retry (once)      a planned tool failed, was never called (step cap),
                          or returned a ``fetch_failed`` gap
        never retry       permanent gaps (a field the company does not report,
                          a P/E that is not meaningful, one quarter of history
                          on a new listing), and fatal errors

    Python decides and Python runs the retries. Reflection spends NO model
    steps, so extending the work after a step cap cannot turn into a loop,
    and the worst case is one extra call per planned tool (which also
    protects the NewsAPI daily quota).

    Reflection sits between the loop and synthesis:
        LoopResult -> reflect() -> (updated LoopResult, Reflection) -> synthesis
    The input is never changed; a new LoopResult is returned, and the retry
    calls are appended to the trace (step = steps_used + 1).

AI DISCLOSURE: structure drafted with Claude (Anthropic). The retry rules and
the one-attempt budget follow decisions made by the project owner; lines
marked ``# AUTHOR:`` are theirs to review.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from finagent.agent.dispatcher import dispatch
from finagent.agent.synthesis import build_data_gaps
from finagent.contracts import LoopResult, Reflection, RetryRecord, RetryWhy
from finagent.tools.tool_specs import TOOL_FUNCTIONS

# AUTHOR: confirm. A failed fetch can be transient (network, rate limit), so
# one more try is worth it. Gaps like "missing" or "not_meaningful" describe
# the company, so retrying them would only waste calls.
RETRYABLE_GAP_REASONS = frozenset({"fetch_failed"})
# AUTHOR: confirm. One extra attempt per tool, never a loop.
MAX_RETRIES_PER_TOOL = 1


def _has_retryable_gap(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    return any(isinstance(gap, dict) and gap.get("reason") in RETRYABLE_GAP_REASONS
               for gap in data.get("gaps", []))


def _retry_reason(tool: str, loop_result: LoopResult) -> RetryWhy | None:
    """Why this planned tool deserves another attempt, or None."""
    if tool not in loop_result["results"]:
        failed = any(step["tool"] == tool and not step["ok"]
                     for step in loop_result["trace"])
        return "failed" if failed else "never_called"
    if _has_retryable_gap(loop_result["results"][tool]):
        return "fetch_failed"
    return None


def _summary(retried: list[RetryRecord], remaining: list[str]) -> str:
    count = len(remaining)
    gaps = f"{count} gap remains" if count == 1 else f"{count} gaps remain"
    if not retried:
        return f"Nothing to retry; {gaps} (retrying cannot fix them)."
    recovered = sum(r["outcome"] == "recovered" for r in retried)
    still = len(retried) - recovered
    noun = "tool" if len(retried) == 1 else "tools"
    return (f"{len(retried)} {noun} retried: {recovered} recovered, "
            f"{still} still failing; {gaps}.")


def reflect(
    loop_result: LoopResult,
    *,
    functions: dict[str, Callable[..., Any]] | None = None,
    specs: list[dict[str, Any]] | None = None,
) -> tuple[LoopResult, Reflection]:
    """Retry what a retry can fix, once, and report what remains.

    Args:
        loop_result: The finished loop. Not modified.
        functions: Tool name -> function (defaults to the real tools).
        specs: Tool menu used to validate arguments (defaults to the real one).

    Returns:
        A new ``LoopResult`` (recovered results added, retry calls appended
        to the trace) and a ``Reflection`` record. Never raises for a tool
        problem: the dispatcher turns errors into outcomes.
    """
    if loop_result["stop_reason"] == "fatal_error":
        return loop_result, {
            "retried": [], "remaining_gaps": [],
            "summary": "The run ended in a fatal error; nothing to reflect on.",
        }

    functions = TOOL_FUNCTIONS if functions is None else functions
    plan = loop_result["plan"]
    planned = [step["tool"] for step in plan["steps"]] if plan else list(functions)

    results = dict(loop_result["results"])
    trace = list(loop_result["trace"])
    fixup_step = loop_result["steps_used"] + 1
    retried: list[RetryRecord] = []

    for tool in planned:
        why = _retry_reason(tool, loop_result)
        if why is None:
            continue
        # One attempt per tool (MAX_RETRIES_PER_TOOL = 1): no loop here.
        outcome = dispatch(tool, json.dumps({"ticker": loop_result["ticker"]}),
                           functions=functions, specs=specs)
        trace.append({"step": fixup_step, "tool": outcome.name,
                      "arguments": outcome.arguments, "ok": outcome.ok,
                      "error": outcome.error})
        if outcome.ok:
            results[tool] = outcome.data
        recovered = outcome.ok and not _has_retryable_gap(outcome.data)
        retried.append({
            "tool": tool, "why": why,
            "outcome": "recovered" if recovered else "still_failing",
            "error": None if recovered else (outcome.error or "fetch_failed gap"),
        })

    updated: LoopResult = {**loop_result, "results": results, "trace": trace}
    remaining = build_data_gaps(updated)
    reflection: Reflection = {
        "retried": retried, "remaining_gaps": remaining,
        "summary": _summary(retried, remaining),
    }
    return updated, reflection