"""The tool-calling loop: the model decides which tool to call next.

Agent Design (rubric: Agent Functions, "uses tools dynamically").
    Workflow vs agent: in a workflow, CODE picks the next step. Here the
    MODEL picks it, one turn at a time, so this is the agent loop. Python
    only (a) checks the ticker first, (b) runs what the model asks for
    safely, and (c) enforces a hard step cap.

    One run:
        pre-flight   Python runs get_company_profile. A bad ticker or a
                     non-stock (ETF, index) stops here, before any model
                     call, so it costs nothing. The profile is then handed
                     to the model as starting context.
        loop         model turn -> tool calls -> results back to the model.
                     Repeats until the model calls no tool ("done"), the
                     step cap is hit ("step_cap"), or a core-data error or
                     model-API failure happens ("fatal_error").
        hand-over    a LoopResult: the raw tool results (facts for synthesis
                     and the evaluator), the trace, and the stop reason.

    Bad tool requests never crash the run: the dispatcher returns error text,
    the model sees it and can correct itself, and the call counts as a step.

Testing: the model client is injected, so tests use a scripted fake and need
no API key. Real runs build a client with ``make_client``.

AI DISCLOSURE: # AUTHOR:`` are theirs to review and reword.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from finagent import config
from finagent.agent.dispatcher import dispatch, render_for_model
from finagent.contracts import LoopResult, StopReason, TraceStep
from finagent.tools.company_profile import get_company_profile
from finagent.tools.tool_specs import TOOL_FUNCTIONS, TOOL_SPECS

# AUTHOR: review. This prompt only gathers data; a later stage writes the
# analysis, so the model is told not to.
SYSTEM_PROMPT = """\
You are the data-gathering stage of a stock research agent. Your only job is \
to collect facts with the tools provided. A later stage writes the analysis, \
so do not write one.

Rules:
- Research only {ticker}. Always call tools with exactly that ticker.
- The company profile is already fetched and shown below. Do not ask for it.
- Choose the tools that matter for this company. Use each tool at most once \
unless it failed.
- Never guess or invent numbers. If a tool fails, you may retry once, then \
move on.
- When you have enough data, or tools keep failing, reply with one short \
sentence and call no more tools.
"""


def make_client() -> Any:
    """Build the real model client (OpenRouter via the ``openai`` SDK).

    Imported here, not at the top of the file, so tests and offline work
    never need the SDK or an API key. A missing key fails now, with a
    message naming it, instead of mid-run.
    """
    from openai import OpenAI

    return OpenAI(
        api_key=config.require_env(config.OPENROUTER_KEY_NAME),
        base_url=config.BASE_URL,
    )


def _pin_ticker(
    functions: dict[str, Callable[..., Any]], ticker: str
) -> dict[str, Callable[..., Any]]:
    """Wrap every tool so it only accepts this run's ticker.

    A model can wander ("let me also check MSFT"). Raising ValueError here
    reuses the dispatcher's normal path: the model gets an error message
    and the tool is never called.
    """

    def pin(name: str, function: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(**kwargs: Any) -> Any:
            if kwargs.get("ticker") != ticker:
                raise ValueError(
                    f"this run analyzes {ticker}; call {name} with "
                    f"ticker='{ticker}'"
                )
            return function(**kwargs)

        return wrapper

    return {name: pin(name, function) for name, function in functions.items()}


def _trace_step(step: int, tool: str, arguments: dict[str, Any],
                ok: bool, error: str | None) -> TraceStep:
    return {"step": step, "tool": tool, "arguments": arguments,
            "ok": ok, "error": error}


def _profile_to_dict(profile: Any) -> dict[str, Any]:
    """Profile dataclass -> plain JSON-safe dict (dates become strings)."""
    return json.loads(json.dumps(asdict(profile), default=str))


def _assistant_message(message: Any, tool_calls: list[Any]) -> dict[str, Any]:
    """Rebuild the model's tool request as a message to send back to it."""
    return {
        "role": "assistant",
        "content": message.content,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name,
                             "arguments": call.function.arguments},
            }
            for call in tool_calls
        ],
    }


def run_research_loop(
    ticker: str,
    client: Any = None,
    *,
    profile_fn: Callable[[str], Any] = get_company_profile,
    functions: dict[str, Callable[..., Any]] | None = None,
    specs: list[dict[str, Any]] | None = None,
    max_steps: int = config.MAX_STEPS,
) -> LoopResult:
    """Gather data on ``ticker`` with a model-driven tool loop.

    Args:
        ticker: Stock symbol to research.
        client: Model client with ``chat.completions.create``. Built with
            ``make_client()`` when omitted. Tests pass a scripted fake.
        profile_fn: Pre-flight check. Injected so tests need no network.
        functions: Tool name -> function map (defaults to the real tools).
        specs: Tool menu shown to the model (defaults to the real menu).
        max_steps: Cap on model turns. The pre-flight does not count.

    Returns:
        A ``LoopResult``. Problems are reported through ``stop_reason``,
        ``fatal_error`` and the trace; this function does not raise for
        them. A missing API key is the exception: it raises at once.
    """
    ticker = ticker.strip().upper()
    client = make_client() if client is None else client
    functions = TOOL_FUNCTIONS if functions is None else functions
    specs = TOOL_SPECS if specs is None else specs

    trace: list[TraceStep] = []
    results: dict[str, Any] = {}

    def finish(stop_reason: StopReason, steps_used: int,
               profile: dict[str, Any] | None,
               fatal_error: str | None = None) -> LoopResult:
        return {
            "ticker": ticker,
            "stop_reason": stop_reason,
            "steps_used": steps_used,
            "profile": profile,
            "results": results,
            "trace": trace,
            "fatal_error": fatal_error,
        }

    # ---- pre-flight: no usable profile means nothing worth analysing ----
    try:
        profile = profile_fn(ticker)
    except Exception as exc:  # noqa: BLE001
        # Any failure here is a core-data failure: stop and report.
        message = f"{type(exc).__name__}: {exc}"
        trace.append(_trace_step(0, "get_company_profile",
                                 {"ticker": ticker}, False, message))
        return finish("fatal_error", 0, None, message)
    trace.append(_trace_step(0, "get_company_profile",
                             {"ticker": ticker}, True, None))
    profile_dict = _profile_to_dict(profile)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT.format(ticker=ticker)},
        {"role": "user",
         "content": f"Gather data to research {ticker}.\n\nCompany profile:\n"
                    f"{json.dumps(profile_dict, indent=2)}"},
    ]
    pinned = _pin_ticker(functions, ticker)

    # ---- the agent loop: the MODEL decides, Python enforces the cap ----
    stop_reason: StopReason | None = None
    fatal_message: str | None = None
    steps_used = 0

    for step in range(1, max_steps + 1):
        try:
            response = client.chat.completions.create(
                model=config.MODEL,
                messages=messages,
                tools=specs,
                max_tokens=config.MAX_TOKENS,
                temperature=config.TEMPERATURE,
            )
        except Exception as exc:  # noqa: BLE001
            # API down, no credit, bad key: the run cannot continue.
            stop_reason = "fatal_error"
            fatal_message = f"model call failed: {type(exc).__name__}: {exc}"
            break
        steps_used = step

        message = response.choices[0].message
        tool_calls = message.tool_calls or []
        if not tool_calls:  # the model says it has enough data
            stop_reason = "done"
            break

        messages.append(_assistant_message(message, tool_calls))
        for call in tool_calls:
            outcome = dispatch(call.function.name, call.function.arguments,
                               functions=pinned, specs=specs)
            trace.append(_trace_step(step, outcome.name, outcome.arguments,
                                     outcome.ok, outcome.error))
            if outcome.ok:
                results[outcome.name] = outcome.data  # latest success wins
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": render_for_model(outcome)})
            if outcome.fatal:
                stop_reason = "fatal_error"
                fatal_message = outcome.error
                break
        if stop_reason is not None:
            break

    if stop_reason is None:  # the for-loop ran out of steps
        stop_reason = "step_cap"
    return finish(stop_reason, steps_used, profile_dict, fatal_message)