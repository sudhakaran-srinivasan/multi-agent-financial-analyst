"""Dispatcher: turns the model's tool request into a real, safe Python call.

Agent Design: the model replies with "call tool X with these arguments"
(arguments arrive as a JSON *string*, written by an LLM, so they can be
wrong). The dispatcher is the switchboard between that request and our
functions. It never lets a bad request crash the run:

    unknown tool, bad arguments     -> ok=False, error text for the model
    tool raises an ordinary error   -> ok=False, error text for the model
    ticker is invalid / not a stock -> ok=False, fatal=True (stop the run)
    success                         -> ok=True, full structured data

The loop (next piece) reads ``fatal`` to stop, and sends ``error`` back to
the model so it can correct itself. Every outcome also goes in the trace.

Note: the loop, not the dispatcher, is responsible for making sure the
ticker matches the stock being analysed in this run.

AI DISCLOSURE: 
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

from finagent.tools.company_profile import ProfileError
from finagent.tools.tool_specs import TOOL_FUNCTIONS, TOOL_SPECS


@dataclass
class ToolOutcome:
    """What happened when we tried to run one tool request."""

    name: str
    arguments: dict[str, Any]
    ok: bool
    data: Any = None  # full structured result, kept for synthesis/evaluator
    error: str | None = None  # text the model sees when ok is False
    fatal: bool = False  # True -> the loop must stop the run


def _parse_arguments(raw: str | dict[str, Any]) -> dict[str, Any]:
    """Accept a JSON string (what models send) or an already-parsed dict."""
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"arguments are not valid JSON ({exc.msg})") from exc
    else:
        parsed = raw
    if not isinstance(parsed, dict):
        raise ValueError("arguments must be a JSON object")
    return parsed


def _check_arguments(arguments: dict[str, Any],
                     schema: dict[str, Any]) -> dict[str, Any]:
    """Validate against the tool's menu entry; return cleaned arguments.

    Checks: required keys present, no unknown keys, strings non-empty.
    The ticker is trimmed and upper-cased so "nvda " works as "NVDA".
    """
    properties: dict[str, Any] = schema["properties"]
    for key in schema.get("required", []):
        if key not in arguments:
            raise ValueError(f"missing required argument '{key}'")

    cleaned: dict[str, Any] = {}
    for key, value in arguments.items():
        if key not in properties:
            raise ValueError(
                f"unexpected argument '{key}'; allowed: {sorted(properties)}"
            )
        if properties[key]["type"] == "string":
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"argument '{key}' must be a non-empty string")
            value = value.strip()
            if key == "ticker":
                value = value.upper()
        cleaned[key] = value
    return cleaned


def dispatch(
    name: str,
    raw_arguments: str | dict[str, Any],
    functions: dict[str, Callable[..., Any]] | None = None,
    specs: list[dict[str, Any]] | None = None,
) -> ToolOutcome:
    """Run one tool request from the model and report what happened.

    Args:
        name: Tool name the model asked for.
        raw_arguments: JSON string (or dict) of arguments.
        functions: Name -> function map. Defaults to the real tools; tests
            inject fakes so no network is needed.
        specs: The tool menu used to validate arguments.

    Returns:
        A ``ToolOutcome``. This function does not raise for tool problems.
    """
    functions = TOOL_FUNCTIONS if functions is None else functions
    specs = TOOL_SPECS if specs is None else specs
    schemas = {s["function"]["name"]: s["function"]["parameters"] for s in specs}

    if name not in functions or name not in schemas:
        return ToolOutcome(
            name, {}, False,
            error=f"unknown tool '{name}'. Available tools: {sorted(functions)}",
        )

    arguments: dict[str, Any] = {}
    try:
        arguments = _parse_arguments(raw_arguments)
        cleaned = _check_arguments(arguments, schemas[name])
    except ValueError as exc:
        return ToolOutcome(name, arguments, False,
                           error=f"bad arguments for {name}: {exc}")

    try:
        data = functions[name](**cleaned)
    except ProfileError as exc:
        # Core-data failure: nothing to analyse. Stop the whole run.
        return ToolOutcome(name, cleaned, False,
                           error=f"{type(exc).__name__}: {exc}", fatal=True)
    except Exception as exc:  # noqa: BLE001
        # Deliberately broad: one failing tool (network, API, parsing) must
        # not kill the run. The model gets the error text; the trace logs it.
        return ToolOutcome(name, cleaned, False,
                           error=f"{type(exc).__name__}: {exc}")
    return ToolOutcome(name, cleaned, True, data=data)


def render_for_model(outcome: ToolOutcome) -> str:
    """Text the model sees for this outcome (a JSON string).

    Tool results are already compact (values, gaps, warnings), so no
    trimming is needed. ``outcome.data`` keeps the full structured copy.
    """
    if outcome.ok:
        return json.dumps(outcome.data, default=str)
    return json.dumps({"error": outcome.error})