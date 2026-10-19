"""The planner: decide WHAT to research before any data tool runs.

Agent Design (rubric: Agent Functions, "planning").
    Why a planner?  Without one, the tool loop picks tools with no stated
    reason, and nothing exists for reflection to check the run against. The
    plan is the agent's written intention: which questions matter for THIS
    company, which tool answers each, and what to weigh most.

    Who does what (the model proposes, Python checks):
        model   writes a Plan as JSON, tailored to the company profile
                (size, profitable or not, days to earnings).
        Python  parses it, rejects unknown tools, removes duplicates, and
                guarantees the core tools (fundamentals, earnings history)
                are present, because a stance cannot be judged without them.
        fallback  if the model call fails or the JSON is unusable, a
                rule-based default plan is used and the reason is recorded.
                The planner is NOT core data, so it never ends a run.

Testing: the model client is injected, so tests use a scripted fake.

AI use: see the AI note in README.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from finagent import config
from finagent.contracts import Plan, PlanStep
from finagent.tools.tool_specs import TOOL_SPECS

# AUTHOR: confirm. Without both of these there is nothing solid to call
# bullish or bearish, so Python adds them if the model leaves them out.
REQUIRED_TOOLS: tuple[str, ...] = ("get_fundamentals", "get_earnings_history")

# A plan this size covers every tool once; more would only be repetition.
_MAX_REASON_CHARS = 300
_NEAR_EARNINGS_DAYS = 60

# AUTHOR: review. The prompt decides how tailored the plans are.
PLANNER_PROMPT = """\
You are the planning stage of a stock research agent. Before any data is \
fetched, write a short research plan for {ticker}.

Choose which of these tools to use, in order, and say in one sentence why \
THIS company needs each one:
{tool_lines}

Rules:
- Use only the tool names listed above.
- Always include get_fundamentals and get_earnings_history (core data).
- Tailor the plan to the profile (size, profitability, upcoming earnings), \
not a generic checklist.
- priority_note: one sentence on what the plan emphasizes and why.

Reply with JSON only, no other text:
{{"steps": [{{"tool": "...", "reason": "..."}}], "priority_note": "..."}}
"""


class PlanError(ValueError):
    """The model's plan was unusable. Caught inside ``plan_research``."""


@dataclass
class PlannerOutcome:
    """A plan plus whether it is the model's or the fallback's.

    ``fallback_reason`` is None when the model's plan was used, otherwise
    the text explaining why the default plan took over (shown in the trace).
    """

    plan: Plan
    fallback_reason: str | None = None


# ---- the rule-based default plan -------------------------------------------
def default_plan(ticker: str, profile: dict[str, Any] | None) -> Plan:
    """A sensible plan from profile facts alone (no model, always works)."""
    profile = profile or {}
    size = profile.get("size_bucket")
    profitable = profile.get("is_profitable")
    days = profile.get("days_to_earnings")

    who = f"{size}-cap " if size else ""
    if profitable is False:
        fundamentals_reason = (
            f"{who}company is not profitable: P/E is not meaningful, so "
            "check cash, debt and growth")
    else:
        fundamentals_reason = (
            f"{who}company: check valuation, margins, growth and balance sheet")

    if days is not None and 0 <= days <= _NEAR_EARNINGS_DAYS:
        earnings_reason = f"earnings due in {days} days: weigh the beat record"
        note = "Earnings are close, so weigh the recent beat/miss record heavily."
    else:
        earnings_reason = "check whether the company usually beats estimates"
        note = ("Judge valuation against growth and financial health; "
                "earnings history is the supporting signal.")
    if profitable is False:
        note = "Not profitable: weigh balance sheet strength and growth over P/E."

    return {
        "ticker": ticker,
        "steps": [
            {"tool": "get_fundamentals", "reason": fundamentals_reason},
            {"tool": "get_earnings_history", "reason": earnings_reason},
            {"tool": "get_news_insights",
             "reason": "recent news and sentiment (may be empty: report as a gap)"},
        ],
        "priority_note": note,
    }


# ---- parsing and checking the model's plan ---------------------------------
def _extract_json(text: str) -> Any:
    """Parse JSON even if the model wrapped it in a code fence or prose."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise PlanError("no JSON object found in the reply")
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise PlanError(f"invalid JSON: {exc}") from exc


def parse_plan(text: str, ticker: str, valid_tools: list[str],
               fallback: Plan) -> Plan:
    """Turn the model's reply into a checked Plan, or raise ``PlanError``.

    Checks: right shape, no unknown tools, no duplicates, non-empty reasons.
    Then guarantees the core tools by appending any the model left out,
    using the fallback plan's reason for that tool.
    """
    data = _extract_json(text)
    if not isinstance(data, dict) or not isinstance(data.get("steps"), list):
        raise PlanError("plan must be an object with a 'steps' list")

    steps: list[PlanStep] = []
    seen: set[str] = set()
    for item in data["steps"]:
        if not isinstance(item, dict):
            raise PlanError("each step must be an object")
        tool, reason = item.get("tool"), item.get("reason")
        if tool not in valid_tools:
            raise PlanError(f"unknown tool {tool!r}; valid: {sorted(valid_tools)}")
        if not isinstance(reason, str) or not reason.strip():
            raise PlanError(f"step for {tool} has no reason")
        if tool in seen:
            continue  # a repeated tool adds nothing
        seen.add(tool)
        steps.append({"tool": tool, "reason": reason.strip()[:_MAX_REASON_CHARS]})
    if not steps:
        raise PlanError("plan has no steps")

    fallback_reasons = {s["tool"]: s["reason"] for s in fallback["steps"]}
    for tool in REQUIRED_TOOLS:
        if tool in valid_tools and tool not in seen:
            steps.append({"tool": tool, "reason": fallback_reasons.get(
                tool, "required core data")})

    note = data.get("priority_note")
    if not isinstance(note, str) or not note.strip():
        raise PlanError("missing priority_note")
    return {"ticker": ticker, "steps": steps,
            "priority_note": note.strip()[:_MAX_REASON_CHARS]}


# ---- the planner itself ----------------------------------------------------
def _tool_lines(specs: list[dict[str, Any]]) -> str:
    lines = []
    for spec in specs:
        function = spec["function"]
        first_sentence = function["description"].split(". ")[0].rstrip(".")
        lines.append(f"- {function['name']}: {first_sentence}.")
    return "\n".join(lines)


def plan_research(ticker: str, profile: dict[str, Any] | None, client: Any, *,
                  specs: list[dict[str, Any]] | None = None) -> PlannerOutcome:
    """Ask the model for a plan; fall back to the default plan on any problem.

    Never raises for model or parsing problems: the reason is returned in
    ``fallback_reason`` so the run can record it and carry on.
    """
    specs = TOOL_SPECS if specs is None else specs
    valid_tools = [spec["function"]["name"] for spec in specs]
    fallback = default_plan(ticker, profile)
    try:
        response = client.chat.completions.create(
            model=config.MODEL,
            messages=[
                {"role": "system", "content": PLANNER_PROMPT.format(
                    ticker=ticker, tool_lines=_tool_lines(specs))},
                {"role": "user", "content": "Company profile:\n"
                 + json.dumps(profile or {}, indent=2)},
            ],
            max_tokens=config.MAX_TOKENS,
            temperature=config.TEMPERATURE,
        )
        text = response.choices[0].message.content or ""
        plan = parse_plan(text, ticker, valid_tools, fallback)
    except Exception as exc:  # noqa: BLE001
        # API failure, empty reply or bad plan: degrade, never stop the run.
        return PlannerOutcome(fallback, f"{type(exc).__name__}: {exc}")
    return PlannerOutcome(plan, None)
