"""Offline tests for finagent.agent.planner, using a scripted fake model.

Definition of "this works":
    1. A valid model plan is used as written (order and reasons kept).
    2. Unknown tools, bad JSON, empty plans and model failures fall back to
       the default plan and say why. The planner never raises.
    3. Core tools (fundamentals, earnings) are always in the final plan.
    4. Duplicate tools are removed; JSON inside a code fence still parses.
    5. The default plan depends on the profile (profit, earnings timing).
    6. The prompt sent to the model names the real tools and the profile.
"""

import json
from types import SimpleNamespace as NS

import pytest

from finagent.agent.planner import (
    PlanError,
    default_plan,
    parse_plan,
    plan_research,
)
from finagent.tools.tool_specs import TOOL_SPECS

VALID = [s["function"]["name"] for s in TOOL_SPECS]
NVDA_PROFILE = {"ticker": "NVDA", "name": "NVIDIA Corporation",
                "size_bucket": "mega", "is_profitable": True,
                "days_to_earnings": 12}
BIOTECH_PROFILE = {"ticker": "TBIO", "name": "Tiny Bio", "size_bucket": "micro",
                   "is_profitable": False, "days_to_earnings": None}


def reply(text):
    return NS(choices=[NS(message=NS(content=text))])


class ScriptedClient:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def plan_json(steps, note="Weigh the beat record."):
    return json.dumps({"steps": steps, "priority_note": note})


def step(tool, reason="because"):
    return {"tool": tool, "reason": reason}


def tools_of(plan):
    return [s["tool"] for s in plan["steps"]]


def test_a_valid_model_plan_is_used_as_written():
    text = plan_json([step("get_earnings_history", "earnings in 12 days"),
                      step("get_fundamentals", "check valuation"),
                      step("get_news_insights", "news")])
    outcome = plan_research("NVDA", NVDA_PROFILE, ScriptedClient(reply(text)))
    assert outcome.fallback_reason is None
    assert tools_of(outcome.plan) == [
        "get_earnings_history", "get_fundamentals", "get_news_insights"]
    assert outcome.plan["steps"][0]["reason"] == "earnings in 12 days"
    assert outcome.plan["priority_note"] == "Weigh the beat record."


def test_json_inside_a_code_fence_still_parses():
    text = "```json\n" + plan_json([step("get_fundamentals"),
                                    step("get_earnings_history")]) + "\n```"
    outcome = plan_research("NVDA", NVDA_PROFILE, ScriptedClient(reply(text)))
    assert outcome.fallback_reason is None


def test_missing_core_tools_are_added_by_python():
    text = plan_json([step("get_news_insights", "only news")])
    outcome = plan_research("NVDA", NVDA_PROFILE, ScriptedClient(reply(text)))
    assert outcome.fallback_reason is None
    assert tools_of(outcome.plan) == [
        "get_news_insights", "get_fundamentals", "get_earnings_history"]


def test_duplicate_tools_are_dropped():
    text = plan_json([step("get_fundamentals", "first"),
                      step("get_fundamentals", "again"),
                      step("get_earnings_history")])
    plan = plan_research("NVDA", NVDA_PROFILE, ScriptedClient(reply(text))).plan
    assert tools_of(plan).count("get_fundamentals") == 1
    assert plan["steps"][0]["reason"] == "first"


@pytest.mark.parametrize("bad_reply, expected", [
    (plan_json([step("get_dividends")]), "unknown tool"),
    ("I think you should look at fundamentals.", "no JSON"),
    ("{not valid json}", "invalid JSON"),
    (plan_json([]), "no steps"),
    (plan_json([{"tool": "get_fundamentals", "reason": " "}]), "no reason"),
    (json.dumps({"steps": [step("get_fundamentals")]}), "priority_note"),
    (json.dumps({"steps": "all of them", "priority_note": "x"}), "steps"),
])
def test_unusable_plans_fall_back_with_a_reason(bad_reply, expected):
    outcome = plan_research("NVDA", NVDA_PROFILE,
                            ScriptedClient(reply(bad_reply)))
    assert outcome.plan == default_plan("NVDA", NVDA_PROFILE)
    assert expected in outcome.fallback_reason


def test_model_failure_falls_back_and_does_not_raise():
    client = ScriptedClient(RuntimeError("402 payment required"))
    outcome = plan_research("NVDA", NVDA_PROFILE, client)
    assert outcome.plan == default_plan("NVDA", NVDA_PROFILE)
    assert "402" in outcome.fallback_reason


def test_empty_model_reply_falls_back():
    outcome = plan_research("NVDA", NVDA_PROFILE,
                            ScriptedClient(reply(None)))
    assert outcome.fallback_reason is not None


def test_default_plan_is_always_valid_and_has_core_tools():
    for profile in (NVDA_PROFILE, BIOTECH_PROFILE, None, {}):
        plan = default_plan("X", profile)
        assert set(tools_of(plan)) <= set(VALID)
        assert {"get_fundamentals", "get_earnings_history"} <= set(tools_of(plan))
        assert plan["priority_note"]


def test_default_plan_changes_with_the_company():
    nvda = default_plan("NVDA", NVDA_PROFILE)
    biotech = default_plan("TBIO", BIOTECH_PROFILE)
    assert "12 days" in nvda["steps"][1]["reason"]
    assert "earnings are close" in nvda["priority_note"].lower()
    assert "not profitable" in biotech["steps"][0]["reason"]
    assert "balance sheet" in biotech["priority_note"]


def test_prompt_names_the_real_tools_and_the_profile():
    client = ScriptedClient(reply(plan_json(
        [step("get_fundamentals"), step("get_earnings_history")])))
    plan_research("NVDA", NVDA_PROFILE, client)
    system, user = client.requests[0]["messages"]
    assert all(name in system["content"] for name in VALID)
    assert "NVDA" in system["content"]
    assert "NVIDIA Corporation" in user["content"]
    assert "tools" not in client.requests[0]  # the planner calls no tools


def test_parse_plan_raises_plan_error_directly():
    with pytest.raises(PlanError):
        parse_plan("nothing here", "NVDA", VALID, default_plan("NVDA", {}))