"""Offline tests for finagent.agent.loop, using a scripted fake model client.

Definition of "this works":
    1. The model's "no more tools" reply ends the run with stop_reason=done.
    2. Tool results are kept as structured data AND sent to the model as text.
    3. A run that never stops is ended by the step cap, not left to run on.
    4. A bad tool request is sent back to the model as an error and the run
       continues; it still counts as a step.
    5. A bad ticker or non-stock stops the run BEFORE any model call.
    6. A model asking about a different ticker is refused; the tool is not called.
    7. A model-API failure ends the run cleanly with fatal_error.
    8. The result always has every LoopResult key.
"""

import copy
import json
from types import SimpleNamespace as NS
from typing import get_type_hints

import pytest

from finagent.agent.loop import run_research_loop
from finagent.contracts import LoopResult
from finagent.tools.company_profile import InvalidTickerError, NotAStockError, Profile
from finagent.tools.tool_specs import TOOL_SPECS


# ---- test helpers ---------------------------------------------------------
def tool_call(call_id, name, arguments):
    """One tool request, shaped like the SDK's tool_call objects."""
    return NS(id=call_id,
              function=NS(name=name, arguments=json.dumps(arguments)))


def turn(*calls):
    """One model reply. No calls means 'I am done'."""
    return NS(choices=[NS(message=NS(content="ok", tool_calls=list(calls)))])


class ScriptedClient:
    """Plays back prepared replies and records what the loop sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests: list[dict] = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def fake_profile(ticker):
    return Profile(ticker=ticker, name="NVIDIA Corporation", quote_type="EQUITY")


def fake_tools():
    """Fake tools that record every call, so tests can see what ran."""
    seen: list[tuple[str, str]] = []

    def make(name, payload):
        def run(ticker):
            seen.append((name, ticker))
            return {"ticker": ticker, **payload}
        return run

    functions = {
        "get_fundamentals": make("get_fundamentals", {"gaps": []}),
        "get_earnings_history": make("get_earnings_history",
                                     {"summary": {"beats": 4}}),
        "get_news_insights": lambda ticker: [],
    }
    return functions, seen


def run(replies, **overrides):
    functions, seen = fake_tools()
    client = ScriptedClient(replies)
    kwargs = {"profile_fn": fake_profile, "functions": functions}
    kwargs.update(overrides)
    result = run_research_loop("nvda", client, **kwargs)
    return result, client, seen


# ---- the tests ------------------------------------------------------------
def test_model_with_no_tool_calls_ends_the_run_as_done():
    result, client, _ = run([turn()])
    assert result["stop_reason"] == "done"
    assert result["steps_used"] == 1
    assert result["results"] == {}
    assert result["fatal_error"] is None
    assert len(client.requests) == 1


def test_normal_run_keeps_results_and_feeds_them_back_to_the_model():
    replies = [
        turn(tool_call("c1", "get_fundamentals", {"ticker": "NVDA"}),
             tool_call("c2", "get_earnings_history", {"ticker": "NVDA"})),
        turn(),
    ]
    result, client, seen = run(replies)

    assert result["stop_reason"] == "done" and result["steps_used"] == 2
    assert set(result["results"]) == {"get_fundamentals", "get_earnings_history"}
    assert [t["tool"] for t in result["trace"]] == [
        "get_company_profile", "get_fundamentals", "get_earnings_history"]
    assert [t["step"] for t in result["trace"]] == [0, 1, 1]
    assert all(t["ok"] for t in result["trace"])
    assert seen == [("get_fundamentals", "NVDA"), ("get_earnings_history", "NVDA")]

    # The second model request contains the tool results as JSON text.
    tool_messages = [m for m in client.requests[1]["messages"]
                     if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["c1", "c2"]
    assert json.loads(tool_messages[0]["content"])["ticker"] == "NVDA"


def test_first_request_has_menu_profile_and_settings():
    _, client, _ = run([turn()])
    request = client.requests[0]
    assert request["tools"] == TOOL_SPECS
    names = [s["function"]["name"] for s in request["tools"]]
    assert "get_company_profile" not in names
    system, user = request["messages"][0], request["messages"][1]
    assert system["role"] == "system" and "NVDA" in system["content"]
    assert "NVIDIA Corporation" in user["content"]
    assert request["max_tokens"] > 0


def test_a_model_that_never_stops_is_ended_by_the_step_cap():
    always_asks = [turn(tool_call(f"c{i}", "get_fundamentals", {"ticker": "NVDA"}))
                   for i in range(10)]
    result, client, _ = run(always_asks, max_steps=3)
    assert result["stop_reason"] == "step_cap"
    assert result["steps_used"] == 3
    assert len(client.requests) == 3
    assert len(result["trace"]) == 1 + 3  # pre-flight + three calls


def test_unknown_tool_is_sent_back_as_an_error_and_the_run_continues():
    replies = [
        turn(tool_call("c1", "get_dividends", {"ticker": "NVDA"})),
        turn(tool_call("c2", "get_fundamentals", {"ticker": "NVDA"})),
        turn(),
    ]
    result, client, _ = run(replies)
    assert result["stop_reason"] == "done" and result["steps_used"] == 3
    failed = [t for t in result["trace"] if not t["ok"]]
    assert len(failed) == 1 and "unknown tool" in failed[0]["error"]
    assert "get_fundamentals" in result["results"]

    error_message = [m for m in client.requests[1]["messages"]
                     if m["role"] == "tool"][0]
    assert "unknown tool" in json.loads(error_message["content"])["error"]


def test_a_crashing_tool_is_not_fatal_and_leaves_no_result():
    def boom(ticker):
        raise RuntimeError("upstream down")

    functions, _ = fake_tools()
    functions["get_earnings_history"] = boom
    replies = [turn(tool_call("c1", "get_earnings_history", {"ticker": "NVDA"})),
               turn()]
    result, _, _ = run(replies, functions=functions)
    assert result["stop_reason"] == "done"
    assert "get_earnings_history" not in result["results"]
    assert "RuntimeError" in result["trace"][-1]["error"]


@pytest.mark.parametrize("error", [InvalidTickerError("no such ticker"),
                                   NotAStockError("this is an ETF")])
def test_bad_ticker_stops_before_any_model_call(error):
    def failing_profile(ticker):
        raise error

    result, client, _ = run([], profile_fn=failing_profile)
    assert result["stop_reason"] == "fatal_error"
    assert result["steps_used"] == 0 and result["profile"] is None
    assert type(error).__name__ in result["fatal_error"]
    assert client.requests == []  # zero model cost
    assert result["trace"][0]["tool"] == "get_company_profile"
    assert result["trace"][0]["ok"] is False


def test_fatal_error_from_a_tool_stops_the_run():
    def not_a_stock(ticker):
        raise NotAStockError("this is an ETF")

    functions, _ = fake_tools()
    functions["get_fundamentals"] = not_a_stock
    replies = [turn(tool_call("c1", "get_fundamentals", {"ticker": "NVDA"}))]
    result, client, _ = run(replies, functions=functions)
    assert result["stop_reason"] == "fatal_error"
    assert "NotAStockError" in result["fatal_error"]
    assert result["profile"] is not None
    assert len(client.requests) == 1


def test_model_asking_about_a_different_ticker_is_refused():
    replies = [turn(tool_call("c1", "get_fundamentals", {"ticker": "MSFT"})),
               turn()]
    result, _, seen = run(replies)
    assert seen == []  # the tool was never called
    failed = result["trace"][-1]
    assert not failed["ok"] and "this run analyzes NVDA" in failed["error"]


def test_model_api_failure_ends_the_run_cleanly():
    result, _, _ = run([RuntimeError("402 payment required")])
    assert result["stop_reason"] == "fatal_error"
    assert "model call failed" in result["fatal_error"]
    assert result["steps_used"] == 0
    assert result["profile"] is not None


def test_result_always_has_every_loop_result_key():
    done, _, _ = run([turn()])
    failed, _, _ = run([], profile_fn=lambda t: (_ for _ in ()).throw(
        InvalidTickerError("x")))
    expected = set(get_type_hints(LoopResult))
    assert set(done) == expected and set(failed) == expected
    json.dumps(done)  # must be JSON-serializable
    json.dumps(failed)