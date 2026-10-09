"""Offline tests for finagent.agent.dispatcher (fake tools, no network).

Definition of "this works":
    1. A valid request runs the tool and keeps the full data.
    2. The ticker is cleaned ("  nvda " -> "NVDA") before the tool sees it.
    3. Every kind of bad request returns ok=False with a useful message,
       and never raises or marks the run fatal.
    4. A tool that crashes is not fatal; an invalid ticker IS fatal.
    5. What the model sees is valid JSON, for both success and error.
"""

import json

from finagent.agent.dispatcher import dispatch, render_for_model
from finagent.tools.company_profile import InvalidTickerError, NotAStockError


def _fake_tools(**overrides):
    """Fake tool map. Each fake records the ticker it was called with."""
    calls: list[str] = []

    def fundamentals(ticker):
        calls.append(ticker)
        return {"ticker": ticker, "gaps": []}

    tools = {"get_fundamentals": fundamentals}
    tools.update(overrides)
    return tools, calls


def test_valid_request_runs_tool_and_keeps_data():
    tools, calls = _fake_tools()
    out = dispatch("get_fundamentals", '{"ticker": "NVDA"}', functions=tools)
    assert out.ok and not out.fatal and out.error is None
    assert out.data == {"ticker": "NVDA", "gaps": []}
    assert out.arguments == {"ticker": "NVDA"}
    assert calls == ["NVDA"]


def test_ticker_is_trimmed_and_uppercased():
    tools, calls = _fake_tools()
    dispatch("get_fundamentals", '{"ticker": "  nvda "}', functions=tools)
    assert calls == ["NVDA"]


def test_dict_arguments_are_accepted_too():
    tools, _ = _fake_tools()
    out = dispatch("get_fundamentals", {"ticker": "KO"}, functions=tools)
    assert out.ok


def test_unknown_tool_lists_available_tools():
    tools, _ = _fake_tools()
    out = dispatch("get_dividends", '{"ticker": "KO"}', functions=tools)
    assert not out.ok and not out.fatal
    assert "get_dividends" in out.error and "get_fundamentals" in out.error


def test_invalid_json_is_reported():
    tools, _ = _fake_tools()
    out = dispatch("get_fundamentals", "{ticker: NVDA", functions=tools)
    assert not out.ok and not out.fatal
    assert "not valid JSON" in out.error


def test_arguments_must_be_an_object():
    tools, _ = _fake_tools()
    out = dispatch("get_fundamentals", '["NVDA"]', functions=tools)
    assert not out.ok and "JSON object" in out.error


def test_missing_ticker_is_reported():
    tools, _ = _fake_tools()
    out = dispatch("get_fundamentals", "{}", functions=tools)
    assert not out.ok and "missing required argument 'ticker'" in out.error


def test_unexpected_argument_is_rejected():
    tools, calls = _fake_tools()
    out = dispatch("get_fundamentals",
                   '{"ticker": "NVDA", "limit": 99}', functions=tools)
    assert not out.ok and "unexpected argument 'limit'" in out.error
    assert calls == []  # the tool was never called


def test_wrong_type_and_blank_ticker_are_rejected():
    tools, calls = _fake_tools()
    for bad in ('{"ticker": 123}', '{"ticker": "   "}', '{"ticker": null}'):
        out = dispatch("get_fundamentals", bad, functions=tools)
        assert not out.ok and "non-empty string" in out.error
    assert calls == []


def test_ordinary_tool_error_is_not_fatal():
    def boom(ticker):
        raise ValueError("upstream timeout")

    tools, _ = _fake_tools(get_fundamentals=boom)
    out = dispatch("get_fundamentals", '{"ticker": "NVDA"}', functions=tools)
    assert not out.ok and not out.fatal
    assert "ValueError" in out.error and "upstream timeout" in out.error


def test_invalid_ticker_and_not_a_stock_are_fatal():
    for exc in (InvalidTickerError("no such ticker"),
                NotAStockError("this is an ETF")):
        def raiser(ticker, exc=exc):
            raise exc

        tools, _ = _fake_tools(get_fundamentals=raiser)
        out = dispatch("get_fundamentals", '{"ticker": "ZZZZ"}', functions=tools)
        assert not out.ok and out.fatal
        assert type(exc).__name__ in out.error


def test_render_for_model_is_valid_json_for_success_and_error():
    tools, _ = _fake_tools()
    good = dispatch("get_fundamentals", '{"ticker": "NVDA"}', functions=tools)
    assert json.loads(render_for_model(good))["ticker"] == "NVDA"

    bad = dispatch("nope", "{}", functions=tools)
    assert "error" in json.loads(render_for_model(bad))