"""Offline tests for finagent.agent.reflection (no model, scripted tool fakes).

Definition of "this works":
    1. Nothing fixable -> no tool is called and the result is unchanged.
    2. A failed planned tool is retried once; a recovery is added to results.
    3. A tool that never ran (step cap) is run once.
    4. A ``fetch_failed`` gap on a "successful" tool triggers a retry.
    5. Permanent gaps (missing field, not meaningful, thin history) are NEVER
       retried, only reported.
    6. At most one attempt per tool, even if it keeps failing.
    7. The input is not modified; retry calls are appended to the trace at
       step = steps_used + 1; steps_used does not change (no model turns).
    8. A fatal run is left alone.
"""

import copy

from finagent.agent.reflection import reflect

PLAN = {"ticker": "NVDA", "steps": [
    {"tool": "get_fundamentals", "reason": "r"},
    {"tool": "get_earnings_history", "reason": "r"},
    {"tool": "get_news_insights", "reason": "r"}], "priority_note": "n"}

FUND = {"ticker": "NVDA", "gaps": [], "warnings": []}
EARN = {"ticker": "NVDA", "summary": {"n": 4}, "gaps": [], "warnings": []}


def step(n, tool, ok=True, error=None):
    return {"step": n, "tool": tool, "arguments": {"ticker": "NVDA"},
            "ok": ok, "error": error}


def run_result(results, *, trace=None, stop_reason="done", steps_used=3, plan=PLAN):
    return {"ticker": "NVDA", "stop_reason": stop_reason, "steps_used": steps_used,
            "profile": {"ticker": "NVDA"}, "plan": plan, "results": results,
            "trace": trace or [], "fatal_error": None}


COMPLETE = {"get_fundamentals": FUND, "get_earnings_history": EARN,
            "get_news_insights": []}


class Tools:
    """Fake tools that count calls, so tests can see what reflection ran."""

    def __init__(self, **behaviour):
        self.calls: list[tuple[str, str]] = []
        self.behaviour = behaviour

    def functions(self):
        def make(name):
            def run(ticker):
                self.calls.append((name, ticker))
                action = self.behaviour.get(name, lambda: {"ticker": ticker})
                return action()
            return run
        return {name: make(name) for name in
                ("get_fundamentals", "get_earnings_history", "get_news_insights")}


def boom(message):
    def raiser():
        raise RuntimeError(message)
    return raiser


def test_nothing_fixable_calls_no_tool_and_changes_nothing():
    tools = Tools()
    result = run_result(COMPLETE)
    updated, reflection = reflect(result, functions=tools.functions())
    assert tools.calls == []
    assert updated["results"] == COMPLETE and updated["trace"] == []
    assert reflection["retried"] == []
    assert reflection["summary"].startswith("Nothing to retry")


def test_a_failed_tool_is_retried_once_and_recovers():
    tools = Tools(get_news_insights=list)
    results = {k: v for k, v in COMPLETE.items() if k != "get_news_insights"}
    trace = [step(1, "get_news_insights", ok=False, error="RuntimeError: timeout")]
    result = run_result(results, trace=trace)
    updated, reflection = reflect(result, functions=tools.functions())

    assert tools.calls == [("get_news_insights", "NVDA")]
    assert "get_news_insights" in updated["results"]
    assert reflection["retried"] == [{"tool": "get_news_insights", "why": "failed",
                                      "outcome": "recovered", "error": None}]
    assert "1 tool retried: 1 recovered, 0 still failing" in reflection["summary"]
    gaps = reflection["remaining_gaps"]
    assert not any("no result" in gap for gap in gaps)  # the failure is gone
    # The retry worked but found no news: that is a normal, permanent gap.
    assert "get_news_insights: no news insights available" in gaps


def test_a_tool_that_keeps_failing_is_tried_only_once():
    tools = Tools(get_news_insights=boom("NewsAPI down"))
    results = {k: v for k, v in COMPLETE.items() if k != "get_news_insights"}
    trace = [step(1, "get_news_insights", ok=False, error="RuntimeError: timeout")]
    updated, reflection = reflect(run_result(results, trace=trace),
                                  functions=tools.functions())

    assert len(tools.calls) == 1  # one attempt, no loop
    assert "get_news_insights" not in updated["results"]
    record = reflection["retried"][0]
    assert record["outcome"] == "still_failing" and "NewsAPI down" in record["error"]
    assert any("get_news_insights: no result" in g for g in reflection["remaining_gaps"])


def test_a_tool_never_called_because_of_the_step_cap_is_run_once():
    tools = Tools()
    results = {"get_fundamentals": FUND}
    result = run_result(results, stop_reason="step_cap", steps_used=8)
    updated, reflection = reflect(result, functions=tools.functions())

    assert [c[0] for c in tools.calls] == ["get_earnings_history", "get_news_insights"]
    assert {r["why"] for r in reflection["retried"]} == {"never_called"}
    assert set(updated["results"]) == set(COMPLETE)


def test_a_fetch_failed_gap_on_a_successful_tool_is_retried():
    bad = {**EARN, "summary": None, "gaps": [
        {"field": "earnings_history", "reason": "fetch_failed"}]}
    tools = Tools(get_earnings_history=lambda: EARN)
    result = run_result({**COMPLETE, "get_earnings_history": bad})
    updated, reflection = reflect(result, functions=tools.functions())

    assert tools.calls == [("get_earnings_history", "NVDA")]
    assert reflection["retried"][0]["why"] == "fetch_failed"
    assert reflection["retried"][0]["outcome"] == "recovered"
    assert updated["results"]["get_earnings_history"] == EARN


def test_a_retry_that_returns_fetch_failed_again_is_still_failing():
    bad = {**EARN, "summary": None, "gaps": [
        {"field": "earnings_history", "reason": "fetch_failed"}]}
    tools = Tools(get_earnings_history=lambda: bad)
    _, reflection = reflect(run_result({**COMPLETE, "get_earnings_history": bad}),
                            functions=tools.functions())
    assert len(tools.calls) == 1
    assert reflection["retried"][0]["outcome"] == "still_failing"
    assert any("fetch_failed" in g for g in reflection["remaining_gaps"])


def test_permanent_gaps_are_reported_but_never_retried():
    """AADX-like: a loss-maker with missing fields and a thin history."""
    fund = {**FUND, "gaps": [
        {"field": "trailingPE", "reason": "negative_earnings"},
        {"field": "enterpriseToEbitda", "reason": "not_meaningful"},
        {"field": "returnOnEquity", "reason": "missing"},
        {"field": "earningsGrowth", "reason": "missing"}]}
    earn = {**EARN, "summary": {"n": 1}, "gaps": [
        {"field": "FY2026Q2.surprise_pct", "reason": "near_zero_estimate"}],
        "warnings": ["thin history: only 1 classifiable quarter(s) (expected 4)"]}
    tools = Tools()
    result = run_result({"get_fundamentals": fund, "get_earnings_history": earn,
                         "get_news_insights": []})
    _, reflection = reflect(result, functions=tools.functions())

    assert tools.calls == []  # a retry could not change any of these
    assert reflection["retried"] == []
    assert "get_fundamentals: returnOnEquity missing" in reflection["remaining_gaps"]
    assert any("thin history" in g for g in reflection["remaining_gaps"])
    assert "gaps remain" in reflection["summary"]


def test_the_input_is_not_modified_and_the_trace_records_the_fixup():
    tools = Tools()
    results = {k: v for k, v in COMPLETE.items() if k != "get_news_insights"}
    trace = [step(1, "get_fundamentals")]
    result = run_result(results, trace=trace, steps_used=3)
    before = copy.deepcopy(result)
    updated, _ = reflect(result, functions=tools.functions())

    assert result == before  # untouched
    assert updated is not result
    assert updated["steps_used"] == 3  # reflection spends no model steps
    fixup = updated["trace"][-1]
    assert fixup["tool"] == "get_news_insights" and fixup["step"] == 4
    assert fixup["ok"] is True and len(updated["trace"]) == 2


def test_retry_uses_the_runs_ticker():
    tools = Tools()
    results = {"get_fundamentals": FUND, "get_earnings_history": EARN}
    reflect(run_result(results), functions=tools.functions())
    assert tools.calls == [("get_news_insights", "NVDA")]


def test_a_fatal_run_is_left_alone():
    tools = Tools()
    fatal = run_result({}, stop_reason="fatal_error")
    fatal["fatal_error"] = "NotAStockError: ETF"
    updated, reflection = reflect(fatal, functions=tools.functions())
    assert updated is fatal and tools.calls == []
    assert reflection["retried"] == []
    assert "fatal" in reflection["summary"]


def test_only_planned_tools_are_considered():
    plan = {**PLAN, "steps": [{"tool": "get_fundamentals", "reason": "r"}]}
    tools = Tools()
    result = run_result({"get_fundamentals": FUND}, plan=plan)
    reflect(result, functions=tools.functions())
    assert tools.calls == []  # earnings and news were never planned


def test_without_a_plan_every_available_tool_is_considered():
    tools = Tools()
    result = run_result({"get_fundamentals": FUND}, plan=None)
    reflect(result, functions=tools.functions())
    assert [c[0] for c in tools.calls] == ["get_earnings_history", "get_news_insights"]


def test_summary_grammar_for_one_gap_and_many():
    from finagent.agent.reflection import _summary

    assert _summary([], ["a"]) == "Nothing to retry; 1 gap remains (retrying cannot fix them)."
    assert "2 gaps remain" in _summary([], ["a", "b"])