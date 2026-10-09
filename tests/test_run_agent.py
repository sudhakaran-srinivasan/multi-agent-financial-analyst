"""Offline tests for finagent.run_agent (no model, no network).

Definition of "this works":
    1. The report shows the stop reason, every trace step and every result.
    2. Errors and the fatal message are visible, not hidden.
    3. main() returns the right exit code: 0 finished, 1 fatal, 2 setup problem.
    4. --json prints valid JSON.
    5. Importing the module runs nothing (the main guard).
"""

import json

from finagent.run_agent import format_report, main
from finagent.samples import SAMPLE_LOOP_RESULT

FATAL_RESULT = {
    "ticker": "ZZZZZZ", "stop_reason": "fatal_error", "steps_used": 0,
    "profile": None, "results": {},
    "trace": [{"step": 0, "tool": "get_company_profile",
               "arguments": {"ticker": "ZZZZZZ"}, "ok": False,
               "error": "InvalidTickerError: no such ticker"}],
    "fatal_error": "InvalidTickerError: no such ticker",
}


def test_report_shows_stop_reason_trace_and_results():
    text = format_report(SAMPLE_LOOP_RESULT)
    assert "NVDA" in text and "stop: done" in text and "model steps: 4" in text
    for tool in ("get_company_profile", "get_fundamentals",
                 "get_earnings_history", "get_news_insights"):
        assert tool in text
    assert "ERROR: unknown tool 'get_dividends'" in text
    assert "NVIDIA Corporation" in text and "mega-cap" in text
    assert "0 items" in text  # the empty news list is visible, not hidden


def test_report_shows_gaps_from_a_result():
    result = {**SAMPLE_LOOP_RESULT, "results": {"get_fundamentals": {
        "gaps": [{"field": "pegRatio", "reason": "missing"}],
        "warnings": ["P/E mismatch"]}}}
    text = format_report(result)
    assert "gaps: 1 [pegRatio: missing]" in text and "warnings: 1" in text


def test_report_for_a_failed_preflight_shows_the_fatal_message():
    text = format_report(FATAL_RESULT)
    assert "stop: fatal_error" in text
    assert "FATAL: InvalidTickerError: no such ticker" in text
    assert "Profile: none" in text and "(none)" in text


def test_exit_code_zero_for_a_finished_run(capsys):
    assert main(["nvda"], runner=lambda t: SAMPLE_LOOP_RESULT) == 0
    assert "stop: done" in capsys.readouterr().out


def test_exit_code_one_for_a_fatal_run(capsys):
    assert main(["ZZZZZZ"], runner=lambda t: FATAL_RESULT) == 1


def test_exit_code_two_for_a_setup_problem(capsys):
    def no_key(ticker):
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    assert main(["NVDA"], runner=no_key) == 2
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err


def test_json_flag_prints_valid_json(capsys):
    main(["NVDA", "--json"], runner=lambda t: SAMPLE_LOOP_RESULT)
    assert json.loads(capsys.readouterr().out)["ticker"] == "NVDA"


def test_the_ticker_is_passed_to_the_runner(capsys):
    seen = []
    main(["KO"], runner=lambda t: seen.append(t) or SAMPLE_LOOP_RESULT)
    assert seen == ["KO"]