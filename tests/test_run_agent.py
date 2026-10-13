"""Offline tests for finagent.run_agent (no model, no network).

Definition of "this works":
    1. The report shows the stop reason, every trace step and every result.
    2. Errors and the fatal message are visible, not hidden.
    3. main() returns the right exit code: 0 finished, 1 fatal, 2 setup problem.
    4. --json prints valid JSON.
    5. Importing the module runs nothing (the main guard).
"""

import json

from finagent.agent.synthesis import DISCLAIMER, SynthesisOutcome
from finagent.run_agent import format_analysis, format_report, main
from finagent.samples import SAMPLE_LOOP_RESULT

FATAL_RESULT = {
    "ticker": "ZZZZZZ", "stop_reason": "fatal_error", "steps_used": 0,
    "profile": None, "plan": None, "results": {},
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


def test_report_shows_the_plan_and_its_emphasis():
    text = format_report(SAMPLE_LOOP_RESULT)
    assert "Plan:" in text and "1. get_fundamentals:" in text
    assert "Emphasis: Earnings are close" in text


def test_report_without_a_plan_has_no_plan_section():
    assert "Plan:" not in format_report(FATAL_RESULT)


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


# ---- --analyze ---------------------------------------------------------------
OUTCOME = SynthesisOutcome(
    analysis={
        "ticker": "NVDA", "stance": "bullish", "confidence": "medium",
        "thesis": "Forward P/E is half of trailing P/E.",
        "evidence_used": [{"claim": "Forward P/E", "value": 15.1,
                           "source": "get_fundamentals.valuation.forwardPE"}],
        "risks": ["Forecast may be missed."],
        "sources": ["Yahoo Finance via yfinance (fundamentals)"],
        "data_gaps": ["get_news_insights: no news insights available"],
    },
    fallback_reason=None,
    notes=["confidence lowered from high to medium: thin history"],
    signals=[{"name": "net_cash", "direction": "positive", "value": 1.0,
              "source": "get_fundamentals.derived.net_cash",
              "why": "cash exceeds debt (a net cash position)"}],
)


def test_analysis_text_shows_every_section_and_the_disclaimer():
    text = format_analysis(OUTCOME)
    assert "stance: bullish" in text and "confidence: medium" in text
    assert "net_cash" in text and "[+]" in text and "Value" in text
    assert "Forward P/E = 15.1  <get_fundamentals.valuation.forwardPE>" in text
    assert "Risks:" in text and "Data gaps:" in text and "Sources:" in text
    assert "confidence lowered from high to medium" in text
    assert text.rstrip().endswith(DISCLAIMER)


def test_analysis_text_shows_a_fallback_reason():
    fallback = SynthesisOutcome(OUTCOME.analysis, "RuntimeError: 402", [], [])
    assert "model analysis not used (RuntimeError: 402)" in format_analysis(fallback)


def test_analyze_flag_runs_the_analyst_after_the_loop(capsys):
    seen = []
    code = main(["NVDA", "--analyze"], runner=lambda t: SAMPLE_LOOP_RESULT,
                analyst=lambda r, reflection=None: seen.append(r) or OUTCOME)
    out = capsys.readouterr().out
    assert code == 0 and seen == [SAMPLE_LOOP_RESULT]
    assert "stop: done" in out and "ANALYSIS" in out and DISCLAIMER in out


def test_without_the_flag_no_analysis_is_run(capsys):
    def boom(result, reflection=None):
        raise AssertionError("analyst must not run")

    assert main(["NVDA"], runner=lambda t: SAMPLE_LOOP_RESULT, analyst=boom) == 0
    assert "ANALYSIS" not in capsys.readouterr().out


def test_a_fatal_loop_skips_the_analysis(capsys):
    def boom(result, reflection=None):
        raise AssertionError("analyst must not run")

    code = main(["ZZZZZZ", "--analyze"], runner=lambda t: FATAL_RESULT, analyst=boom)
    assert code == 1


def test_analyze_with_json_wraps_loop_and_analysis(capsys):
    main(["NVDA", "--analyze", "--json"], runner=lambda t: SAMPLE_LOOP_RESULT,
         analyst=lambda r, reflection=None: OUTCOME)
    payload = json.loads(capsys.readouterr().out)
    assert payload["loop"]["ticker"] == "NVDA"
    assert payload["analysis"]["stance"] == "bullish"
    assert payload["disclaimer"] == DISCLAIMER and payload["signals"]


# ---- reflection in the entry point ------------------------------------------
def test_reflection_runs_after_the_loop_and_is_shown(capsys):
    from finagent.samples import SAMPLE_REFLECTION

    seen = []

    def reflector(result):
        seen.append(result)
        return result, SAMPLE_REFLECTION

    assert main(["NVDA"], runner=lambda t: SAMPLE_LOOP_RESULT,
                reflector=reflector) == 0
    out = capsys.readouterr().out
    assert seen == [SAMPLE_LOOP_RESULT]
    assert "Reflection: 1 tool retried: 0 recovered, 1 still failing" in out
    assert "retried get_news_insights [failed]: still_failing" in out


def test_the_analyst_receives_the_updated_result_and_the_reflection(capsys):
    from finagent.samples import SAMPLE_REFLECTION

    updated = {**SAMPLE_LOOP_RESULT, "steps_used": 4}
    got = {}

    def analyst(result, reflection=None):
        got.update(result=result, reflection=reflection)
        return OUTCOME

    main(["NVDA", "--analyze"], runner=lambda t: SAMPLE_LOOP_RESULT,
         reflector=lambda r: (updated, SAMPLE_REFLECTION), analyst=analyst)
    assert got["result"] is updated and got["reflection"] is SAMPLE_REFLECTION


def test_a_fatal_loop_skips_reflection(capsys):
    def boom(result):
        raise AssertionError("reflector must not run")

    assert main(["ZZZZZZ"], runner=lambda t: FATAL_RESULT, reflector=boom) == 1


def test_json_includes_the_reflection(capsys):
    from finagent.samples import SAMPLE_REFLECTION

    main(["NVDA", "--json"], runner=lambda t: SAMPLE_LOOP_RESULT,
         reflector=lambda r: (r, SAMPLE_REFLECTION))
    payload = json.loads(capsys.readouterr().out)
    assert payload["ticker"] == "NVDA"
    assert payload["reflection"]["retried"][0]["tool"] == "get_news_insights"


# ---- scorecard and stage errors ---------------------------------------------
def test_scorecard_is_a_table_with_value_read_and_why():
    from finagent.run_agent import format_scorecard

    signals = [
        {"name": "fcf_yield", "direction": "negative", "value": -0.066,
         "source": "x", "why": "burns cash"},
        {"name": "net_cash", "direction": "positive", "value": 2.4e10,
         "source": "x", "why": "cash exceeds debt"},
        {"name": "trend", "direction": "neutral", "value": None,
         "source": "x", "why": "no data"},
    ]
    text = format_scorecard(signals)
    assert "Signal" in text and "Value" in text and "Read" in text
    assert "-0.066  [-]  burns cash" in text
    assert "24.0B  [+]  cash exceeds debt" in text
    assert "n/a  [=]  no data" in text


def test_scorecard_with_no_signals_says_so():
    from finagent.run_agent import format_scorecard

    assert "no signals" in format_scorecard([])


def test_a_crashing_analyst_prints_a_stage_error_and_still_exits_zero(capsys):
    def boom(result, reflection=None):
        raise ValueError("kaboom")

    code = main(["NVDA", "--analyze"], runner=lambda t: SAMPLE_LOOP_RESULT,
                reflector=lambda r: (r, None), analyst=boom)
    out = capsys.readouterr().out
    assert code == 0
    assert "STAGE ERRORS" in out and "synthesize: ValueError: kaboom" in out


def test_json_lists_stage_errors(capsys):
    def boom(result):
        raise ValueError("kaboom")

    main(["NVDA", "--json"], runner=lambda t: SAMPLE_LOOP_RESULT, reflector=boom)
    payload = json.loads(capsys.readouterr().out)
    assert payload["stage_errors"] == [
        {"stage": "reflect", "error": "ValueError: kaboom"}]