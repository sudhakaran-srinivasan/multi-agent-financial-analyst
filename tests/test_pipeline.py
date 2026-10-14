"""Offline tests for finagent.pipeline (no model, no network).

Definition of "this works":
    1. Stages run in order and each receives the previous stage's output.
    2. A fatal loop skips every later stage.
    3. A crash in reflection or synthesis is recorded and the run continues
       with what it has (partial results), never losing the loop data.
    4. A setup problem (missing key) still raises.
    5. analyze=False never calls the analyst.
"""
import pytest

from finagent.pipeline import STAGE_REFLECT, STAGE_SYNTHESIZE, run_pipeline
from finagent.samples import SAMPLE_LOOP_RESULT, SAMPLE_REFLECTION

FATAL = {**SAMPLE_LOOP_RESULT, "stop_reason": "fatal_error",
         "fatal_error": "InvalidTickerError: no such ticker"}
UPDATED = {**SAMPLE_LOOP_RESULT, "steps_used": 99}
OUTCOME = object()  # the pipeline only passes it through


def ok_reflector(result):
    return UPDATED, SAMPLE_REFLECTION


def boom(*args, **kwargs):
    raise ValueError("kaboom")


def test_stages_chain_in_order_and_pass_results_forward():
    seen = {}

    def analyst(result, reflection=None):
        seen.update(result=result, reflection=reflection)
        return OUTCOME

    run = run_pipeline("NVDA", runner=lambda t: SAMPLE_LOOP_RESULT,
                       reflector=ok_reflector, analyst=analyst)
    assert run.loop is UPDATED  # reflection's updated result replaces the loop's
    assert run.reflection is SAMPLE_REFLECTION and run.outcome is OUTCOME
    assert seen["result"] is UPDATED and seen["reflection"] is SAMPLE_REFLECTION
    assert run.stage_errors == [] and not run.fatal


def test_a_fatal_loop_skips_reflection_and_analysis():
    run = run_pipeline("ZZZZZZ", runner=lambda t: FATAL,
                       reflector=boom, analyst=boom)
    assert run.fatal and run.reflection is None and run.outcome is None
    assert run.stage_errors == []  # skipped, not failed


def test_a_reflection_crash_is_recorded_and_synthesis_still_runs():
    got = {}

    def analyst(result, reflection=None):
        got.update(result=result, reflection=reflection)
        return OUTCOME

    run = run_pipeline("NVDA", runner=lambda t: SAMPLE_LOOP_RESULT,
                       reflector=boom, analyst=analyst)
    assert [e.stage for e in run.stage_errors] == [STAGE_REFLECT]
    assert "ValueError: kaboom" in run.stage_errors[0].error
    assert run.loop is SAMPLE_LOOP_RESULT and run.reflection is None
    assert got["result"] is SAMPLE_LOOP_RESULT and got["reflection"] is None
    assert run.outcome is OUTCOME


def test_a_synthesis_crash_keeps_the_loop_data():
    run = run_pipeline("NVDA", runner=lambda t: SAMPLE_LOOP_RESULT,
                       reflector=ok_reflector, analyst=boom)
    assert [e.stage for e in run.stage_errors] == [STAGE_SYNTHESIZE]
    assert run.outcome is None
    assert run.loop is UPDATED and run.reflection is SAMPLE_REFLECTION


def test_analyze_false_never_calls_the_analyst():
    run = run_pipeline("NVDA", analyze=False, runner=lambda t: SAMPLE_LOOP_RESULT,
                       reflector=ok_reflector, analyst=boom)
    assert run.outcome is None and run.stage_errors == []


def test_a_setup_problem_raises():
    def no_key(ticker):
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    with pytest.raises(RuntimeError, match="not set"):
        run_pipeline("NVDA", runner=no_key)


def test_the_ticker_comes_from_the_loop_result():
    run = run_pipeline("nvda", analyze=False,
                       runner=lambda t: SAMPLE_LOOP_RESULT, reflector=ok_reflector)
    assert run.ticker == "NVDA"


# ---- memory stage -------------------------------------------------------------
class _Outcome:
    """Just enough of a SynthesisOutcome for the memory stage."""

    def __init__(self, stance="neutral", fallback=None):
        from finagent.samples import SAMPLE_ANALYSIS

        self.analysis = {**SAMPLE_ANALYSIS, "stance": stance}
        self.signals = []
        self.fallback_reason = fallback
        self.notes = []


def _run(tmp_path, outcome):
    return run_pipeline("NVDA", runner=lambda t: SAMPLE_LOOP_RESULT,
                        reflector=ok_reflector, analyst=lambda r, reflection=None: outcome,
                        memory_path=tmp_path / "m.jsonl")


def test_memory_is_off_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = run_pipeline("NVDA", runner=lambda t: SAMPLE_LOOP_RESULT,
                       reflector=ok_reflector, analyst=lambda r, reflection=None: _Outcome())
    assert not run.saved and not (tmp_path / "data").exists()


def test_first_run_saves_and_second_run_reports_what_changed(tmp_path):
    first = _run(tmp_path, _Outcome("neutral"))
    assert first.saved and first.prior is None and first.changes is None
    second = _run(tmp_path, _Outcome("positive"))
    assert second.prior is not None
    assert second.changes.stance == ("neutral", "positive")
    assert len((tmp_path / "m.jsonl").read_text().splitlines()) == 2


def test_a_fallback_analysis_is_not_remembered(tmp_path):
    run = _run(tmp_path, _Outcome(fallback="model reply rejected"))
    assert not run.saved and not (tmp_path / "m.jsonl").exists()


def test_a_memory_failure_is_recorded_and_keeps_the_analysis(tmp_path):
    blocked = tmp_path / "file"
    blocked.write_text("x")  # a FILE where the folder should be: saving fails
    outcome = _Outcome()
    run = run_pipeline("NVDA", runner=lambda t: SAMPLE_LOOP_RESULT,
                       reflector=ok_reflector, analyst=lambda r, reflection=None: outcome,
                       memory_path=blocked / "m.jsonl")
    assert run.outcome is outcome and not run.saved
    assert [e.stage for e in run.stage_errors] == ["memory"]