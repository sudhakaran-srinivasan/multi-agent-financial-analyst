"""Offline tests for finagent.memory (uses pytest's tmp_path, no real files).

Definition of "this works":
    1. save_run appends; earlier lines are never changed.
    2. load_prior returns the NEWEST record for that ticker only.
    3. A record older than the cutoff is ignored but stays on disk.
    4. No file, wrong ticker, damaged lines and future dates never crash.
    5. diff_runs reports only real changes, computed from stored fields.
"""
import json
from datetime import UTC, datetime, timedelta

from finagent.memory import (
    MAX_AGE_DAYS,
    RunDiff,
    diff_runs,
    format_diff,
    load_prior,
    record_from_outcome,
    save_run,
)
from finagent.samples import SAMPLE_ANALYSIS

NOW = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)


class FakeOutcome:
    analysis = SAMPLE_ANALYSIS
    signals = ({"name": "fcf_yield", "direction": "negative", "value": -0.07,
                "source": "x", "why": "y"},)


def make(ticker="NVDA", days_ago=1, stance="neutral", signals=None, gaps=None):
    return {
        "ticker": ticker,
        "saved_at": (NOW - timedelta(days=days_ago)).isoformat(),
        "stance": stance, "confidence": "medium", "thesis": "t",
        "signals": signals if signals is not None else [],
        "data_gaps": gaps or [],
    }


def test_save_appends_and_never_rewrites(tmp_path):
    path = tmp_path / "sub" / "memory.jsonl"  # folder is created if missing
    save_run(make(days_ago=3), path)
    first = path.read_text()
    save_run(make(days_ago=1), path)
    assert path.read_text().startswith(first)
    assert len(path.read_text().splitlines()) == 2


def test_load_prior_returns_the_newest_for_that_ticker(tmp_path):
    path = tmp_path / "m.jsonl"
    for rec in (make(days_ago=10, stance="negative"), make(days_ago=2, stance="positive"),
                make("AADX", days_ago=1, stance="negative")):
        save_run(rec, path)
    assert load_prior("nvda", path, now=NOW)["stance"] == "positive"


def test_an_expired_record_is_ignored_but_kept(tmp_path):
    path = tmp_path / "m.jsonl"
    save_run(make(days_ago=MAX_AGE_DAYS + 5), path)
    assert load_prior("NVDA", path, now=NOW) is None
    assert len(path.read_text().splitlines()) == 1  # still on disk


def test_missing_file_wrong_ticker_and_bad_lines_are_safe(tmp_path):
    path = tmp_path / "m.jsonl"
    assert load_prior("NVDA", path, now=NOW) is None  # no file
    path.write_text("not json\n[1, 2]\n" + json.dumps({"ticker": "NVDA"}) + "\n")
    save_run(make("AADX"), path)
    assert load_prior("NVDA", path, now=NOW) is None  # nothing usable
    save_run(make(days_ago=1), path)
    assert load_prior("NVDA", path, now=NOW) is not None  # good line still found


def test_a_future_dated_record_is_ignored(tmp_path):
    path = tmp_path / "m.jsonl"
    save_run(make(days_ago=-3), path)
    assert load_prior("NVDA", path, now=NOW) is None


def test_record_from_outcome_keeps_only_the_compact_fields():
    rec = record_from_outcome(FakeOutcome(), now=NOW)
    assert rec["ticker"] == SAMPLE_ANALYSIS["ticker"].upper()
    assert rec["signals"] == [{"name": "fcf_yield", "direction": "negative",
                               "value": -0.07}]
    assert "evidence_used" not in rec and rec["saved_at"] == NOW.isoformat()


def test_diff_reports_only_real_changes():
    old = make(days_ago=7, signals=[{"name": "a", "direction": "negative", "value": 1},
                                    {"name": "b", "direction": "positive", "value": 2}],
               gaps=["g1", "g2"])
    new = make(days_ago=0, stance="positive",
               signals=[{"name": "a", "direction": "neutral", "value": 1},
                        {"name": "b", "direction": "positive", "value": 2},
                        {"name": "c", "direction": "negative", "value": 3}],
               gaps=["g2", "g3"])
    diff = diff_runs(old, new)
    assert diff.days_since == 7
    assert diff.stance == ("neutral", "positive") and diff.confidence is None
    assert diff.signal_changes == [("a", "negative", "neutral")]  # c is new, b same
    assert diff.new_gaps == ["g3"] and diff.resolved_gaps == ["g1"]


def test_identical_runs_say_so():
    diff = diff_runs(make(days_ago=2), make(days_ago=0))
    assert diff.unchanged
    assert "no change" in format_diff(diff)[1]


def test_format_diff_lists_each_change():
    text = "\n".join(format_diff(RunDiff(
        days_since=3, stance=("neutral", "positive"),
        signal_changes=[("fcf_yield", "negative", "neutral")])))
    assert "3 day(s) ago" in text
    assert "stance: neutral -> positive" in text
    assert "fcf_yield: negative -> neutral" in text
