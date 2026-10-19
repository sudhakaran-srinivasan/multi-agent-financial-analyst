"""Run memory: the agent keeps a record of each analysis and compares runs.

AGENT DESIGN (rubric: Agent Functions - memory, learning across runs)
    Component : memory  (save_run, load_prior, diff_runs)
    Consumes  : the finished analysis and its signals (a SynthesisOutcome)
    Produces  : an append-only file of run records, and a RunDiff that says
                what changed since the last run of the same ticker
    Why       : a doctor reads your last chart, then asks "what changed?".
                The agent does the same: it can say "stance moved from neutral
                to positive because fcf_yield improved", which is learning
                across runs, instead of re-deriving everything from scratch.

DESIGN REASONING (AUTHOR: decisions made with the project owner)
    Store everything, append-only
        One JSON object per line (JSONL). Nothing is edited or deleted, so the
        history stays available for trends, audits and the evaluator.
    Filter at READ time, not write time
        Only the newest record younger than MAX_AGE_DAYS is used as context.
        Deleting is irreversible; if the cutoff turns out wrong we change one
        constant and the old data is still there.
    Why 30 days
        Companies report roughly every 90 days and a lot changes in a quarter.
        An older record would mislead more than it helps.
    Only the newest record is used
        Older records add cost and noise, and stale numbers must never leak
        into today's analysis. Memory supplies the prior VIEW as context; the
        numbers always come fresh from the tools.
    What is stored
        Stance, confidence, thesis, the signals (name, direction, value) and
        the data gaps. Not the raw evidence or the full trace: both are bulky
        and the signals already hold the same numbers in compact form.
    Python computes the diff
        The change list comes from stored fields, not from the model, so it
        cannot invent a change. The model may later explain it.

AI DISCLOSURE: first draft generated with Claude (Anthropic), reviewed and
modified by the author. Author-owned decisions are marked  # AUTHOR:.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from finagent.agent.synthesis import SynthesisOutcome

DEFAULT_PATH = Path("data/memory.jsonl")
# AUTHOR: a record older than this is ignored as context (kept on disk).
MAX_AGE_DAYS = 30


class SignalRecord(TypedDict):
    name: str
    direction: str
    value: float | str | None


class RunRecord(TypedDict):
    ticker: str
    saved_at: str  # ISO 8601, UTC
    stance: str
    confidence: str
    thesis: str
    signals: list[SignalRecord]
    data_gaps: list[str]


@dataclass
class RunDiff:
    """What changed between the prior run and this one (computed by Python)."""

    days_since: int
    stance: tuple[str, str] | None = None  # (before, after) when changed
    confidence: tuple[str, str] | None = None
    signal_changes: list[tuple[str, str, str]] = field(default_factory=list)
    new_gaps: list[str] = field(default_factory=list)
    resolved_gaps: list[str] = field(default_factory=list)

    @property
    def unchanged(self) -> bool:
        return not (self.stance or self.confidence or self.signal_changes
                    or self.new_gaps or self.resolved_gaps)


def _now() -> datetime:
    return datetime.now(UTC)


def record_from_outcome(outcome: SynthesisOutcome,
                        now: datetime | None = None) -> RunRecord:
    """Pick the fields worth remembering out of a finished analysis."""
    analysis = outcome.analysis
    return RunRecord(
        ticker=analysis["ticker"].upper(),
        saved_at=(now or _now()).isoformat(),
        stance=analysis["stance"],
        confidence=analysis["confidence"],
        thesis=analysis["thesis"],
        signals=[SignalRecord(name=s["name"], direction=s["direction"],
                              value=s["value"]) for s in outcome.signals],
        data_gaps=list(analysis["data_gaps"]),
    )


def save_run(record: RunRecord, path: Path = DEFAULT_PATH) -> None:
    """Append one record as a single JSON line. Never rewrites old lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _parse_time(text: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(text))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def load_prior(ticker: str, path: Path = DEFAULT_PATH, *,
               now: datetime | None = None,
               max_age_days: int = MAX_AGE_DAYS) -> RunRecord | None:
    """Newest saved record for ``ticker`` that is not older than the cutoff.

    Returns None when there is no file, no record, or only expired ones.
    A damaged line is skipped, never fatal: memory is helpful context, and a
    bad line must not stop an analysis.
    """
    if not path.exists():
        return None
    now = now or _now()
    wanted = ticker.strip().upper()
    best: tuple[datetime, RunRecord] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict) or record.get("ticker") != wanted:
            continue
        saved = _parse_time(record.get("saved_at"))
        if saved is None or saved > now:  # unreadable or from the future
            continue
        if (now - saved).days > max_age_days:
            continue
        if best is None or saved > best[0]:
            best = (saved, record)  # type: ignore[assignment]
    return best[1] if best else None


def diff_runs(prior: RunRecord, current: RunRecord) -> RunDiff:
    """Compare two records: only facts that are in the stored fields."""
    then = _parse_time(prior["saved_at"]) or _now()
    now = _parse_time(current["saved_at"]) or _now()
    diff = RunDiff(days_since=max((now - then).days, 0))
    if prior["stance"] != current["stance"]:
        diff.stance = (prior["stance"], current["stance"])
    if prior["confidence"] != current["confidence"]:
        diff.confidence = (prior["confidence"], current["confidence"])
    before = {s["name"]: s["direction"] for s in prior["signals"]}
    for sig in current["signals"]:
        old = before.get(sig["name"])
        if old is not None and old != sig["direction"]:
            diff.signal_changes.append((sig["name"], old, sig["direction"]))
    diff.new_gaps = [g for g in current["data_gaps"] if g not in prior["data_gaps"]]
    diff.resolved_gaps = [g for g in prior["data_gaps"]
                          if g not in current["data_gaps"]]
    return diff


def format_diff(diff: RunDiff) -> list[str]:
    """The diff as plain lines for reports (and, later, the model's prompt)."""
    lines = [f"Compared with the run {diff.days_since} day(s) ago:"]
    if diff.unchanged:
        return lines + ["  no change in stance, confidence, signals or gaps"]
    if diff.stance:
        lines.append(f"  stance: {diff.stance[0]} -> {diff.stance[1]}")
    if diff.confidence:
        lines.append(f"  confidence: {diff.confidence[0]} -> {diff.confidence[1]}")
    lines += [f"  {name}: {old} -> {new}" for name, old, new in diff.signal_changes]
    lines += [f"  new gap: {g}" for g in diff.new_gaps]
    lines += [f"  resolved gap: {g}" for g in diff.resolved_gaps]
    return lines
