"""Offline tests for the evaluator-optimizer (scripted judge and reviser).

Definition of "this works":
    1. The judge's reply is validated strictly: all six criteria, whole
       numbers 1-5, valid JSON. Anything else raises JudgeError.
    2. The LLM judge retries an unusable reply, then gives up with JudgeError.
    3. Python decides the verdict (mean >= 4.0, no score < 3, no critical issue).
    4. The loop stops on a pass, revises on a fail, and never exceeds MAX_ROUNDS.
    5. After the last round, the BEST draft is returned and marked not passed.
    6. A broken judge, a broken reviser or a rule-based fallback revision stops
       the loop safely and keeps the best draft; nothing raises.
    7. Notes for the reviser put critical issues first and are capped.
"""
import json
from types import SimpleNamespace as NS

import pytest

from finagent.agent.synthesis import SynthesisOutcome
from finagent.samples import SAMPLE_ANALYSIS
from finagent.workflows.evaluation_rules import MAX_ROUNDS, RUBRIC_CRITERIA
from finagent.workflows.evaluator_optimizer import (
    MAX_FEEDBACK_ITEMS,
    MAX_JUDGE_ATTEMPTS,
    JudgeError,
    Judgment,
    build_feedback,
    build_judge_messages,
    evaluate_and_optimize,
    make_llm_judge,
    parse_judgment,
    to_evaluation,
)

RESULTS = {"get_fundamentals": {"valuation": {"forwardPE": 15.1}}}


def scores(value=4, **overrides):
    out = {c: value for c in RUBRIC_CRITERIA}
    out.update(overrides)
    return [{"criterion": c, "score": v, "comment": f"{c} note"}
            for c, v in out.items()]


def judgment(value=4, critical=None, recs=None, **overrides):
    return Judgment(scores(value, **overrides), critical_issues=critical or [],
                    recommendations=recs or [])


def reply_json(value=4, **overrides):
    entries = {c: {"score": overrides.get(c, value), "comment": "ok"}
               for c in RUBRIC_CRITERIA}
    return json.dumps({"scores": entries, "strengths": ["s"], "issues": [],
                       "recommendations": ["r"], "critical_issues": []})


def outcome(stance="neutral", fallback=None):
    return SynthesisOutcome({**SAMPLE_ANALYSIS, "stance": stance},
                            fallback, [], [])


class ScriptJudge:
    """Returns scripted Judgments in order; an Exception item is raised."""

    def __init__(self, *items):
        self.items, self.calls = list(items), []

    def __call__(self, analysis, results, signals):
        self.calls.append(analysis)
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class ScriptReviser:
    def __init__(self, *items):
        self.items, self.calls = list(items), []

    def __call__(self, feedback, previous):
        self.calls.append((feedback, previous))
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


# ---- parsing -----------------------------------------------------------------
def test_a_valid_reply_becomes_a_judgment():
    j = parse_judgment("Sure:\n" + reply_json(4, risk_coverage=3))
    assert [s["criterion"] for s in j.scores] == list(RUBRIC_CRITERIA)
    assert j.scores[3]["score"] == 3 and j.recommendations == ["r"]


@pytest.mark.parametrize("bad", [
    "no json here",
    "{not json}",
    json.dumps({"strengths": []}),
    json.dumps({"scores": {"evidence_support": {"score": 4, "comment": ""}}}),
])
def test_unusable_replies_raise(bad):
    with pytest.raises(JudgeError):
        parse_judgment(bad)


@pytest.mark.parametrize("value", [0, 6, 3.5, True, "4", None])
def test_scores_must_be_whole_numbers_from_1_to_5(value):
    data = json.loads(reply_json())
    data["scores"]["numeric_accuracy"]["score"] = value
    with pytest.raises(JudgeError):
        parse_judgment(json.dumps(data))


def test_the_judge_prompt_puts_data_in_the_user_message_only():
    system, user = build_judge_messages(SAMPLE_ANALYSIS, RESULTS, [])
    assert "numeric_accuracy" in system["content"]
    assert "forwardPE" in user["content"] and "forwardPE" not in system["content"]


# ---- the LLM judge and its retry ---------------------------------------------
class Client:
    def __init__(self, *texts):
        self.texts, self.requests = list(texts), []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        text = self.texts.pop(0)
        return NS(choices=[NS(message=NS(content=text))])


def test_the_judge_retries_an_unusable_reply_then_succeeds():
    client = Client("garbage", reply_json())
    j = make_llm_judge(client)(SAMPLE_ANALYSIS, RESULTS, [])
    assert len(client.requests) == 2 and len(j.scores) == len(RUBRIC_CRITERIA)
    assert "unusable" in client.requests[1]["messages"][-1]["content"]


def test_the_judge_gives_up_after_the_attempt_limit():
    client = Client(*["garbage"] * MAX_JUDGE_ATTEMPTS)
    with pytest.raises(JudgeError, match="failed"):
        make_llm_judge(client)(SAMPLE_ANALYSIS, RESULTS, [])
    assert len(client.requests) == MAX_JUDGE_ATTEMPTS


# ---- feedback ----------------------------------------------------------------
def test_feedback_lists_critical_issues_first_then_weak_scores_then_advice():
    notes = build_feedback(judgment(5, critical=["number invented"],
                                    recs=["add a risk"], risk_coverage=2))
    assert notes[0] == "CRITICAL: number invented"
    assert notes[1].startswith("risk_coverage scored 2")
    assert notes[-1] == "add a risk"


def test_feedback_is_deduplicated_and_capped():
    notes = build_feedback(judgment(5, recs=[f"fix {i}" for i in range(20)] + ["fix 0"]))
    assert len(notes) == MAX_FEEDBACK_ITEMS and len(set(notes)) == len(notes)


# ---- the loop ----------------------------------------------------------------
def test_a_passing_draft_is_not_revised():
    judge, reviser = ScriptJudge(judgment(5)), ScriptReviser()
    run = evaluate_and_optimize(outcome(), RESULTS, judge, reviser)
    assert run.passed and run.revisions == 0 and not reviser.calls
    assert run.final.analysis["stance"] == "neutral"


def test_a_failing_draft_is_revised_then_passes():
    first, better = outcome("neutral"), outcome("bullish")
    judge = ScriptJudge(judgment(2, recs=["fix it"]), judgment(5))
    reviser = ScriptReviser(better)
    run = evaluate_and_optimize(first, RESULTS, judge, reviser)
    assert run.passed and run.revisions == 1 and run.final is better
    feedback, previous = reviser.calls[0]
    assert "fix it" in feedback and previous is first.analysis
    assert [r.label for r in run.rounds] == ["draft", "revision 1"]


def test_the_loop_never_exceeds_max_rounds_and_keeps_the_best_draft():
    drafts = [outcome("a"), outcome("b"), outcome("c")]
    judge = ScriptJudge(judgment(2), judgment(3), judgment(2))
    reviser = ScriptReviser(drafts[1], drafts[2])
    run = evaluate_and_optimize(drafts[0], RESULTS, judge, reviser)
    assert run.revisions == MAX_ROUNDS and not run.passed
    assert run.final is drafts[1]  # score 3 beat 2 and 2
    assert any("did NOT pass" in n for n in run.notes)
    assert to_evaluation(run)["verdict"] == "revise"


def test_a_tie_keeps_the_original_draft():
    drafts = [outcome("a"), outcome("b"), outcome("c")]
    judge = ScriptJudge(judgment(3), judgment(3), judgment(3))
    run = evaluate_and_optimize(drafts[0], RESULTS, judge,
                                ScriptReviser(drafts[1], drafts[2]))
    assert run.final is drafts[0]


def test_a_critical_issue_blocks_a_pass_even_with_high_scores():
    judge = ScriptJudge(judgment(5, critical=["stance contradicts evidence"]),
                        judgment(5))
    run = evaluate_and_optimize(outcome("a"), RESULTS, judge,
                                ScriptReviser(outcome("b")))
    assert run.revisions == 1 and run.passed


def test_fewer_critical_issues_beat_more_when_nothing_passes():
    drafts = [outcome("a"), outcome("b"), outcome("c")]
    judge = ScriptJudge(judgment(5, critical=["x", "y"]),
                        judgment(5, critical=["x"]),
                        judgment(5, critical=["x", "y", "z"]))
    run = evaluate_and_optimize(drafts[0], RESULTS, judge,
                                ScriptReviser(drafts[1], drafts[2]))
    assert run.final is drafts[1]


def test_a_judge_failure_on_the_first_draft_means_not_evaluated():
    draft = outcome()
    run = evaluate_and_optimize(draft, RESULTS, ScriptJudge(JudgeError("bad")),
                                ScriptReviser())
    assert run.final is draft and not run.evaluated and not run.passed
    assert "not evaluated" in run.notes[0] and to_evaluation(run) is None


def test_a_judge_failure_after_a_revision_keeps_the_judged_draft():
    draft = outcome("a")
    run = evaluate_and_optimize(
        draft, RESULTS, ScriptJudge(judgment(2), JudgeError("bad")),
        ScriptReviser(outcome("b")))
    assert run.final is draft and not run.passed  # the unjudged revision is dropped


def test_a_reviser_crash_stops_the_loop_with_the_draft():
    draft = outcome()
    run = evaluate_and_optimize(draft, RESULTS, ScriptJudge(judgment(2)),
                                ScriptReviser(RuntimeError("boom")))
    assert run.final is draft and "reviser failed" in run.stopped_because


def test_a_rule_based_fallback_revision_is_not_used():
    draft = outcome()
    run = evaluate_and_optimize(
        draft, RESULTS, ScriptJudge(judgment(2)),
        ScriptReviser(outcome(fallback="model reply rejected")))
    assert run.final is draft and run.revisions == 0
    assert "rule-based" in run.stopped_because


def test_to_evaluation_matches_the_contract_shape():
    run = evaluate_and_optimize(outcome(), RESULTS, ScriptJudge(judgment(5)),
                                ScriptReviser())
    ev = to_evaluation(run)
    assert ev["verdict"] == "pass" and ev["rounds_used"] == 0
    assert ev["overall_score"] == pytest.approx(5.0)
    assert ev["final_analysis"] is run.final.analysis