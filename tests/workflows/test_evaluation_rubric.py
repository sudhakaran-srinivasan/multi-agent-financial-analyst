"""Offline tests for the rubric and the critical-issue rule.

Definition of "this works":
    1. Every criterion the rules score has an anchored description, and no extra.
    2. The judge prompt contains every criterion and the no-outside-facts rule.
    3. A critical issue forces "revise" even when the scores would pass.
"""
from finagent.workflows.evaluation_rubric import RUBRIC, build_rubric_text
from finagent.workflows.evaluation_rules import RUBRIC_CRITERIA, decide_verdict


def _scores(value):
    return [{"criterion": c, "score": value, "comment": ""} for c in RUBRIC_CRITERIA]


def test_rubric_and_rules_name_the_same_criteria_in_the_same_order():
    assert tuple(RUBRIC) == RUBRIC_CRITERIA


def test_every_criterion_has_all_three_anchors():
    for name, c in RUBRIC.items():
        assert all(c[k] for k in ("question", "score_5", "score_3", "score_1")), name


def test_the_judge_prompt_lists_every_criterion_and_the_rules():
    text = build_rubric_text()
    assert all(name in text for name in RUBRIC_CRITERIA)
    assert "Do NOT assume missing evidence exists" in text
    assert "CRITICAL ISSUES" in text


def test_a_critical_issue_blocks_a_pass():
    assert decide_verdict(_scores(5))[0] == "pass"
    assert decide_verdict(_scores(5), critical_issues=["number invented"])[0] == "revise"
    assert decide_verdict(_scores(5), critical_issues=[])[0] == "pass"