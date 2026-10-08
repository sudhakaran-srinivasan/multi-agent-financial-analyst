"""Samples must match the contracts; the verdict rule must behave."""
import types
import typing
from typing import Literal, Union, get_args, get_origin, get_type_hints

import pytest

from finagent.contracts import (
    Analysis, CriterionScore, Evaluation, NewsInsight, Plan,
)
from finagent.samples import (
    SAMPLE_ANALYSIS, SAMPLE_EVALUATION, SAMPLE_NEWS_INSIGHT, SAMPLE_PLAN,
)
from finagent.workflows.evaluation_rules import (
    CRITERION_FLOOR, MAX_ROUNDS, PASS_AVERAGE, RUBRIC_CRITERIA, decide_verdict,
)


def check(value, tp, path="value"):
    """Recursively verify `value` conforms to type `tp` (raises AssertionError)."""
    origin = get_origin(tp)
    if origin is Literal:
        assert value in get_args(tp), f"{path}={value!r} not in {get_args(tp)}"
    elif origin is list:
        assert isinstance(value, list), f"{path} must be a list"
        for i, item in enumerate(value):
            check(item, get_args(tp)[0], f"{path}[{i}]")
    elif origin in (Union, types.UnionType):
        for option in get_args(tp):
            try:
                check(value, option, path)
                return
            except AssertionError:
                continue
        raise AssertionError(f"{path}={value!r} matches none of {get_args(tp)}")
    elif typing.is_typeddict(tp):
        hints = get_type_hints(tp)
        assert set(value) == set(hints), f"{path} keys {set(value)} != {set(hints)}"
        for key, sub in hints.items():
            check(value[key], sub, f"{path}.{key}")
    elif tp is type(None):
        assert value is None, f"{path} must be None"
    elif tp is float:  # like PEP 484: an int is acceptable where float is declared
        assert isinstance(value, (int, float)) and not isinstance(value, bool), \
            f"{path}={value!r} is not a number"
    elif tp is bool:
        assert isinstance(value, bool), f"{path}={value!r} is not a bool"
    else:
        # bool is a subclass of int, so exclude it explicitly for int fields
        assert isinstance(value, tp) and not (tp is int and isinstance(value, bool)), \
            f"{path}={value!r} is not {tp}"


@pytest.mark.parametrize("sample, contract", [
    (SAMPLE_NEWS_INSIGHT, NewsInsight),
    (SAMPLE_PLAN, Plan),
    (SAMPLE_ANALYSIS, Analysis),
    (SAMPLE_EVALUATION, Evaluation),
])
def test_samples_conform_to_contracts(sample, contract):
    check(sample, contract)


def test_checker_catches_a_bad_label():
    bad = {**SAMPLE_NEWS_INSIGHT, "sentiment": "very positive"}
    with pytest.raises(AssertionError):
        check(bad, NewsInsight)


def test_checker_catches_a_missing_key():
    bad = {k: v for k, v in SAMPLE_NEWS_INSIGHT.items() if k != "source_url"}
    with pytest.raises(AssertionError):
        check(bad, NewsInsight)


def scores(*values):
    return [CriterionScore(criterion=c, score=v, comment="")
            for c, v in zip(RUBRIC_CRITERIA, values)]


def test_high_scores_pass():
    assert decide_verdict(scores(5, 4, 4, 4, 4)) == ("pass", 4.2)


def test_one_critical_defect_forces_revise_despite_good_average():
    verdict, overall = decide_verdict(scores(5, 5, 5, 5, 2))
    assert overall == pytest.approx(4.4) and verdict == "revise"


def test_low_average_revises_even_without_a_critical_defect():
    assert decide_verdict(scores(3, 3, 4, 3, 3))[0] == "revise"


def test_no_scores_cannot_pass():
    assert decide_verdict([]) == ("revise", 0.0)


def test_sample_evaluation_verdict_matches_the_rule():
    verdict, overall = decide_verdict(SAMPLE_EVALUATION["scores"])
    assert verdict == SAMPLE_EVALUATION["verdict"]
    assert overall == pytest.approx(SAMPLE_EVALUATION["overall_score"])


def test_rule_constants_are_sane():
    assert 1 <= CRITERION_FLOOR <= PASS_AVERAGE <= 5
    assert MAX_ROUNDS == 2 and len(RUBRIC_CRITERIA) == 5


def test_checker_rejects_bool_where_number_expected():
    bad = {**SAMPLE_ANALYSIS["evidence_used"][0], "value": True}
    with pytest.raises(AssertionError):
        check(bad, typing.get_type_hints(Analysis)["evidence_used"].__args__[0])