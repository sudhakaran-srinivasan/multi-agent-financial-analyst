"""Offline tests for finagent.agent.synthesis, using a scripted fake model.

Definition of "this works":
    1. A good model reply becomes an Analysis with every contract key.
    2. data_gaps and sources are built by Python, never copied from the model.
    3. Evidence with an invented number or a missing source is dropped; with no
       verifiable evidence left the run falls back to the rule-based analysis.
    4. Confidence is capped: missing fundamentals or earnings -> low; patchy or
       thin data -> at most medium; empty news does not lower it.
    5. Expected gaps (loss-maker) are listed apart from genuine gaps and do not
       lower confidence.
    6. No fundamentals -> no model call. Model failure -> rule-based fallback.
    7. Python never raises for a model problem.
"""

import json
from types import SimpleNamespace as NS
from typing import get_type_hints

import pytest

from finagent.agent.synthesis import (
    apply_ceiling,
    build_data_gaps,
    check_evidence,
    confidence_ceiling,
    synthesize,
)
from finagent.contracts import Analysis

NVDA_FUND = {
    "ticker": "NVDA",
    "valuation": {"trailingPE": 30.24526, "forwardPE": 15.14251},
    "profitability": {"profitMargins": 0.63663},
    "growth": {"revenueGrowth": 1.059},
    "balance_sheet": {"currentRatio": 4.589},
    "price": {"currentPrice": 239.24, "twoHundredDayAverage": 201.0205},
    "sentiment": {"targetMeanPrice": 328.71695, "numberOfAnalystOpinions": 59.0},
    "derived": {"net_cash": 23608999936.0, "fcf_yield": 0.02198},
    "gaps": [], "warnings": [],
}
NVDA_EARN = {"summary": {"n": 4, "beats": 4, "in_line": 0, "misses": 0,
                         "beat_rate": 1.0}, "gaps": [], "warnings": []}

AADX_FUND = {
    "ticker": "AADX",
    "valuation": {"trailingPE": None, "forwardPE": 28.80503},
    "profitability": {"profitMargins": -0.30236998, "returnOnEquity": None},
    "growth": {"revenueGrowth": 0.474, "earningsGrowth": None},
    "balance_sheet": {"currentRatio": 2.509},
    "price": {"currentPrice": 11.45, "twoHundredDayAverage": 16.786556},
    "sentiment": {"targetMeanPrice": 25.125, "numberOfAnalystOpinions": 8.0},
    "derived": {"net_cash": -413432000.0, "fcf_yield": -0.06556797},
    "gaps": [{"field": "trailingPE", "reason": "negative_earnings"},
             {"field": "enterpriseToEbitda", "reason": "not_meaningful"},
             {"field": "pegRatio", "reason": "negative_earnings"},
             {"field": "returnOnEquity", "reason": "missing"},
             {"field": "earningsGrowth", "reason": "missing"}],
    "warnings": [],
}
AADX_EARN = {
    "summary": {"n": 1, "beats": 0, "in_line": 0, "misses": 1, "beat_rate": 0.0},
    "gaps": [{"field": "FY2026Q2.surprise_pct", "reason": "near_zero_estimate"}],
    "warnings": ["thin history: only 1 classifiable quarter(s) (expected 4)"],
}

PLAN = {"ticker": "X", "steps": [
    {"tool": "get_fundamentals", "reason": "r"},
    {"tool": "get_earnings_history", "reason": "r"},
    {"tool": "get_news_insights", "reason": "r"}], "priority_note": "n"}


def loop_result(ticker, results, *, stop_reason="done", trace=None, plan=PLAN):
    return {"ticker": ticker, "stop_reason": stop_reason, "steps_used": 3,
            "profile": {"ticker": ticker}, "plan": plan, "results": results,
            "trace": trace or [], "fatal_error": None}


NVDA_RESULT = loop_result("NVDA", {"get_fundamentals": NVDA_FUND,
                                   "get_earnings_history": NVDA_EARN,
                                   "get_news_insights": []})
AADX_RESULT = loop_result("AADX", {"get_fundamentals": AADX_FUND,
                                   "get_earnings_history": AADX_EARN,
                                   "get_news_insights": []})


def reply(**overrides):
    data = {
        "stance": "bullish", "confidence": "high",
        "thesis": "Forward P/E is half of trailing P/E, so growth is expected.",
        "evidence_used": [
            {"claim": "Forward P/E", "value": 15.1,
             "source": "get_fundamentals.valuation.forwardPE"},
            {"claim": "Beats", "value": 4,
             "source": "get_earnings_history.summary.beats"}],
        "risks": ["Valuation depends on forecasts that may be missed."],
    }
    data.update(overrides)
    return NS(choices=[NS(message=NS(content=json.dumps(data)))])


class ScriptedClient:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_good_reply_becomes_a_complete_analysis():
    outcome = synthesize(NVDA_RESULT, ScriptedClient(reply(confidence="medium")))
    analysis = outcome.analysis
    assert set(analysis) == set(get_type_hints(Analysis))
    assert outcome.fallback_reason is None
    assert analysis["stance"] == "bullish" and analysis["confidence"] == "medium"
    assert len(analysis["evidence_used"]) == 2
    assert analysis["ticker"] == "NVDA"


def test_gaps_and_sources_come_from_python_not_the_model():
    client = ScriptedClient(reply(
        data_gaps=["nothing is missing"], sources=["https://made.up"]))
    analysis = synthesize(NVDA_RESULT, client).analysis
    assert any("no news insights available" in g for g in analysis["data_gaps"])
    assert "https://made.up" not in analysis["sources"]
    assert "Yahoo Finance via yfinance (fundamentals)" in analysis["sources"]


def test_the_prompt_carries_facts_signals_gaps_and_the_ceiling():
    client = ScriptedClient(reply())
    synthesize(NVDA_RESULT, client)
    system, user = client.requests[0]["messages"]
    assert "NVDA" in system["content"] and "at most high" in system["content"]
    facts = json.loads(user["content"])
    assert {"plan", "profile", "results", "signals", "data_gaps"} <= set(facts)
    assert facts["signals"]  # computed by Python


def test_invented_evidence_number_is_dropped():
    bad = [{"claim": "Forward P/E", "value": 9.9,
            "source": "get_fundamentals.valuation.forwardPE"},
           {"claim": "Beats", "value": 4,
            "source": "get_earnings_history.summary.beats"}]
    outcome = synthesize(NVDA_RESULT, ScriptedClient(reply(evidence_used=bad)))
    assert [e["claim"] for e in outcome.analysis["evidence_used"]] == ["Beats"]
    assert any("is 15.14251, not 9.9" in n for n in outcome.notes)


def test_rounded_numbers_are_accepted():
    kept, dropped = check_evidence(
        [{"claim": "PE", "value": 30.2,
          "source": "get_fundamentals.valuation.trailingPE"}],
        NVDA_RESULT["results"])
    assert len(kept) == 1 and dropped == []


def test_evidence_citing_a_gap_or_unknown_source_is_dropped():
    kept, dropped = check_evidence(
        [{"claim": "ROE", "value": 1.2,
          "source": "get_fundamentals.profitability.returnOnEquity"},
         {"claim": "?", "value": 1, "source": "get_fundamentals.nowhere"}],
        AADX_RESULT["results"])
    assert kept == [] and len(dropped) == 2


def test_no_verifiable_evidence_falls_back_to_the_rule_based_analysis():
    bad = [{"claim": "x", "value": 1.0, "source": "get_fundamentals.nowhere"}]
    outcome = synthesize(NVDA_RESULT, ScriptedClient(reply(evidence_used=bad)))
    assert "no verifiable evidence" in outcome.fallback_reason
    assert outcome.analysis["confidence"] == "low"
    assert "Rule-based summary" in outcome.analysis["thesis"]


@pytest.mark.parametrize("bad_reply", [
    NS(choices=[NS(message=NS(content="I think it is great."))]),
    NS(choices=[NS(message=NS(content="{not json}"))]),
    NS(choices=[NS(message=NS(content=None))]),
    reply(stance="buy"),
    reply(confidence="very high"),
    reply(thesis="  "),
    reply(risks=[]),
])
def test_unusable_replies_fall_back_without_raising(bad_reply):
    outcome = synthesize(NVDA_RESULT, ScriptedClient(bad_reply))
    assert outcome.fallback_reason is not None
    assert set(outcome.analysis) == set(get_type_hints(Analysis))
    assert outcome.analysis["confidence"] == "low"


def test_model_api_failure_falls_back():
    outcome = synthesize(NVDA_RESULT, ScriptedClient(RuntimeError("402")))
    assert "402" in outcome.fallback_reason
    assert outcome.analysis["stance"] in ("bullish", "neutral", "bearish")


def test_rule_based_stance_follows_the_signal_balance():
    outcome = synthesize(NVDA_RESULT, ScriptedClient(RuntimeError("down")))
    assert outcome.analysis["stance"] == "bullish"  # 7 positive, 0 negative
    aadx = synthesize(AADX_RESULT, ScriptedClient(RuntimeError("down")))
    # AADX: 4 negative vs 3 positive (liquidity is fine): too mixed to call.
    assert aadx.analysis["stance"] == "neutral"
    assert "3 positive and 4 negative" in aadx.analysis["thesis"]


def test_rule_based_stance_turns_bearish_when_negatives_clearly_lead():
    results = {"get_fundamentals": {
        "profitability": {"profitMargins": -0.2},
        "growth": {"revenueGrowth": -0.1},
        "derived": {"net_cash": -5.0e8, "fcf_yield": -0.05},
        "price": {"currentPrice": 10.0, "twoHundredDayAverage": 15.0}},
        "get_earnings_history": NVDA_EARN}
    outcome = synthesize(loop_result("X", results),
                         ScriptedClient(RuntimeError("down")))
    assert outcome.analysis["stance"] == "bearish"


def test_no_fundamentals_means_no_model_call():
    result = loop_result("X", {"get_earnings_history": NVDA_EARN})
    client = ScriptedClient()
    outcome = synthesize(result, client)
    assert client.requests == []
    assert outcome.analysis["stance"] == "neutral"
    assert outcome.analysis["confidence"] == "low"
    assert outcome.analysis["evidence_used"] == []
    assert any("get_fundamentals: no result" in g
               for g in outcome.analysis["data_gaps"])


# ---- confidence ceiling ------------------------------------------------------
def test_ceiling_is_high_for_complete_data_and_empty_news_does_not_lower_it():
    assert confidence_ceiling(NVDA_RESULT["results"]) == ("high", None)


def test_missing_earnings_history_caps_at_low():
    ceiling, reason = confidence_ceiling({"get_fundamentals": NVDA_FUND})
    assert ceiling == "low" and "earnings" in reason


def test_missing_fundamentals_caps_at_low():
    ceiling, reason = confidence_ceiling({"get_earnings_history": NVDA_EARN})
    assert ceiling == "low" and "fundamentals" in reason


def test_three_genuine_core_gaps_cap_at_medium():
    fund = {**NVDA_FUND, "gaps": [
        {"field": "returnOnEquity", "reason": "missing"},
        {"field": "earningsGrowth", "reason": "missing"},
        {"field": "grossMargins", "reason": "missing"}]}
    ceiling, _ = confidence_ceiling({"get_fundamentals": fund,
                                     "get_earnings_history": NVDA_EARN})
    assert ceiling == "medium"


def test_expected_gaps_do_not_lower_confidence():
    fund = {**NVDA_FUND, "gaps": [
        {"field": "trailingPE", "reason": "negative_earnings"},
        {"field": "pegRatio", "reason": "negative_earnings"},
        {"field": "enterpriseToEbitda", "reason": "not_meaningful"}]}
    assert confidence_ceiling({"get_fundamentals": fund,
                               "get_earnings_history": NVDA_EARN})[0] == "high"


def test_thin_earnings_history_caps_at_medium():
    ceiling, reason = confidence_ceiling(AADX_RESULT["results"])
    assert ceiling == "medium" and "1 quarter" in reason


def test_model_cannot_exceed_the_ceiling_and_the_cut_is_noted():
    outcome = synthesize(AADX_RESULT, ScriptedClient(
        reply(stance="bearish", confidence="high", evidence_used=[
            {"claim": "FCF yield", "value": -0.0656,
             "source": "get_fundamentals.derived.fcf_yield"}])))
    assert outcome.analysis["confidence"] == "medium"
    assert any("lowered from high to medium" in n for n in outcome.notes)


def test_model_may_choose_lower_than_the_ceiling():
    assert apply_ceiling("low", "high") == "low"
    assert apply_ceiling("high", "medium") == "medium"


# ---- data gaps ---------------------------------------------------------------
def test_expected_gaps_are_listed_apart_from_genuine_ones():
    gaps = build_data_gaps(AADX_RESULT)
    assert "get_fundamentals: returnOnEquity missing" in gaps
    assert "get_fundamentals: earningsGrowth missing" in gaps
    assert ("get_fundamentals: not applicable for this company: trailingPE, "
            "enterpriseToEbitda, pegRatio") in gaps
    assert any("thin history" in g for g in gaps)
    assert "get_news_insights: no news insights available" in gaps


def test_failed_planned_tool_is_reported_with_its_error():
    trace = [{"step": 1, "tool": "get_earnings_history", "arguments": {},
              "ok": False, "error": "RuntimeError: Finnhub down"}]
    result = loop_result("NVDA", {"get_fundamentals": NVDA_FUND}, trace=trace)
    gaps = build_data_gaps(result)
    assert any("get_earnings_history: no result (RuntimeError: Finnhub down)" in g
               for g in gaps)


def test_step_cap_is_reported_as_a_gap():
    result = loop_result("NVDA", NVDA_RESULT["results"], stop_reason="step_cap")
    assert any("step cap" in g for g in build_data_gaps(result))


def test_news_urls_become_sources():
    news = [{"category": "earnings", "summary": "s", "sentiment": "positive",
             "source_url": "https://example.com/a"}]
    result = loop_result("NVDA", {**NVDA_RESULT["results"],
                                  "get_news_insights": news})
    outcome = synthesize(result, ScriptedClient(reply()))
    assert "https://example.com/a" in outcome.analysis["sources"]
    assert not any("no news" in g for g in outcome.analysis["data_gaps"])


# ---- reflection as optional context ------------------------------------------
REFLECTION = {"retried": [{"tool": "get_news_insights", "why": "failed",
                           "outcome": "still_failing", "error": "RuntimeError: x"}],
              "remaining_gaps": ["get_news_insights: no result (RuntimeError: x)"],
              "summary": "1 tool retried: 0 recovered, 1 still failing; 1 gap remains."}


def test_reflection_is_passed_to_the_model_and_noted():
    client = ScriptedClient(reply())
    outcome = synthesize(NVDA_RESULT, client, REFLECTION)
    facts = json.loads(client.requests[0]["messages"][1]["content"])
    assert facts["reflection"]["retried"][0]["tool"] == "get_news_insights"
    assert any(n.startswith("reflection:") for n in outcome.notes)


def test_synthesis_works_without_reflection():
    client = ScriptedClient(reply())
    outcome = synthesize(NVDA_RESULT, client)
    assert json.loads(client.requests[0]["messages"][1]["content"])["reflection"] is None
    assert not any(n.startswith("reflection:") for n in outcome.notes)


# ---- stale news --------------------------------------------------------------
def _with_news(items):
    return {**AADX_RESULT, "results": {**AADX_RESULT["results"],
                                       "get_news_insights": items}}


def _news(published):
    return {"category": "general", "summary": "s", "sentiment": "neutral",
            "source_url": "https://x", "published": published}


def test_old_news_is_listed_as_a_stale_gap():
    from datetime import date

    gaps = build_data_gaps(_with_news([_news("2026-07-20"), _news("2026-08-01")]),
                           today=date(2026, 10, 13))
    assert any("stale" in g and "73 days old" in g for g in gaps)


def test_fresh_news_is_not_a_gap():
    from datetime import date

    gaps = build_data_gaps(_with_news([_news("2026-10-12"), _news("2026-07-01")]),
                           today=date(2026, 10, 13))
    assert not any("get_news_insights" in g for g in gaps)


def test_news_without_a_usable_date_is_not_flagged_stale():
    from datetime import date

    item = {k: v for k, v in _news("x").items() if k != "published"}
    gaps = build_data_gaps(_with_news([item, _news("not a date")]),
                           today=date(2026, 10, 13))
    assert not any("stale" in g for g in gaps)