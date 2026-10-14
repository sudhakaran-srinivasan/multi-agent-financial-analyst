"""Offline tests for the get_news_insights tool (no model, no network).

Definition of "this works":
    1. It runs ingest -> classify -> specialists and returns NewsInsight items
       that match the contract (with a published date).
    2. A failed fetch raises (never [] ); found-nothing and all-off-topic
       return [] without building a model client unless one is needed.
    3. The company name reaches the fetcher and the prompts; a failed profile
       lookup degrades to the ticker alone.
"""
import json
from datetime import UTC, datetime
from types import SimpleNamespace as NS

import pytest

from finagent.contracts import NewsInsight
from finagent.tools.news import get_news_insights
from finagent.workflows.news_classify import NewsClassifyError
from finagent.workflows.news_ingest import Article, NewsBatch, NewsFetchError

WHEN = datetime(2026, 10, 13, tzinfo=UTC)


def art(headline):
    return Article(headline, "A short summary with no numbers.",
                   f"https://x/{headline}", "Yahoo", WHEN)


def batch(*headlines):
    return NewsBatch([art(h) for h in headlines], 14, widened=False)


class RoutingClient:
    """Answers by prompt type: classify, extract or summarize."""

    def __init__(self, labels=None, classify_error=None):
        self.labels = labels or {}
        self.classify_error = classify_error
        self.requests = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        system, user = (m["content"] for m in kwargs["messages"])
        if system.startswith("You label one news item"):
            if self.classify_error:
                raise self.classify_error
            label = next((v for k, v in self.labels.items() if k in user), "general")
            text = json.dumps({"label": label, "reason": "test"})
        elif "extract" in system:
            text = json.dumps({"key_facts": ["A fact."], "sentiment": "positive"})
        else:
            text = json.dumps({"summary": "A short summary."})
        return NS(choices=[NS(message=NS(content=text))])


def profile_ok(ticker):
    return NS(name="Applied Aerospace & Defense, Inc.")


def make_fetcher(result, seen=None):
    def fetcher(ticker, company_name=None):
        if seen is not None:
            seen.update(ticker=ticker, company_name=company_name)
        if isinstance(result, Exception):
            raise result
        return result
    return fetcher


def test_the_tool_returns_news_insights_matching_the_contract():
    client = RoutingClient({"Q2 results": "earnings"})
    out = get_news_insights("aadx", client=client, profile_fn=profile_ok,
                            fetcher=make_fetcher(batch("Q2 results", "New deal")))
    assert [i["category"] for i in out] == ["earnings", "general"]
    for item in out:
        assert set(item) == set(NewsInsight.__annotations__)
        assert item["published"] == "2026-10-13" and item["sentiment"] == "positive"


def test_the_ticker_is_normalized_and_the_company_name_is_passed_on():
    seen = {}
    client = RoutingClient()
    get_news_insights(" aadx ", client=client, profile_fn=profile_ok,
                      fetcher=make_fetcher(batch("Deal"), seen))
    assert seen == {"ticker": "AADX",
                    "company_name": "Applied Aerospace & Defense, Inc."}
    system = client.requests[0]["messages"][0]["content"]
    assert "AADX (Applied Aerospace & Defense, Inc.)" in system


def test_a_failed_profile_lookup_falls_back_to_the_ticker():
    def boom(ticker):
        raise RuntimeError("yahoo down")

    seen = {}
    client = RoutingClient()
    out = get_news_insights("AADX", client=client, profile_fn=boom,
                            fetcher=make_fetcher(batch("Deal"), seen))
    assert len(out) == 1 and seen["company_name"] is None
    assert "company AADX." in client.requests[0]["messages"][0]["content"]


def test_a_failed_fetch_raises_instead_of_returning_empty():
    with pytest.raises(NewsFetchError):
        get_news_insights("AADX", client=RoutingClient(), profile_fn=profile_ok,
                          fetcher=make_fetcher(NewsFetchError("down")))


def test_no_articles_returns_empty_without_building_a_client(monkeypatch):
    def no_client():
        raise AssertionError("a client must not be built when there is no news")

    monkeypatch.setattr("finagent.agent.loop.make_client", no_client)
    out = get_news_insights("AADX", profile_fn=profile_ok,
                            fetcher=make_fetcher(NewsBatch([], 14, widened=False)))
    assert out == []


def test_all_off_topic_returns_empty():
    client = RoutingClient({"Deere": "irrelevant", "Lilly": "irrelevant"})
    out = get_news_insights("AADX", client=client, profile_fn=profile_ok,
                            fetcher=make_fetcher(batch("Deere", "Lilly")))
    assert out == []


def test_a_broken_classifier_raises_instead_of_returning_empty():
    client = RoutingClient(classify_error=ConnectionError("model down"))
    with pytest.raises(NewsClassifyError):
        get_news_insights("AADX", client=client, profile_fn=profile_ok,
                          fetcher=make_fetcher(batch("Deal")))