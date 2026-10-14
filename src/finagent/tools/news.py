"""get_news_insights: the news tool the research agent calls.

AGENT DESIGN (rubric: Agent Functions - tool use; Workflow Patterns)
    Component : get_news_insights  (agent tool that wraps the news workflow)
    Consumes  : a ticker
    Produces  : list[NewsInsight]  (category, summary, sentiment, url, date)

    The tool is thin on purpose. The real work lives in the workflow modules,
    one station each, so each can be tested alone:

        workflows/news_ingest.py      Ingest + Preprocess   (Python)
        workflows/news_classify.py    Classify              (LLM + Python check)
        workflows/news_specialists.py Route, Extract, Summarize (LLM + checks)

    Return value rules (the same failure rule as the other tools):
        * fetch failed, or every article failed a station  -> RAISES, so the
          dispatcher records ok=False and reflection can retry;
        * searched fine but nothing relevant                -> [] ("no news").
      That is how the agent tells an outage from a quiet stock.

    The company name (used to rank articles and to tell the classifier which
    company it is judging) comes from the Yahoo profile. If that lookup fails
    the tool carries on with the ticker alone: a ranking hint is not worth
    failing the whole tool for.

AI DISCLOSURE: pipeline code drafted with Claude (Anthropic), reviewed and
modified by the author. The team's ingest.ipynb was the starting reference.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from finagent.contracts import NewsInsight
from finagent.tools.company_profile import get_company_profile
from finagent.workflows.news_classify import classify_articles, make_llm_classifier
from finagent.workflows.news_ingest import NewsBatch, collect_articles
from finagent.workflows.news_specialists import build_insights, make_llm_analyzer


def _company_name(ticker: str, profile_fn: Callable[[str], Any]) -> str | None:
    """Best-effort company name; None if the lookup fails for any reason."""
    try:
        name = profile_fn(ticker).name
    except Exception:  # noqa: BLE001 - optional hint, never fail the tool for it
        return None
    return name if isinstance(name, str) and name.strip() and name != ticker else None


def get_news_insights(
    ticker: str,
    *,
    client: Any = None,
    fetcher: Callable[..., NewsBatch] = collect_articles,
    profile_fn: Callable[[str], Any] = get_company_profile,
) -> list[NewsInsight]:
    """Return analyzed news for ``ticker``.

    Args:
        ticker: Stock symbol, already validated by the profile step.
        client: Model client (built on demand); injectable for tests.
        fetcher: Ingest + preprocess function; injectable for tests.
        profile_fn: Company-profile lookup; injectable for tests.

    Raises:
        NewsFetchError: the news provider call failed.
        NewsClassifyError / NewsAnalysisError: every article failed a station.
    """
    ticker = ticker.strip().upper()
    name = _company_name(ticker, profile_fn)
    batch = fetcher(ticker, company_name=name)
    if not batch.articles:
        return []  # searched fine, found nothing

    if client is None:
        # Imported here: agent.loop imports the tool registry, which imports
        # this module, so a top-level import would be circular.
        from finagent.agent.loop import make_client

        client = make_client()
    subject = f"{ticker} ({name})" if name else ticker

    classified = classify_articles(batch.articles, subject,
                                   make_llm_classifier(client))
    if not classified.items:
        return []  # everything found was off-topic
    return build_insights(classified, subject, make_llm_analyzer(client)).insights