"""News tool (STUB until the news pipeline is plugged in).

Agent Functions / Tools: the research agent calls ``get_news_insights`` like
any other tool. The real implementation is the prompt-chaining pipeline
(ingest, preprocess, classify and route, extract, summarize) built by the
news-agent owner. This stub only fixes the function's *signature*, so the
real pipeline can replace the body without changing anything else.

Why an empty list and not sample data?
    Mock text (for example "NVIDIA beat estimates") would leak into a live
    analysis of a different company and look like real evidence. An empty
    list is honest: "no news insights available yet". The agent then treats
    news as a data gap instead of inventing support for its thesis.

AI DISCLOSURE: 
"""

from __future__ import annotations

from finagent.contracts import NewsInsight


def get_news_insights(ticker: str) -> list[NewsInsight]:
    """Return classified news insights for ``ticker`` (stub: none yet).

    Args:
        ticker: Stock symbol, already validated and upper-cased.

    Returns:
        A list of ``NewsInsight`` items. Empty means none are available.
    """
    return []