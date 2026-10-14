"""News ingest and preprocess: the first two stations of the news chain.

AGENT DESIGN (rubric: Workflow Patterns - prompt chaining)
    Chain : Ingest News -> Preprocess -> Classify -> Extract -> Summarize
    Here  : Ingest (fetch articles from Finnhub) and Preprocess (clean them).
            Both are plain Python: deterministic, free and unit-testable.
            An LLM is used only from Classify onward, where judgement is needed.

    Why Python for preprocess? Cleaning, de-duplicating and age filtering have
    exactly one right answer. A model would add cost and randomness for no gain.

    Failure rule (same as the other tools): a failed fetch RAISES, so the
    dispatcher records ok=False and reflection can retry. An empty list means
    "searched fine, found nothing".

    Free tier (checked before building around it): Finnhub company-news is
    free at 60 calls/minute. We make ONE call per run: it fetches the wide
    window once, and the 14-day / 90-day choice is made locally.

AI DISCLOSURE: first draft generated with Claude (Anthropic), reviewed and
modified by the author. Author-owned decisions are marked  # AUTHOR:.
Reference: the team's ingest.ipynb (news idea and yfinance news exploration).
"""
from __future__ import annotations

import html
import os
import re
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any

# AUTHOR: news is a signal about what is changing NOW; older items are priced in.
LOOKBACK_DAYS = 14
# AUTHOR: thin-coverage companies (small caps) get one wider look, flagged stale.
WIDE_LOOKBACK_DAYS = 90
MIN_ARTICLES = 3  # fewer than this in the short window triggers the wider look
# AUTHOR: two caps. Finnhub tags a mega-cap's feed loosely (Nvidia's newest
# items included Deere, Lilly and Citigroup), so we classify a LARGER pool
# (cheap one-call-per-article) and keep only the first MAX_ARTICLES relevant
# ones (the expensive extract + summarize chain runs on those).
MAX_CANDIDATES = 20  # articles passed to the classifier
MAX_ARTICLES = 8  # relevant articles kept for the specialists
MAX_SUMMARY_CHARS = 500  # vendor summaries are short; this guards odd outliers

_TAG = re.compile(r"<[^>]+>")
_SPACES = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^a-z0-9]+")


class NewsFetchError(RuntimeError):
    """The news provider call failed (not the same as 'no news found')."""


@dataclass(frozen=True)
class Article:
    """One news item after parsing. Frozen: stages return new objects."""

    headline: str
    summary: str
    url: str
    source: str
    published: datetime  # timezone-aware UTC


@dataclass(frozen=True)
class NewsBatch:
    """Articles chosen for analysis, plus how they were chosen."""

    articles: list[Article]
    window_days: int  # the window that produced `articles`
    widened: bool  # True: the short window was too thin, so items are older


def _make_client() -> Any:
    import finnhub  # local import keeps the parsing code offline-testable

    key = os.environ.get("FINNHUB_API_KEY")
    if not key:
        raise RuntimeError("FINNHUB_API_KEY is not set (see .env.example)")
    return finnhub.Client(api_key=key)


def parse_article(row: Any) -> Article | None:
    """Turn one Finnhub row into an Article, or None if it is unusable."""
    if not isinstance(row, dict):
        return None
    headline = row.get("headline")
    url = row.get("url")
    stamp = row.get("datetime")
    if not (isinstance(headline, str) and headline.strip()
            and isinstance(url, str) and url.strip()):
        return None
    if isinstance(stamp, bool) or not isinstance(stamp, (int, float)):
        return None
    try:
        published = datetime.fromtimestamp(stamp, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None
    summary = row.get("summary")
    source = row.get("source")
    return Article(
        headline=headline,
        summary=summary if isinstance(summary, str) else "",
        url=url,
        source=source if isinstance(source, str) and source else "unknown",
        published=published,
    )


def fetch_articles(ticker: str, *, client: Any = None, now: datetime | None = None,
                   days: int = WIDE_LOOKBACK_DAYS) -> list[Article]:
    """Fetch company news for the last ``days`` days (one API call).

    Raises NewsFetchError if the provider call fails. Malformed rows are
    skipped, not fatal: one bad row should not cost us the other articles.
    """
    now = now or datetime.now(UTC)
    if client is None:
        client = _make_client()
    try:
        rows = client.company_news(
            ticker.strip().upper(),
            _from=(now - timedelta(days=days)).date().isoformat(),
            to=now.date().isoformat(),
        )
    except Exception as exc:
        raise NewsFetchError(
            f"Finnhub news call failed: {type(exc).__name__}: {exc}") from exc
    if not isinstance(rows, list):
        raise NewsFetchError(
            f"Finnhub news returned {type(rows).__name__}, expected a list")
    parsed = (parse_article(row) for row in rows)
    return [article for article in parsed if article is not None]


def clean_text(text: str, limit: int | None = None) -> str:
    """Strip HTML tags and entities, collapse whitespace, optionally truncate."""
    cleaned = _SPACES.sub(" ", html.unescape(_TAG.sub(" ", text))).strip()
    if limit is not None and len(cleaned) > limit:
        cleaned = cleaned[:limit].rstrip() + "..."
    return cleaned


def _fingerprint(headline: str) -> str:
    """Same story from two outlets differs in punctuation and case only."""
    return _NON_WORD.sub("", headline.lower())


def preprocess(articles: list[Article], *, now: datetime, max_age_days: int,
               max_articles: int = MAX_CANDIDATES) -> list[Article]:
    """Clean, drop old and duplicate items, keep the newest ``max_articles``."""
    cutoff = now - timedelta(days=max_age_days)
    newest_first = sorted(articles, key=lambda a: a.published, reverse=True)
    seen: set[str] = set()
    kept: list[Article] = []
    for article in newest_first:
        headline = clean_text(article.headline)
        if not headline or article.published < cutoff or article.published > now:
            continue
        key = _fingerprint(headline)
        if not key or key in seen:  # sorted newest first, so we keep the newest
            continue
        seen.add(key)
        kept.append(replace(
            article, headline=headline,
            summary=clean_text(article.summary, MAX_SUMMARY_CHARS)))
        if len(kept) == max_articles:
            break
    return kept


_COMPANY_SUFFIXES = frozenset({
    "inc", "corp", "corporation", "co", "company", "ltd", "limited", "plc",
    "holdings", "group", "the", "&", "and", "llc", "sa", "nv"})


def name_terms(ticker: str, company_name: str | None) -> tuple[str, ...]:
    """Lower-case search terms for a company: its ticker and a short name.

    "Applied Aerospace & Defense, Inc." gives "applied aerospace"; "NVIDIA
    Corporation" gives "nvidia". Used only to RANK articles, never to drop them.
    """
    terms = [ticker.strip().lower()]
    words = [w for w in re.split(r"[^a-z0-9]+", (company_name or "").lower())
             if w and w not in _COMPANY_SUFFIXES]
    if words:
        terms.append(" ".join(words[:2]))
    return tuple(dict.fromkeys(t for t in terms if t))


def mentions_company(article: Article, terms: tuple[str, ...]) -> bool:
    """True if the headline or summary names the company or its ticker."""
    text = f"{article.headline} {article.summary}".lower()
    return any(re.search(rf"\b{re.escape(term)}\b", text) for term in terms)


def rank_candidates(articles: list[Article],
                    terms: tuple[str, ...]) -> list[Article]:
    """Articles naming the company first; newest-first order kept within groups.

    This is a stable sort, so it only moves likely-relevant items forward. A
    market story that never names the company is still a candidate.
    """
    return sorted(articles, key=lambda a: not mentions_company(a, terms))


def select_articles(articles: list[Article], *, now: datetime,
                    terms: tuple[str, ...] = ()) -> NewsBatch:
    """Prefer the short window; widen once if coverage is thin."""
    short = preprocess(articles, now=now, max_age_days=LOOKBACK_DAYS)
    if len(short) >= MIN_ARTICLES:
        return NewsBatch(rank_candidates(short, terms), LOOKBACK_DAYS, widened=False)
    wide = preprocess(articles, now=now, max_age_days=WIDE_LOOKBACK_DAYS)
    if len(wide) > len(short):
        return NewsBatch(rank_candidates(wide, terms), WIDE_LOOKBACK_DAYS,
                         widened=True)
    return NewsBatch(rank_candidates(short, terms), LOOKBACK_DAYS, widened=False)


def collect_articles(ticker: str, *, client: Any = None,
                     now: datetime | None = None,
                     company_name: str | None = None) -> NewsBatch:
    """Ingest + preprocess in one call: the input to the Classify station."""
    now = now or datetime.now(UTC)
    raw = fetch_articles(ticker, client=client, now=now)
    return select_articles(raw, now=now, terms=name_terms(ticker, company_name))