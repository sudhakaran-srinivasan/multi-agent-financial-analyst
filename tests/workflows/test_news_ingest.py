"""Offline tests for news ingest + preprocess (no network).

Definition of "this works":
    1. Real Finnhub rows parse into Articles; malformed rows are skipped.
    2. A failed provider call raises NewsFetchError (never returns []).
    3. Preprocess strips HTML, drops duplicates (keeping the newest), drops
       stale and future-dated items, sorts newest first and caps the count.
    4. The short window is used when coverage is fine; it widens once, and
       says so, when coverage is thin; no news at all gives an empty batch.
"""
from datetime import UTC, datetime, timedelta

import pytest

from finagent.workflows.news_ingest import (
    LOOKBACK_DAYS,
    MAX_CANDIDATES,
    WIDE_LOOKBACK_DAYS,
    Article,
    NewsFetchError,
    clean_text,
    collect_articles,
    fetch_articles,
    mentions_company,
    name_terms,
    parse_article,
    preprocess,
    rank_candidates,
    select_articles,
)

NOW = datetime(2026, 10, 13, 22, 0, tzinfo=UTC)


def stamp(days_ago: float) -> int:
    return int((NOW - timedelta(days=days_ago)).timestamp())


def row(headline="Nvidia unveils new chip", days_ago=1, **extra):
    base = {"category": "company", "datetime": stamp(days_ago),
            "headline": headline, "related": "NVDA", "source": "Yahoo",
            "summary": "A short summary.", "url": "https://finnhub.io/api/news?id=1"}
    return {**base, **extra}


def article(headline="Nvidia unveils new chip", days_ago=1, summary="s"):
    return Article(headline, summary, "https://x/1", "Yahoo",
                   NOW - timedelta(days=days_ago))


class FakeClient:
    def __init__(self, rows=None, error=None):
        self.rows, self.error, self.calls = rows or [], error, []

    def company_news(self, symbol, _from, to):
        self.calls.append((symbol, _from, to))
        if self.error:
            raise self.error
        return self.rows


# ---- parsing -----------------------------------------------------------------
def test_a_real_finnhub_row_becomes_an_article():
    real = {"category": "company", "datetime": 1791927301,
            "headline": "3 Absurdly Cheap Stocks to Buy Before the End of 2026",
            "id": 142854315, "image": "https://s.yimg.com/x.png", "related": "NVDA",
            "source": "Yahoo", "summary": "Priced incredibly low.",
            "url": "https://finnhub.io/api/news?id=0225"}
    a = parse_article(real)
    assert a.headline.startswith("3 Absurdly") and a.source == "Yahoo"
    assert a.published.tzinfo is UTC and a.published.year == 2026


@pytest.mark.parametrize("bad", [
    None, "text", {}, row(headline=""), row(headline=None), row(url=""),
    row(datetime=None), row(datetime="yesterday"), row(datetime=True),
    row(datetime=1e30)])
def test_unusable_rows_become_none(bad):
    assert parse_article(bad) is None


def test_a_missing_summary_and_source_get_safe_defaults():
    a = parse_article(row(summary=None, source=""))
    assert a.summary == "" and a.source == "unknown"


# ---- fetching ----------------------------------------------------------------
def test_fetch_asks_for_the_ticker_and_date_range_in_one_call():
    client = FakeClient([row()])
    out = fetch_articles("nvda", client=client, now=NOW)
    assert len(out) == 1
    assert client.calls == [("NVDA", "2026-07-15", "2026-10-13")]


def test_fetch_skips_bad_rows_but_keeps_good_ones():
    out = fetch_articles("NVDA", client=FakeClient([row(), {"junk": 1}, None]),
                         now=NOW)
    assert len(out) == 1


def test_a_failed_call_raises_so_the_trace_can_record_it():
    with pytest.raises(NewsFetchError, match="ConnectionError"):
        fetch_articles("NVDA", client=FakeClient(error=ConnectionError("down")),
                       now=NOW)


def test_a_non_list_reply_raises():
    with pytest.raises(NewsFetchError):
        fetch_articles("NVDA", client=FakeClient(rows={"error": "limit"}), now=NOW)


def test_no_news_is_an_empty_list_not_an_error():
    assert fetch_articles("NVDA", client=FakeClient([]), now=NOW) == []


def test_a_missing_api_key_is_a_setup_error(monkeypatch):
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    pytest.importorskip("finnhub")
    with pytest.raises(RuntimeError, match="FINNHUB_API_KEY"):
        fetch_articles("NVDA", now=NOW)


# ---- cleaning ----------------------------------------------------------------
def test_clean_text_strips_tags_entities_and_extra_space():
    assert clean_text("<p>Nvidia &amp;  AMD\n beat</p>") == "Nvidia & AMD beat"


def test_clean_text_truncates_long_text():
    assert clean_text("word " * 200, limit=20).endswith("...")
    assert len(clean_text("word " * 200, limit=20)) <= 23


# ---- preprocess --------------------------------------------------------------
def test_preprocess_drops_duplicates_and_keeps_the_newest():
    older = article("Nvidia Beats Estimates!", days_ago=3)
    newer = article("nvidia beats estimates", days_ago=1)
    out = preprocess([older, newer], now=NOW, max_age_days=14)
    assert len(out) == 1 and out[0].published == newer.published


def test_preprocess_drops_stale_and_future_items():
    items = [article("old", days_ago=20), article("fresh", days_ago=2),
             article("future", days_ago=-1)]
    out = preprocess(items, now=NOW, max_age_days=14)
    assert [a.headline for a in out] == ["fresh"]


def test_preprocess_sorts_newest_first_and_caps():
    items = [article(f"story {i}", days_ago=i / 10) for i in range(1, 40)]
    out = preprocess(items, now=NOW, max_age_days=14)
    assert len(out) == MAX_CANDIDATES
    assert out == sorted(out, key=lambda a: a.published, reverse=True)
    assert out[0].headline == "story 1"


def test_preprocess_cleans_html_in_headline_and_summary():
    out = preprocess([article("<b>Big &amp; bold</b>", summary="<i>hi</i>  there")],
                     now=NOW, max_age_days=14)
    assert out[0].headline == "Big & bold" and out[0].summary == "hi there"


def test_preprocess_drops_empty_headlines():
    assert preprocess([article("<br>")], now=NOW, max_age_days=14) == []


def test_preprocess_does_not_mutate_its_input():
    items = [article("a", 1), article("a", 2)]
    snapshot = list(items)
    preprocess(items, now=NOW, max_age_days=14)
    assert items == snapshot


# ---- window selection --------------------------------------------------------
def test_enough_recent_news_uses_the_short_window():
    items = [article(f"story {i}", days_ago=i) for i in range(1, 6)]
    batch = select_articles(items, now=NOW)
    assert batch.window_days == LOOKBACK_DAYS and not batch.widened
    assert len(batch.articles) == 5


def test_thin_coverage_widens_once_and_says_so():
    items = [article("recent", days_ago=2), article("older a", days_ago=40),
             article("older b", days_ago=70)]
    batch = select_articles(items, now=NOW)
    assert batch.widened and batch.window_days == WIDE_LOOKBACK_DAYS
    assert len(batch.articles) == 3


def test_widening_that_finds_nothing_new_is_not_flagged():
    batch = select_articles([article("recent", days_ago=2)], now=NOW)
    assert not batch.widened and len(batch.articles) == 1


def test_no_news_gives_an_empty_unwidened_batch():
    batch = select_articles([], now=NOW)
    assert batch.articles == [] and not batch.widened


def test_collect_articles_runs_fetch_then_select():
    client = FakeClient([row(f"story {i}", days_ago=i) for i in range(1, 5)])
    batch = collect_articles("NVDA", client=client, now=NOW)
    assert len(batch.articles) == 4 and len(client.calls) == 1


# ---- ranking: likely-relevant first -------------------------------------------
@pytest.mark.parametrize("ticker, name, expected", [
    ("NVDA", "NVIDIA Corporation", ("nvda", "nvidia")),
    ("AADX", "Applied Aerospace & Defense, Inc.", ("aadx", "applied aerospace")),
    ("KO", "The Coca-Cola Company", ("ko", "coca cola")),
    ("NVDA", None, ("nvda",)),
])
def test_name_terms_gives_ticker_and_short_company_name(ticker, name, expected):
    assert name_terms(ticker, name) == expected


def test_mentions_company_uses_whole_words():
    terms = ("nvda", "nvidia")
    assert mentions_company(article("Nvidia unveils chip"), terms)
    assert mentions_company(article("Chip stocks (NVDA) rise"), terms)
    assert mentions_company(article("Stocks", summary="Nvidia led gains"), terms)
    assert not mentions_company(article("Deere vs Caterpillar"), terms)
    assert not mentions_company(article("Nvidiafan blog"), terms)


def test_rank_moves_mentions_first_and_keeps_newest_first_within_groups():
    items = [article("Deere news", 1), article("Nvidia beats", 2),
             article("Fed holds rates", 3), article("NVDA rallies", 4)]
    ranked = rank_candidates(items, ("nvda", "nvidia"))
    assert [a.headline for a in ranked] == [
        "Nvidia beats", "NVDA rallies", "Deere news", "Fed holds rates"]


def test_rank_never_drops_articles():
    items = [article("Deere news", 1), article("Fed holds rates", 2)]
    assert len(rank_candidates(items, ("nvda",))) == 2


def test_collect_articles_ranks_with_the_company_name():
    client = FakeClient([row("Deere beats", 1), row("Nvidia unveils chip", 2),
                         row("Fed holds", 3)])
    batch = collect_articles("NVDA", client=client, now=NOW,
                             company_name="NVIDIA Corporation")
    assert batch.articles[0].headline == "Nvidia unveils chip"


def test_a_relevant_article_below_the_newest_noise_is_still_found():
    """Regression: capping to the newest N before ranking hid relevant items."""
    noise = [article(f"Deere story {i}", days_ago=i / 10) for i in range(1, 31)]
    relevant = article("Nvidia unveils new chip", days_ago=10)
    batch = select_articles(noise + [relevant], now=NOW, terms=("nvidia",))
    assert len(batch.articles) == MAX_CANDIDATES
    assert batch.articles[0].headline == "Nvidia unveils new chip"
