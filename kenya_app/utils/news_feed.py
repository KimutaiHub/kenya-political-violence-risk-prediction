"""
Google News RSS - replaces the earlier GDELT integration entirely. No API
key required. Used as embedded contextual panels inside County Intelligence
and Event Scorer - there is no standalone News page.
"""

from datetime import datetime, timedelta
from urllib.parse import quote_plus

import streamlit as st

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False


def _run_query(query: str, limit: int, days: int) -> list:
    scoped_query = f"{query} when:{days}d"
    url = f"https://news.google.com/rss/search?q={quote_plus(scoped_query)}&hl=en-KE&gl=KE&ceid=KE:en"
    try:
        feed = feedparser.parse(url)
    except Exception:
        return []
    if getattr(feed, "bozo", False) and not getattr(feed, "entries", None):
        return []
    return feed.entries[:limit]


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_kenya_conflict_news(county: str = None, limit: int = 8, days: int = 7) -> list:
    """
    Fetch recent Kenya conflict-related headlines, optionally scoped to a
    county and to the last `days` days. Returns [] on any failure rather
    than raising - callers should treat an empty list as "show a friendly
    empty state", not an error state, since a quiet news day/week is a
    normal, valid outcome.

    Tries a conflict-keyword-scoped query first; if that comes back empty
    (common for smaller counties where local coverage doesn't use exactly
    those words), falls back to a plain county-name query, widening the
    time window slightly since a looser query benefits from more room.
    """
    if not FEEDPARSER_AVAILABLE:
        return []

    keywords = "(protests OR violence OR unrest OR attack OR clash OR riot OR killed OR chaos)"
    primary_query = f'"{county}" {keywords}' if county else f"Kenya {keywords}"

    entries = _run_query(primary_query, limit, days)

    if not entries and county:
        entries = _run_query(f'"{county}" Kenya', limit, days * 2)

    cutoff = datetime.utcnow() - timedelta(days=days)
    articles = []
    for entry in entries:
        published = ""
        published_dt = None
        if getattr(entry, "published_parsed", None):
            try:
                published_dt = datetime(*entry.published_parsed[:6])
                published = published_dt.strftime("%d %b %Y")
            except (TypeError, ValueError):
                published_dt = None

        if published_dt is not None and published_dt < cutoff:
            continue

        source = "Google News"
        if isinstance(getattr(entry, "source", None), dict):
            source = entry.source.get("title", "Google News")
        elif hasattr(entry, "source"):
            source = getattr(entry.source, "title", "Google News")

        articles.append({
            "title": getattr(entry, "title", "(untitled)"),
            "link": getattr(entry, "link", "#"),
            "source": source,
            "published": published,
        })

    return articles


def news_connection_ok() -> bool:
    """Lightweight connectivity check for the System Status page."""
    try:
        fetch_kenya_conflict_news(limit=1)
        return True
    except Exception:
        return False
