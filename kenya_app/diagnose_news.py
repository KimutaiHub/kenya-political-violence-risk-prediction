"""
Standalone news feed diagnostic - run this directly to see exactly what the
app's news function would return, with zero Streamlit caching involved.

Usage:
    cd kenya_app
    python diagnose_news.py
    python diagnose_news.py "Nyandarua"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import the raw, uncached function directly - not through Streamlit's
# @st.cache_data wrapper, so there's no possibility of a stale result here.
from utils.news_feed import _run_query, FEEDPARSER_AVAILABLE

print(f"feedparser available: {FEEDPARSER_AVAILABLE}")
if not FEEDPARSER_AVAILABLE:
    print("feedparser is not installed in this environment - run: pip install feedparser")
    sys.exit(1)

county = sys.argv[1] if len(sys.argv) > 1 else None
keywords = "(protests OR violence OR unrest OR attack OR clash OR riot OR killed OR chaos)"
query = f'"{county}" {keywords}' if county else f"Kenya {keywords}"

print(f"\nQuery: {query}")
print(f"Window: last 7 days\n")

entries = _run_query(query, limit=10, days=7)
print(f"Raw entries returned: {len(entries)}\n")

for e in entries:
    title = getattr(e, "title", "(untitled)")
    published = getattr(e, "published", "no date")
    link = getattr(e, "link", "no link")
    print(f"- {title}")
    print(f"  {published}")
    print(f"  {link}\n")

if not entries and county:
    print("Primary query returned nothing. Trying fallback query...")
    fallback_entries = _run_query(f'"{county}" Kenya', limit=10, days=14)
    print(f"Fallback entries: {len(fallback_entries)}")
    for e in fallback_entries:
        print(f"- {getattr(e, 'title', '(untitled)')}")
