"""
v2 design system - Palantir/Bloomberg/EOC-inspired intelligence dashboard.

Every function that emits HTML routes through _md(), which dedents first.
Markdown treats 4+ leading spaces as a code block; without dedenting, any
CSS/HTML written at normal Python indentation renders as visible raw text
instead of being applied. This bit the project once already - don't
reintroduce it by adding a new st.markdown(f\"\"\"...\"\"\", unsafe_allow_html=True)
call anywhere without going through _md() or one of the helpers below.
"""

import textwrap

import streamlit as st

PALETTE = {
    "bg": "#111827",
    "panel": "#1B2430",
    "panel_raised": "#212B3A",
    "border": "#374151",
    "text": "#E5E7EB",
    "muted": "#9CA3AF",
    "faint": "#6B7280",
    "gold": "#D4B483",
    "gold_bright": "#E8C89A",
    "green": "#22C55E",
    "orange": "#F59E0B",
    "red": "#EF4444",
    "blue": "#60A5FA",
}

NAV_PAGES = [
    ("app.py", "Overview", "◆"),
    ("pages/1_County_Intelligence.py", "County Intelligence", "▣"),
    ("pages/2_Event_Scorer.py", "Predictor", "▶"),
    ("pages/3_Historical_Analytics.py", "Historical Analytics", "▤"),
    ("pages/4_Risk_Map.py", "Risk Map", "▥"),
    ("pages/5_About.py", "About", "ℹ"),
    ("pages/6_System_Status.py", "System Status", "●"),
]


def _md(html: str):
    st.markdown(textwrap.dedent(html), unsafe_allow_html=True)


def inject_v2_styles():
    css = f"""
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
    <style>
    html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, sans-serif;
    }}
    .stApp {{
    background: {PALETTE['bg']};
    color: {PALETTE['text']};
    }}
    h1, h2, h3 {{
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    color: {PALETTE['text']} !important;
    }}
    [data-testid="stSidebar"] {{
    background: {PALETTE['panel']};
    border-right: 1px solid {PALETTE['border']};
    }}
    [data-testid="stSidebarNav"] {{
    display: none;
    }}
    [data-testid="stMetricValue"] {{
    font-family: 'IBM Plex Mono', monospace !important;
    }}
    .block-container {{
    padding-top: 3.5rem;
    max-width: 1280px;
    }}
    hr {{
    border: none;
    border-top: 1px solid {PALETTE['border']};
    margin: 1rem 0;
    }}
    .v2-panel {{
    background: {PALETTE['panel']};
    border: 1px solid {PALETTE['border']};
    border-radius: 12px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.9rem;
    }}
    .v2-title {{
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: {PALETTE['muted']};
    margin-bottom: 0.5rem;
    font-weight: 700;
    font-family: 'IBM Plex Mono', monospace;
    }}
    .v2-value {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem;
    font-weight: 500;
    color: {PALETTE['text']};
    line-height: 1;
    font-variant-numeric: tabular-nums;
    }}
    .v2-sub {{
    color: {PALETTE['muted']};
    font-size: 0.8rem;
    margin-top: 0.4rem;
    }}
    .v2-header {{
    padding: 1.1rem 1.3rem;
    border-radius: 14px;
    background: linear-gradient(135deg, {PALETTE['panel']} 0%, {PALETTE['panel_raised']} 100%);
    border: 1px solid {PALETTE['border']};
    margin-bottom: 1rem;
    }}
    .v2-header h1 {{
    margin: 0;
    font-size: 1.7rem;
    color: {PALETTE['text']};
    }}
    .v2-header p {{
    margin: 0.35rem 0 0 0;
    color: {PALETTE['muted']};
    font-size: 0.88rem;
    }}
    .v2-section-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.74rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: {PALETTE['gold']};
    font-weight: 700;
    margin: 1.1rem 0 0.6rem 0;
    }}
    .status-pill {{
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    font-family: 'IBM Plex Mono', monospace;
    }}
    .status-high {{
    background: rgba(239,68,68,0.15);
    color: {PALETTE['red']};
    border: 1px solid rgba(239,68,68,0.35);
    }}
    .status-moderate {{
    background: rgba(245,158,11,0.15);
    color: {PALETTE['orange']};
    border: 1px solid rgba(245,158,11,0.35);
    }}
    .status-low {{
    background: rgba(34,197,94,0.15);
    color: {PALETTE['green']};
    border: 1px solid rgba(34,197,94,0.35);
    }}
    .status-neutral {{
    background: rgba(156,163,175,0.15);
    color: {PALETTE['muted']};
    border: 1px solid rgba(156,163,175,0.35);
    }}
    .v2-disclaimer {{
    border-left: 2px solid {PALETTE['gold']};
    padding: 0.6rem 0.9rem;
    background: rgba(212,180,131,0.06);
    font-size: 0.83rem;
    color: {PALETTE['muted']};
    border-radius: 0 8px 8px 0;
    }}
    .v2-newsitem {{
    background: {PALETTE['panel_raised']};
    border: 1px solid {PALETTE['border']};
    border-radius: 8px;
    padding: 0.6rem 0.75rem;
    margin-bottom: 0.5rem;
    transition: border-color 0.15s ease;
    }}
    .v2-newsitem:last-child {{
    margin-bottom: 0;
    }}
    .v2-newsitem:hover {{
    border-color: {PALETTE['gold']};
    }}
    .v2-newsitem a {{
    color: {PALETTE['text']};
    text-decoration: none;
    font-weight: 500;
    font-size: 0.88rem;
    }}
    .v2-newsitem a:hover {{
    color: {PALETTE['gold']};
    }}
    .v2-newsmeta {{
    color: {PALETTE['faint']};
    font-size: 0.74rem;
    margin-top: 0.3rem;
    }}
    [data-testid="stPageLink"] {{
    background: {PALETTE['panel']};
    border: 1px solid {PALETTE['border']};
    border-radius: 8px;
    padding: 0.3rem 0.4rem !important;
    min-height: 2.6rem;
    display: flex !important;
    align-items: center;
    justify-content: center;
    }}
    [data-testid="stPageLink"]:hover {{
    border-color: {PALETTE['gold']};
    }}
    [data-testid="stPageLink"] p {{
    font-family: 'Inter', sans-serif !important;
    font-size: 0.72rem !important;
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: unset !important;
    text-align: center;
    line-height: 1.15;
    }}
    .stButton > button {{
    border: 1px solid {PALETTE['border']};
    background: {PALETTE['panel_raised']};
    color: {PALETTE['text']};
    border-radius: 8px;
    }}
    .stButton > button:hover {{
    border-color: {PALETTE['gold']};
    color: {PALETTE['gold']};
    }}
    </style>
    """
    _md(css)


def top_nav(active_index: int = 0):
    cols = st.columns(len(NAV_PAGES))
    for col, (target, label, icon) in zip(cols, NAV_PAGES):
        with col:
            try:
                st.page_link(target, label=f"{icon} {label}", use_container_width=True)
            except Exception:
                st.markdown(f"**{icon} {label}**")
    _md('<hr style="margin-top:0.4rem;">')


def intelligence_header(title: str, subtitle: str):
    _md(f"""
    <div class="v2-header">
    <h1>{title}</h1>
    <p>{subtitle}</p>
    </div>
    """)


def section_label(text: str):
    _md(f'<div class="v2-section-label">{text}</div>')


def metric_panel(title: str, value, delta: str = "", delta_color: str = "normal"):
    arrow = ""
    color = PALETTE["muted"]
    if delta:
        if delta.startswith("-"):
            arrow = "▼ "
            color = PALETTE["red"] if delta_color == "inverse" else PALETTE["green"]
        else:
            arrow = "▲ "
            color = PALETTE["green"] if delta_color == "normal" else PALETTE["red"]
    _md(f"""
    <div class="v2-panel">
    <div class="v2-title">{title}</div>
    <div class="v2-value">{value}</div>
    <div class="v2-sub" style="color:{color};">{arrow}{delta}</div>
    </div>
    """)


def status_pill(level: str) -> str:
    """Returns the pill HTML (caller embeds it inside a larger block) rather
    than rendering directly, so it can be composed inside panels."""
    level_lower = level.lower()
    cls = "status-low"
    if level_lower in ("high", "severe", "critical", "ongoing", "offline", "error"):
        cls = "status-high"
    elif level_lower in ("moderate", "medium", "degraded", "recently active"):
        cls = "status-moderate"
    elif level_lower in ("low", "dormant", "online", "ok", "healthy", "connected"):
        cls = "status-low"
    else:
        cls = "status-neutral"
    return f'<span class="status-pill {cls}">{level.upper()}</span>'


def render_status_pill(level: str):
    _md(status_pill(level))


def disclaimer(text: str):
    _md(f'<div class="v2-disclaimer">{text}</div>')


class panel:
    """Context manager for a v2-panel block, so pages don't have to hand-open
    and hand-close raw <div> markdown calls (fragile - easy to mismatch)."""

    def __init__(self, title: str = None):
        self.title = title

    def __enter__(self):
        _md('<div class="v2-panel">')
        if self.title:
            _md(f'<div class="v2-title">{self.title}</div>')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _md("</div>")
        return False


def news_list(articles: list, empty_message: str = "No recent headlines found."):
    if not articles:
        st.caption(empty_message)
        return
    for art in articles:
        meta = art.get("source", "")
        if art.get("published"):
            meta += f" · {art['published']}"
        _md(f"""
        <div class="v2-newsitem">
        <a href="{art.get('link', '#')}" target="_blank">{art.get('title', '(untitled)')}</a>
        <div class="v2-newsmeta">{meta}</div>
        </div>
        """)
