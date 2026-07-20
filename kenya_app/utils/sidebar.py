"""
Dynamic left-sidebar content, contextual per page. Deliberately data-driven
(computed from what's actually loaded, not static copy) so it can't drift
out of sync with the rest of the app.

Not shown on Overview or System Status - both already show everything
relevant to them in the main content area, so a sidebar there would just
duplicate the page rather than add context.
"""

import streamlit as st

from utils.ui import PALETTE, status_pill


def _mini_stat(label: str, value):
    st.markdown(
        f"<div class='v2-title' style='margin-bottom:0.1rem;'>{label}</div>"
        f"<div class='v2-value' style='font-size:1.3rem;'>{value}</div>",
        unsafe_allow_html=True,
    )


def render_county_intelligence_context(df, vuln, status, county: str):
    with st.sidebar:
        st.markdown('<div class="v2-section-label" style="margin-top:0;">SELECTED COUNTY</div>', unsafe_allow_html=True)
        _mini_stat("County", county)
        st.write("")

        vuln_row = vuln[vuln["COUNTY"] == county].iloc[0]
        status_row = status[status["COUNTY"] == county].iloc[0]
        _mini_stat("National Rank", f"#{int(vuln_row['RANK'])} / 47")
        st.write("")
        st.markdown(status_pill(status_row["STATUS"]), unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="v2-section-label">TOP 3 ALERTS</div>', unsafe_allow_html=True)
        top3 = vuln.sort_values("PREDICTED_TOTAL_SEVERITY", ascending=False).head(3)
        for _, row in top3.iterrows():
            marker = "→ " if row["COUNTY"] == county else "  "
            st.markdown(f"{marker}**{row['COUNTY']}** (#{int(row['RANK'])})")


def render_event_scorer_context(df, vuln):
    with st.sidebar:
        st.markdown('<div class="v2-section-label" style="margin-top:0;">DATASET COVERAGE</div>', unsafe_allow_html=True)
        _mini_stat("Counties Available", df["COUNTY"].nunique())
        st.write("")
        _mini_stat("Latest Known Week", df["WEEK"].max().strftime("%d %b %Y"))

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="v2-section-label">HIGHEST STANDING RISK</div>', unsafe_allow_html=True)
        top3 = vuln.sort_values("PREDICTED_TOTAL_SEVERITY", ascending=False).head(3)
        for _, row in top3.iterrows():
            st.markdown(f"**{row['COUNTY']}** — mean severity {row['PREDICTED_MEAN_SEVERITY']:.2f}")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.caption(
            "Tip: population and lag/rolling features are auto-filled from the county's most "
            "recent known data - only the fields shown in the form are yours to set."
        )


def render_historical_analytics_context(df, year_range=None):
    with st.sidebar:
        st.markdown('<div class="v2-section-label" style="margin-top:0;">DATASET SUMMARY</div>', unsafe_allow_html=True)
        _mini_stat("Total Rows", f"{len(df):,}")
        st.write("")
        _mini_stat("Year Range", f"{int(df['YEAR'].min())}–{int(df['YEAR'].max())}")
        st.write("")
        _mini_stat("Total Fatalities", f"{int(df['FATALITIES'].sum()):,}")

        if year_range:
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown('<div class="v2-section-label">CURRENT WINDOW</div>', unsafe_allow_html=True)
            windowed = df[(df["YEAR"] >= year_range[0]) & (df["YEAR"] <= year_range[1])]
            _mini_stat("Rows in Window", f"{len(windowed):,}")
            st.write("")
            _mini_stat("Fatalities in Window", f"{int(windowed['FATALITIES'].sum()):,}")


def render_risk_map_context(status):
    with st.sidebar:
        st.markdown('<div class="v2-section-label" style="margin-top:0;">STATUS BREAKDOWN</div>', unsafe_allow_html=True)
        for level, label in [("Ongoing", "Ongoing"), ("Recently active", "Recently Active"), ("Dormant", "Dormant")]:
            count = int((status["STATUS"] == level).sum())
            row = st.columns([1, 1])
            with row[0]:
                st.markdown(status_pill(level), unsafe_allow_html=True)
            with row[1]:
                st.markdown(f"<div class='v2-value' style='font-size:1.1rem; text-align:right;'>{count}</div>", unsafe_allow_html=True)
