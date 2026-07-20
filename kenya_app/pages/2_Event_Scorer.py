import datetime

import plotly.graph_objects as go
import streamlit as st

from utils.ui import (
    inject_v2_styles, top_nav, intelligence_header, section_label,
    status_pill, panel, disclaimer, news_list, PALETTE,
)
from utils.data_loader import load_conflict_data, load_vulnerability_index, load_models, county_status_table
from utils.features import build_event_row, score_event, EVENT_TYPE_TAXONOMY, ELECTION_YEARS
from utils.recommendations import get_event_recommendations
from utils.news_feed import fetch_kenya_conflict_news
from utils.brief import county_brief
from utils.sidebar import render_event_scorer_context

st.set_page_config(page_title="Event Scorer — Kenya Conflict Monitor", page_icon="▶", layout="wide")
inject_v2_styles()
top_nav(active_index=2)

df = load_conflict_data()
vuln = load_vulnerability_index()
status = county_status_table(df)
stage1_model, stage2_model, stage2_label_encoder, feature_lists, models_ok, model_errors = load_models()
counties = sorted(df["COUNTY"].unique())

render_event_scorer_context(df, vuln)

intelligence_header(
    "PREDICTOR",
    "Score a hypothetical or real event by county, type, and date"
)

if not models_ok:
    st.error("Prediction models failed to load. See System Status page for details.")
    st.stop()

section_label("Inputs")
input_col, result_col = st.columns([1, 1.4])

with input_col:
    with panel():
        county = st.selectbox("County", counties, index=counties.index("Nairobi") if "Nairobi" in counties else 0)
        event_type = st.selectbox("Event type", list(EVENT_TYPE_TAXONOMY.keys()))
        sub_event_type = st.selectbox("Sub-event type", EVENT_TYPE_TAXONOMY[event_type])
        event_date = st.date_input("Event date", value=datetime.date.today())
        election_toggle = st.toggle(
            "Treat as an election-year context",
            value=(event_date.year in ELECTION_YEARS),
            help=f"Historical election years: {', '.join(str(y) for y in ELECTION_YEARS)}",
        )
        run = st.button("Score this event", type="primary", use_container_width=True)

with result_col:
    if not run:
        with panel("Prediction Summary"):
            st.caption("Fill in the event details and click Score to see a result.")
    else:
        try:
            event_row = build_event_row(df, county, event_type, sub_event_type, event_date, is_election_year_override=election_toggle)
            result = score_event(stage1_model, stage2_model, stage2_label_encoder, event_row)
        except Exception as e:
            with panel("Prediction Summary"):
                st.error(f"Scoring failed ({type(e).__name__}). See System Status page for model diagnostics.")
            st.stop()

        fatal_pct = result["fatal_probability"] * 100
        severity = result["severity_class"]

        with panel("Prediction Summary"):
            top_row = st.columns([1, 1])
            with top_row[0]:
                st.markdown(f"<div style='margin-top:0.2rem;'>{status_pill(severity)}</div>", unsafe_allow_html=True)
            with top_row[1]:
                confidence = "High" if abs(fatal_pct - 50) > 30 else ("Moderate" if abs(fatal_pct - 50) > 15 else "Low")
                st.markdown(f"<div style='text-align:right;'>Confidence: {status_pill(confidence)}</div>", unsafe_allow_html=True)

            st.markdown(
                f"<div class='v2-value' style='font-size:2.3rem; margin-top:0.6rem;'>{fatal_pct:.1f}%</div>"
                f"<div class='v2-sub'>probability of at least one fatality this county-week</div>",
                unsafe_allow_html=True,
            )

            if result["severity_probabilities"]:
                st.write("")
                probs = result["severity_probabilities"]
                order = ["Low", "Medium", "High"]
                labels = [k for k in order if k in probs]
                values = [probs[k] * 100 for k in labels]
                colors = {"Low": PALETTE["green"], "Medium": PALETTE["orange"], "High": PALETTE["red"]}
                fig = go.Figure(go.Bar(
                    x=values, y=labels, orientation="h",
                    marker_color=[colors[l] for l in labels],
                    text=[f"{v:.1f}%" for v in values], textposition="outside",
                ))
                fig.update_layout(
                    template="plotly_dark", height=160, margin=dict(l=10, r=40, t=5, b=5),
                    paper_bgcolor=PALETTE["panel"], plot_bgcolor=PALETTE["panel"],
                    xaxis=dict(range=[0, 100]), font=dict(family="Inter"),
                )
                st.plotly_chart(fig, use_container_width=True)

if run:
    section_label("Operational Assessment")
    op_col, rec_col = st.columns([1, 1])

    with op_col:
        with panel("Operational Assessment"):
            severity = result["severity_class"]
            st.markdown(
                f"An event of type **{event_type} — {sub_event_type}** in **{county}** on "
                f"**{event_date.strftime('%d %b %Y')}** is assessed as **{severity}** severity, "
                f"with a **{fatal_pct:.1f}%** predicted probability of at least one fatality."
            )
            if election_toggle:
                st.markdown("Election-year context is **active** for this assessment.")

    with rec_col:
        with panel("Recommended Actions"):
            for action in get_event_recommendations(severity):
                st.markdown(f"- {action}")

    section_label("County Context")
    ctx_left, ctx_right = st.columns([1.3, 1])

    with ctx_left:
        with panel(f"{county} - Intelligence Summary"):
            st.markdown(county_brief(df, vuln, status, county))

    with ctx_right:
        with panel("Recent News"):
            news = fetch_kenya_conflict_news(county=county, limit=5)
            news_list(news, empty_message="No recent county-specific headlines found.")

    with st.expander("What went into this prediction"):
        display_row = event_row.T.rename(columns={0: "Value"})
        display_row["Value"] = display_row["Value"].astype(str)
        st.dataframe(display_row, use_container_width=True)
        st.caption(
            "Population, density, and lag/rolling-window features are auto-filled from the county's "
            "most recent known values as of the selected date - not entered by hand."
        )

st.write("")
disclaimer(
    "This scores a single hypothetical event against patterns learned from historical data. It is not "
    "a real-time forecast and does not know about events that haven't happened yet."
)
