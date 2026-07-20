import plotly.graph_objects as go
import streamlit as st

from utils.ui import (
    inject_v2_styles, top_nav, intelligence_header, metric_panel,
    section_label, status_pill, panel, disclaimer, news_list, PALETTE,
)
from utils.data_loader import load_conflict_data, load_vulnerability_index, load_models, county_status_table
from utils.brief import county_brief
from utils.recommendations import severity_level_from_percentile, get_standing_recommendations
from utils.news_feed import fetch_kenya_conflict_news
from utils.shap_explain import build_explainer, explain_row
from utils.sidebar import render_county_intelligence_context

st.set_page_config(page_title="County Intelligence — Kenya Conflict Monitor", page_icon="▣", layout="wide")
inject_v2_styles()
top_nav(active_index=1)

df = load_conflict_data()
vuln = load_vulnerability_index()
status = county_status_table(df)
stage1_model, stage2_model, stage2_label_encoder, feature_lists, models_ok, model_errors = load_models()

counties = sorted(vuln["COUNTY"].unique())
default_idx = counties.index("Nairobi") if "Nairobi" in counties else 0

col_select, col_spacer = st.columns([1, 3])
with col_select:
    county = st.selectbox("Select county", counties, index=default_idx)

render_county_intelligence_context(df, vuln, status, county)

vuln_row = vuln[vuln["COUNTY"] == county].iloc[0]
status_row = status[status["COUNTY"] == county].iloc[0]
level = severity_level_from_percentile(vuln_row["PREDICTED_TOTAL_SEVERITY"], vuln["PREDICTED_TOTAL_SEVERITY"])

intelligence_header(
    f"{county.upper()} — COUNTY INTELLIGENCE",
    f"National rank #{int(vuln_row['RANK'])} of 47 · Status: {status_row['STATUS']}"
)

# ---- Row 1: KPIs ----------------------------------------------------------
section_label("County Overview")
c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_panel("Predicted Severity", f"{vuln_row['PREDICTED_MEAN_SEVERITY']:.2f}", "mean, test period")
with c2:
    total_fatalities = int(df[df["COUNTY"] == county]["FATALITIES"].sum())
    metric_panel("Historical Fatalities", f"{total_fatalities:,}", "full dataset")
with c3:
    total_events = int(df[df["COUNTY"] == county]["EVENTS"].sum())
    metric_panel("Historical Events", f"{total_events:,}", "full dataset")
with c4:
    with panel("Risk Level"):
        st.markdown(f"<div style='margin-top:0.2rem;'>{status_pill(level)}</div>", unsafe_allow_html=True)
        st.caption(f"Percentile-based classification vs. all 47 counties")

# ---- Row 2: Brief + Recommendations ---------------------------------------
section_label("Risk Assessment")
left, right = st.columns([1.5, 1])

with left:
    with panel("Intelligence Brief"):
        st.markdown(county_brief(df, vuln, status, county))

with right:
    with panel("Recommended Actions"):
        for action in get_standing_recommendations(level):
            st.markdown(f"- {action}")

# ---- Row 3: Historical trends -----------------------------------------
section_label("Historical Trends")
county_weekly = df[df["COUNTY"] == county].groupby("WEEK")[["EVENTS", "FATALITIES"]].sum().reset_index()

with panel():
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=county_weekly["WEEK"], y=county_weekly["EVENTS"], name="Events",
        line=dict(color=PALETTE["blue"], width=1.5),
    ))
    fig.add_trace(go.Bar(
        x=county_weekly["WEEK"], y=county_weekly["FATALITIES"], name="Fatalities",
        marker_color=PALETTE["red"], opacity=0.6,
    ))
    fig.update_layout(
        template="plotly_dark", height=320, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=PALETTE["panel"], plot_bgcolor=PALETTE["panel"],
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        font=dict(family="Inter"),
    )
    st.plotly_chart(fig, use_container_width=True)

# ---- Row 4: SHAP explanation + News ---------------------------------------
section_label("Model Explanation & Context")
exp_left, exp_right = st.columns([1.3, 1])

with exp_left:
    with panel("What Drives This County's Risk (SHAP)"):
        if not models_ok:
            st.caption("Unavailable - model failed to load. See System Status page.")
        else:
            try:
                county_history = df[df["COUNTY"] == county].sort_values("WEEK")
                representative_row = county_history.tail(1)
                background = df[df["WEEK"] < df["WEEK"].max()]

                explainer, feature_names = build_explainer(
                    stage1_model, background,
                    feature_lists["categorical_features"], feature_lists["numeric_features"],
                )
                shap_df = explain_row(
                    stage1_model, explainer, feature_names, representative_row,
                    feature_lists["categorical_features"], feature_lists["numeric_features"],
                )
                fig = go.Figure(go.Bar(
                    x=shap_df["SHAP Value"], y=shap_df["Feature"], orientation="h",
                    marker_color=[PALETTE["red"] if v > 0 else PALETTE["green"] for v in shap_df["SHAP Value"]],
                ))
                fig.update_layout(
                    template="plotly_dark", height=320, margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor=PALETTE["panel"], plot_bgcolor=PALETTE["panel"],
                    xaxis_title="Impact on fatal-event probability", font=dict(family="Inter"),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    "Based on the county's most recent recorded week. Red bars push the prediction "
                    "toward \"fatal\"; green bars push toward \"non-fatal\"."
                )
            except Exception as e:
                st.caption(f"Explanation unavailable for this county right now ({type(e).__name__}).")

with exp_right:
    with panel("Recent News"):
        news = fetch_kenya_conflict_news(county=county, limit=5)
        news_list(news, empty_message="No recent county-specific headlines found.")
        st.caption("Independent of model predictions - shown for situational awareness only.")

# ---- Row 5: County statistics ----------------------------------------
section_label("County Statistics")
with panel():
    latest = df[df["COUNTY"] == county].sort_values("WEEK").iloc[-1]
    stat_cols = st.columns(4)
    stats_to_show = [
        ("Population (est.)", f"{latest['COUNTY_POPULATION']:,.0f}"),
        ("Density (per km²)", f"{latest['COUNTY_RAW_DENSITY']:,.1f}"),
        ("Area (km²)", f"{latest['COUNTY_AREA_KM2']:,.0f}"),
        ("4-wk rolling events", f"{status_row['EVENTS_RECENT_4W']:.0f}"),
    ]
    for col, (label, value) in zip(stat_cols, stats_to_show):
        with col:
            st.metric(label, value)

st.write("")
disclaimer(
    "This page reflects standing risk from historical patterns, not a forecast of what will happen "
    "next in this county. Use the <b>Event Scorer</b> to test a specific hypothetical or real event."
)
