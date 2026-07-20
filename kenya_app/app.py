import streamlit as st
import plotly.express as px

from utils.ui import (
    inject_v2_styles, top_nav, intelligence_header, metric_panel,
    section_label, status_pill, panel, disclaimer, PALETTE,
)
from utils.data_loader import (
    load_conflict_data, load_vulnerability_index, load_vulnerability_stats,
    load_models, get_latest_week, county_status_table, week_over_week_delta,
)
from utils.brief import national_brief
from utils.recommendations import severity_level_from_percentile

st.set_page_config(
    page_title="Kenya Conflict Intelligence System",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_v2_styles()
top_nav(active_index=0)

df = load_conflict_data()
vuln = load_vulnerability_index()
stats = load_vulnerability_stats()
status = county_status_table(df)
latest_week = get_latest_week(df)
_, _, _, _, models_ok, model_errors = load_models()

intelligence_header(
    "KENYA CONFLICT INTELLIGENCE SYSTEM",
    f"Operational decision-support dashboard · ACLED data through {latest_week.strftime('%d %b %Y')}"
)

if not models_ok:
    st.error(
        "One or more prediction models failed to load - Event Scorer and SHAP explanations will be "
        "unavailable until this is resolved. See the System Status page for details."
    )

section_label("National Overview")

col1, col2, col3, col4 = st.columns(4)

high_risk_count = int((vuln["PREDICTED_TOTAL_SEVERITY"] > vuln["PREDICTED_TOTAL_SEVERITY"].quantile(0.8)).sum())
ongoing = int((status["STATUS"] == "Ongoing").sum())
_, events_delta_text = week_over_week_delta(df, "EVENTS")
_, fatalities_delta_text = week_over_week_delta(df, "FATALITIES")

with col1:
    metric_panel("High Risk Counties", high_risk_count, "top quintile nationally")
with col2:
    metric_panel("Ongoing Events", ongoing, f"of 47 counties")
with col3:
    metric_panel("Events, Latest Week", int(status["EVENTS_THIS_WEEK"].sum()), events_delta_text)
with col4:
    metric_panel("Model Agreement (ρ)", f"{stats['rho']:.3f}", "predicted vs. actual rank")

section_label("Risk Intelligence")

left, right = st.columns([1.6, 1])

with left:
    with panel("Top Vulnerable Counties"):
        top5 = vuln.sort_values("PREDICTED_TOTAL_SEVERITY", ascending=False).head(5)
        fig = px.bar(
            top5, x="PREDICTED_TOTAL_SEVERITY", y="COUNTY", orientation="h",
            text="PREDICTED_TOTAL_SEVERITY", color="PREDICTED_TOTAL_SEVERITY",
            color_continuous_scale=[PALETTE["panel_raised"], PALETTE["gold"]],
        )
        fig.update_layout(
            template="plotly_dark", height=300, margin=dict(l=10, r=10, t=10, b=10),
            coloraxis_showscale=False, paper_bgcolor=PALETTE["panel"], plot_bgcolor=PALETTE["panel"],
            font=dict(family="Inter"),
        )
        fig.update_yaxes(categoryorder="total ascending")
        fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

with right:
    with panel("National Intelligence Brief"):
        st.markdown(national_brief(vuln, status))

st.write("")
section_label("Recent / Ongoing Activity")

with panel():
    recent = status.sort_values(["STATUS", "EVENTS_THIS_WEEK"], ascending=[True, False])
    recent = recent[recent["STATUS"] != "Dormant"].head(12)
    st.dataframe(
        recent[["COUNTY", "STATUS", "EVENTS_THIS_WEEK", "FATALITIES_THIS_WEEK", "EVENTS_RECENT_4W"]]
        .rename(columns={
            "EVENTS_THIS_WEEK": "Events (this wk)", "FATALITIES_THIS_WEEK": "Fatalities (this wk)",
            "EVENTS_RECENT_4W": "Events (4wk)",
        }),
        use_container_width=True, hide_index=True,
    )

st.write("")
disclaimer(
    "This is a decision-support tool built on historical pattern recognition. It estimates relative "
    "risk, it does not predict specific events. Open <b>County Intelligence</b> for a full per-county "
    "assessment, or the <b>Event Scorer</b> to score a hypothetical or real incident."
)
