import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.ui import inject_v2_styles, top_nav, intelligence_header, section_label, panel, disclaimer, PALETTE
from utils.data_loader import load_conflict_data
from utils.sidebar import render_historical_analytics_context

st.set_page_config(page_title="Historical Analytics — Kenya Conflict Monitor", page_icon="▤", layout="wide")
inject_v2_styles()
top_nav(active_index=3)

df = load_conflict_data()

intelligence_header("HISTORICAL ANALYTICS", "Long-run patterns across counties, time, and event types")

# ---- Timeline slider -------------------------------------------------------
section_label("Time Window")
min_year, max_year = int(df["YEAR"].min()), int(df["YEAR"].max())
with panel():
    year_range = st.slider("Select year range", min_year, max_year, (max(min_year, max_year - 10), max_year))

windowed = df[(df["YEAR"] >= year_range[0]) & (df["YEAR"] <= year_range[1])]

render_historical_analytics_context(df, year_range)

# ---- National trend ---------------------------------------------------
section_label("National Trend")
with panel():
    weekly = windowed.groupby("WEEK")[["EVENTS", "FATALITIES"]].sum().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly["WEEK"], y=weekly["EVENTS"], name="Events", line=dict(color=PALETTE["blue"], width=1.2)))
    fig.add_trace(go.Scatter(x=weekly["WEEK"], y=weekly["FATALITIES"], name="Fatalities", line=dict(color=PALETTE["red"], width=1.2)))
    fig.update_layout(
        template="plotly_dark", height=320, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=PALETTE["panel"], plot_bgcolor=PALETTE["panel"],
        legend=dict(orientation="h", yanchor="bottom", y=1.02), font=dict(family="Inter"),
    )
    st.plotly_chart(fig, use_container_width=True)

# ---- Heatmap: county x year -------------------------------------------
section_label("County × Year Heatmap")
with panel():
    metric = st.radio("Metric", ["FATALITIES", "EVENTS"], horizontal=True, key="heatmap_metric")
    pivot = windowed.pivot_table(index="COUNTY", columns="YEAR", values=metric, aggfunc="sum", fill_value=0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    fig = px.imshow(
        pivot, aspect="auto", color_continuous_scale=[PALETTE["panel_raised"], PALETTE["gold"], PALETTE["red"]],
        labels=dict(x="Year", y="County", color=metric.title()),
    )
    fig.update_layout(
        template="plotly_dark", height=900, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=PALETTE["panel"], plot_bgcolor=PALETTE["panel"], font=dict(family="Inter", size=10),
    )
    st.plotly_chart(fig, use_container_width=True)

# ---- County comparison --------------------------------------------------
section_label("County Comparison")
with panel():
    all_counties = sorted(df["COUNTY"].unique())
    default_compare = [c for c in ["Nairobi", "Turkana", "Mandera"] if c in all_counties]
    compare_counties = st.multiselect("Compare counties", all_counties, default=default_compare)
    compare_metric = st.radio("Compare by", ["FATALITIES", "EVENTS"], horizontal=True, key="compare_metric")

    if compare_counties:
        comp_data = windowed[windowed["COUNTY"].isin(compare_counties)]
        comp_weekly = comp_data.groupby(["COUNTY", "WEEK"])[compare_metric].sum().reset_index()
        fig = px.line(
            comp_weekly, x="WEEK", y=compare_metric, color="COUNTY",
            color_discrete_sequence=[PALETTE["gold"], PALETTE["blue"], PALETTE["red"], PALETTE["green"], PALETTE["orange"]],
        )
        fig.update_layout(
            template="plotly_dark", height=340, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor=PALETTE["panel"], plot_bgcolor=PALETTE["panel"], font=dict(family="Inter"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Select at least one county to compare.")

st.write("")
disclaimer(
    "All figures on this page reflect historical, already-observed data for the selected time window - "
    "nothing here is a model prediction."
)
