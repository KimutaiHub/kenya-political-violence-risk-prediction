import streamlit as st
import plotly.graph_objects as go

from utils.ui import inject_v2_styles, top_nav, intelligence_header, section_label, panel, disclaimer, status_pill, PALETTE
from utils.data_loader import load_conflict_data, load_vulnerability_index, load_counties_geojson, get_latest_week, county_status_table
from utils.sidebar import render_risk_map_context

st.set_page_config(page_title="Risk Map — Kenya Conflict Monitor", page_icon="▥", layout="wide")
inject_v2_styles()
top_nav(active_index=4)

df = load_conflict_data()
vuln = load_vulnerability_index()
geojson = load_counties_geojson()
status = county_status_table(df)
latest_week = get_latest_week(df)

render_risk_map_context(status)

intelligence_header("COUNTY RISK MAP", f"Latest known week: {latest_week.strftime('%d %b %Y')}")

metric_choice = st.radio(
    "Colour counties by",
    ["Predicted vulnerability (model)", "Event status (recency)"],
    horizontal=True,
)

merged = vuln.merge(status, on="COUNTY", how="right")

if metric_choice == "Predicted vulnerability (model)":
    z = merged["PREDICTED_MEAN_SEVERITY"]
    colorscale = [[0.0, PALETTE["green"]], [0.5, PALETTE["orange"]], [1.0, PALETTE["red"]]]
    colorbar_title = "Predicted<br>mean severity"
    hover_extra = merged["PREDICTED_MEAN_SEVERITY"].round(3).astype(str)
    hover_label = "Predicted mean severity"
else:
    status_map = {"Dormant": 0, "Recently active": 1, "Ongoing": 2}
    z = merged["STATUS"].map(status_map)
    colorscale = [[0.0, PALETTE["panel_raised"]], [0.5, PALETTE["orange"]], [1.0, PALETTE["red"]]]
    colorbar_title = "Status"
    hover_extra = merged["STATUS"]
    hover_label = "Status"

with panel():
    fig = go.Figure(go.Choropleth(
        geojson=geojson, featureidkey="properties.COUNTY", locations=merged["COUNTY"], z=z,
        colorscale=colorscale, marker_line_color=PALETTE["panel"], marker_line_width=0.8,
        colorbar=dict(title=colorbar_title, tickfont=dict(color=PALETTE["muted"]), title_font=dict(color=PALETTE["muted"])),
        customdata=hover_extra,
        hovertemplate="<b>%{location}</b><br>" + hover_label + ": %{customdata}<extra></extra>",
    ))
    fig.update_geos(visible=False, fitbounds="locations", bgcolor=PALETTE["panel"])
    fig.update_layout(
        paper_bgcolor=PALETTE["panel"], plot_bgcolor=PALETTE["panel"],
        margin=dict(l=0, r=0, t=10, b=0), height=560, font=dict(color=PALETTE["text"], family="Inter"),
    )
    st.plotly_chart(fig, use_container_width=True)

st.write("")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(status_pill("High") + " &nbsp; event logged this week (Ongoing)", unsafe_allow_html=True)
with c2:
    st.markdown(status_pill("Moderate") + " &nbsp; quiet this week, active in past 4 (Recently active)", unsafe_allow_html=True)
with c3:
    st.markdown(status_pill("Low") + " &nbsp; no events in past 4 weeks (Dormant)", unsafe_allow_html=True)

st.write("")
section_label("County Status Table")
status_filter = st.multiselect("Filter by status", ["Ongoing", "Recently active", "Dormant"], default=["Ongoing", "Recently active"])
display_df = merged[merged["STATUS"].isin(status_filter)].sort_values(["STATUS", "PREDICTED_MEAN_SEVERITY"], ascending=[True, False])

with panel():
    st.dataframe(
        display_df[["COUNTY", "STATUS", "EVENTS_THIS_WEEK", "FATALITIES_THIS_WEEK", "EVENTS_RECENT_4W", "FATALITIES_RECENT_4W", "PREDICTED_MEAN_SEVERITY", "RANK"]]
        .rename(columns={
            "EVENTS_THIS_WEEK": "Events (this wk)", "FATALITIES_THIS_WEEK": "Fatalities (this wk)",
            "EVENTS_RECENT_4W": "Events (4wk)", "FATALITIES_RECENT_4W": "Fatalities (4wk)",
            "PREDICTED_MEAN_SEVERITY": "Predicted severity", "RANK": "Vulnerability rank",
        }),
        hide_index=True, use_container_width=True,
    )

st.write("")
disclaimer(
    "\"Predicted vulnerability\" reflects standing risk from historical patterns, not a forecast tied "
    "to this week's specific events. \"Event status\" is a simple recency read of the raw data. Open a "
    "county's page in <b>County Intelligence</b> for a full assessment."
)
