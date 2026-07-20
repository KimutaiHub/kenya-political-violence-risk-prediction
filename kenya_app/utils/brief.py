"""
Short, templated situation-brief text. Deliberately rule-based rather than
LLM-generated - the point is a fast, deterministic, auditable summary of
numbers already on screen, not new claims.
"""

import pandas as pd


def national_brief(vuln: pd.DataFrame, status: pd.DataFrame) -> str:
    top = vuln.sort_values("PREDICTED_TOTAL_SEVERITY", ascending=False).iloc[0]
    ongoing = int((status["STATUS"] == "Ongoing").sum())
    recently_active = int((status["STATUS"] == "Recently active").sum())
    dormant = 47 - ongoing - recently_active

    return (
        f"**{top['COUNTY']}** currently holds the highest predicted severity ranking nationally "
        f"(rank 1 of 47). **{ongoing} counties** have events logged in the most recent week; "
        f"**{recently_active} additional counties** were active within the past four weeks but quiet "
        f"this week. The remaining **{dormant} counties** show no recent activity. This reflects "
        f"standing risk based on historical patterns, not a forecast of the coming week."
    )


def county_brief(df: pd.DataFrame, vuln: pd.DataFrame, status: pd.DataFrame, county: str) -> str:
    vuln_row = vuln[vuln["COUNTY"] == county].iloc[0]
    status_row = status[status["COUNTY"] == county].iloc[0]
    rank = int(vuln_row["RANK"])

    total_fatalities = int(df[df["COUNTY"] == county]["FATALITIES"].sum())
    total_events = int(df[df["COUNTY"] == county]["EVENTS"].sum())

    rank_desc = "highest" if rank <= 5 else ("elevated" if rank <= 15 else ("moderate" if rank <= 30 else "lower"))

    status_text = {
        "Ongoing": "an event was logged in the county this week",
        "Recently active": "no event this week, but activity within the past four weeks",
        "Dormant": "no recorded activity in the past four weeks",
    }.get(status_row["STATUS"], "status unavailable")

    return (
        f"**{county}** ranks **#{rank} of 47** counties by predicted severity — a **{rank_desc}** "
        f"position nationally. Historically, the county has recorded **{total_events:,} events** and "
        f"**{total_fatalities:,} fatalities** across the full dataset. Currently, {status_text}. "
        f"Predicted mean severity stands at **{vuln_row['PREDICTED_MEAN_SEVERITY']:.2f}**."
    )
