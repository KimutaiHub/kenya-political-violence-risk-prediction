"""
Central data-access layer. Every page reads through here so there is exactly
one place that knows about file paths and caching.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

APP_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = APP_ROOT / "data"
MODELS_DIR = APP_ROOT / "models"

LATEST_KNOWN_WEEK_LOOKBACK = 1  # weeks considered "current" for the ongoing/recent toggle
RECENT_WINDOW_WEEKS = 4          # matches the model's own EVENTS_ROLLING_4W feature window


@st.cache_data(show_spinner=False)
def load_conflict_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / "kenya_conflict_full.parquet")
    df["WEEK"] = pd.to_datetime(df["WEEK"])
    return df


@st.cache_data(show_spinner=False)
def load_vulnerability_index() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "county_vulnerability_index.csv")


@st.cache_data(show_spinner=False)
def load_vulnerability_stats() -> dict:
    return json.load(open(DATA_DIR / "vulnerability_stats.json"))


@st.cache_data(show_spinner=False)
def load_master_comparison() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "master_model_comparison.csv")


@st.cache_data(show_spinner=False)
def load_counties_geojson() -> dict:
    return json.load(open(DATA_DIR / "kenya_counties.geojson"))


from utils.model_loader import load_all_models, all_models_ok


@st.cache_resource(show_spinner=False)
def load_models():
    """
    Returns (stage1_model, stage2_model, stage2_label_encoder, feature_lists, load_ok, errors).
    load_ok is False if ANY model file failed to load (e.g. a numpy/sklearn
    version mismatch) - callers should check this before using the models,
    rather than letting a raw exception surface mid-page.
    """
    loaded = load_all_models(str(MODELS_DIR))
    errors = {name: r.error for name, r in loaded.items() if not r.ok}
    ok = all_models_ok(loaded)
    return (
        loaded["stage1"].model,
        loaded["stage2"].model,
        loaded["label_encoder"].model,
        loaded["feature_lists"].model,
        ok,
        errors,
    )


def week_over_week_delta(df: pd.DataFrame, column: str, county: str = None) -> tuple:
    """
    Real (not placeholder) week-over-week change for a metric, comparing the
    latest known week to the week before it. Returns (delta_value, pct_text).
    """
    data = df if county is None else df[df["COUNTY"] == county]
    weekly = data.groupby("WEEK")[column].sum().sort_index()
    if len(weekly) < 2:
        return 0, ""
    latest, prior = weekly.iloc[-1], weekly.iloc[-2]
    delta = latest - prior
    if prior == 0:
        pct_text = f"{'+' if delta >= 0 else ''}{int(delta)} vs prior week"
    else:
        pct = (delta / prior) * 100
        pct_text = f"{'+' if pct >= 0 else ''}{pct:.0f}% vs prior week"
    return delta, pct_text


def get_latest_week(df: pd.DataFrame) -> pd.Timestamp:
    return df["WEEK"].max()


def county_status_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every county, classify current status relative to the latest known week:
    - Ongoing: an event was logged in the latest week
    - Recently active: no event this week, but at least one within the past
      RECENT_WINDOW_WEEKS (reuses the same window as EVENTS_ROLLING_4W)
    - Dormant: nothing in the recent window
    """
    latest_week = get_latest_week(df)
    cutoff = latest_week - pd.Timedelta(weeks=RECENT_WINDOW_WEEKS - 1)

    counties = sorted(df["COUNTY"].unique())
    rows = []
    for county in counties:
        cdf = df[df["COUNTY"] == county]
        this_week = cdf[cdf["WEEK"] == latest_week]
        recent = cdf[(cdf["WEEK"] >= cutoff) & (cdf["WEEK"] <= latest_week)]

        events_this_week = int(this_week["EVENTS"].sum())
        fatalities_this_week = int(this_week["FATALITIES"].sum())
        events_recent = int(recent["EVENTS"].sum())
        fatalities_recent = int(recent["FATALITIES"].sum())

        if events_this_week > 0:
            status = "Ongoing"
        elif events_recent > 0:
            status = "Recently active"
        else:
            status = "Dormant"

        rows.append({
            "COUNTY": county,
            "STATUS": status,
            "EVENTS_THIS_WEEK": events_this_week,
            "FATALITIES_THIS_WEEK": fatalities_this_week,
            "EVENTS_RECENT_4W": events_recent,
            "FATALITIES_RECENT_4W": fatalities_recent,
        })

    return pd.DataFrame(rows)
