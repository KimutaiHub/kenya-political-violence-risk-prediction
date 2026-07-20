"""
Single source of truth for turning one (county, event type, date) combination
into a model-ready feature row. This mirrors the feature engineering in the
capstone notebook exactly - if the notebook's feature set ever changes, this
is the one place to update so the app and the model stay in sync.
"""

from typing import Optional

import numpy as np
import pandas as pd

ELECTION_YEARS = [2002, 2007, 2013, 2017, 2022]

EVENT_TYPE_TAXONOMY = {
    "Battles": ["Armed clash", "Non-state actor overtakes territory", "Government regains territory"],
    "Explosions/Remote violence": [
        "Remote explosive/landmine/IED", "Air/drone strike", "Grenade",
        "Shelling/artillery/missile attack", "Suicide bomb",
    ],
    "Protests": ["Peaceful protest", "Protest with intervention", "Excessive force against protesters"],
    "Riots": ["Mob violence", "Violent demonstration"],
    "Strategic developments": [
        "Change to group/activity", "Other", "Looting/property destruction", "Arrests",
        "Agreement", "Disrupted weapons use", "Headquarters or base established",
        "Non-violent transfer of territory",
    ],
    "Violence against civilians": ["Attack", "Sexual violence", "Abduction/forced disappearance"],
}


def _latest_county_context(df: pd.DataFrame, county: str) -> dict:
    """Pull the most recent known population/geographic features for a county."""
    cdf = df[df["COUNTY"] == county].sort_values("WEEK")
    if cdf.empty:
        raise ValueError(f"No historical data for county: {county}")
    latest = cdf.iloc[-1]
    return {
        "POPULATION_EXPOSURE_MISSING": int(latest["POPULATION_EXPOSURE_MISSING"]),
        "LOG_POPULATION_EXPOSURE": float(latest["LOG_POPULATION_EXPOSURE"]),
        "LOG_PWD_POPULATION": float(latest["LOG_PWD_POPULATION"]),
        "LOG_PWD_DENSITY": float(latest["LOG_PWD_DENSITY"]),
        "LOG_PWD_AREA_KM2": float(latest["LOG_PWD_AREA_KM2"]),
        "LOG_PWD_G": float(latest["LOG_PWD_G"]),
        "LOG_PWD_D10": float(latest["LOG_PWD_D10"]),
        "CENTROID_LATITUDE": float(latest["CENTROID_LATITUDE"]),
        "CENTROID_LONGITUDE": float(latest["CENTROID_LONGITUDE"]),
    }


def _lag_and_rolling(df: pd.DataFrame, county: str, as_of_week: pd.Timestamp) -> dict:
    """
    Recreate EVENTS_LAG_1W / FATALITIES_LAG_1W / *_ROLLING_4W exactly as the
    notebook does: built from the county's real weekly history strictly
    before as_of_week, not including the hypothetical event being scored.
    """
    cdf = df[(df["COUNTY"] == county) & (df["WEEK"] < as_of_week)].sort_values("WEEK")
    if cdf.empty:
        return {"EVENTS_LAG_1W": 0.0, "FATALITIES_LAG_1W": 0.0,
                "EVENTS_ROLLING_4W": 0.0, "FATALITIES_ROLLING_4W": 0.0}

    weekly = cdf.groupby("WEEK")[["EVENTS", "FATALITIES"]].sum().sort_index()
    last_4 = weekly.tail(4)
    last_1 = weekly.tail(1)

    return {
        "EVENTS_LAG_1W": float(last_1["EVENTS"].sum()) if not last_1.empty else 0.0,
        "FATALITIES_LAG_1W": float(last_1["FATALITIES"].sum()) if not last_1.empty else 0.0,
        "EVENTS_ROLLING_4W": float(last_4["EVENTS"].sum()),
        "FATALITIES_ROLLING_4W": float(last_4["FATALITIES"].sum()),
    }


def build_event_row(
    df: pd.DataFrame,
    county: str,
    event_type: str,
    sub_event_type: str,
    event_date: pd.Timestamp,
    is_election_year_override: Optional[bool] = None,
) -> pd.DataFrame:
    """
    Build one model-ready row for a hypothetical or real event.

    is_election_year_override: if provided, forces IS_ELECTION_YEAR instead of
    deriving it from event_date.year - this is what powers the election-year
    toggle in the Event Scorer.
    """
    event_date = pd.Timestamp(event_date)
    week_start = event_date - pd.Timedelta(days=event_date.weekday())  # normalize to week

    context = _latest_county_context(df, county)
    lags = _lag_and_rolling(df, county, week_start)

    if is_election_year_override is None:
        is_election_year = int(event_date.year in ELECTION_YEARS)
    else:
        is_election_year = int(is_election_year_override)

    row = {
        "COUNTY": county,
        "EVENT_TYPE": event_type,
        "SUB_EVENT_TYPE": sub_event_type,
        "EVENTS": 1.0,
        "YEAR": event_date.year,
        "MONTH": event_date.month,
        "QUARTER": (event_date.month - 1) // 3 + 1,
        "WEEK_OF_YEAR": int(event_date.isocalendar()[1]),
        "IS_ELECTION_YEAR": is_election_year,
        **context,
        **lags,
    }
    return pd.DataFrame([row])


def score_event(stage1_model, stage2_model, stage2_label_encoder, event_row: pd.DataFrame) -> dict:
    """Run one event row through the two-stage hurdle model and return a result dict."""
    stage1_pred = int(stage1_model.predict(event_row)[0])
    stage1_proba = float(stage1_model.predict_proba(event_row)[0, 1])

    result = {
        "is_fatal_prediction": stage1_pred,
        "fatal_probability": stage1_proba,
        "severity_class": "Non-fatal",
        "severity_probabilities": None,
    }

    if stage1_pred == 1:
        stage2_input = event_row.copy()
        model_step = stage2_model.named_steps.get("model") if hasattr(stage2_model, "named_steps") else stage2_model
        if hasattr(model_step, "classes_") and set(model_step.classes_) <= {0, 1, 2}:
            # XGBoost path - encoded labels
            pred_encoded = int(stage2_model.predict(stage2_input)[0])
            severity_class = stage2_label_encoder.inverse_transform([pred_encoded])[0]
            proba = stage2_model.predict_proba(stage2_input)[0]
            proba_dict = dict(zip(stage2_label_encoder.inverse_transform(model_step.classes_), proba))
        else:
            severity_class = stage2_model.predict(stage2_input)[0]
            proba = stage2_model.predict_proba(stage2_input)[0]
            proba_dict = dict(zip(model_step.classes_, proba))

        result["severity_class"] = severity_class
        result["severity_probabilities"] = proba_dict

    return result
