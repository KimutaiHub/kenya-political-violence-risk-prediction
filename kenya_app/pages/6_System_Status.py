import datetime
import os
from pathlib import Path

import streamlit as st

from utils.ui import inject_v2_styles, top_nav, intelligence_header, section_label, panel, status_pill, disclaimer
from utils.data_loader import load_conflict_data, get_latest_week, load_models, DATA_DIR, MODELS_DIR
from utils.news_feed import fetch_kenya_conflict_news

st.set_page_config(page_title="System Status - Kenya Conflict Monitor", page_icon="●", layout="wide")
inject_v2_styles()
top_nav(active_index=6)

intelligence_header("SYSTEM STATUS", "Dataset, model, and news connection health")

df = load_conflict_data()
latest_week = get_latest_week(df)
stage1_model, stage2_model, stage2_label_encoder, feature_lists, models_ok, model_errors = load_models()

section_label("Dataset Status")
with panel():
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"**Rows loaded**")
        st.markdown(f"<div class='v2-value'>{len(df):,}</div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"**Counties**")
        st.markdown(f"<div class='v2-value'>{df['COUNTY'].nunique()}</div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"**Latest known week**")
        st.markdown(f"<div class='v2-value' style='font-size:1.3rem;'>{latest_week.strftime('%d %b %Y')}</div>", unsafe_allow_html=True)
    with c4:
        days_stale = (datetime.datetime.now() - latest_week.to_pydatetime()).days
        freshness = "Low" if days_stale <= 14 else ("Moderate" if days_stale <= 45 else "High")
        st.markdown(f"**Data staleness**")
        st.markdown(f"<div style='margin-top:0.3rem;'>{status_pill(freshness)} <span style='color:var(--text-muted, #9CA3AF); font-size:0.8rem;'>({days_stale} days)</span></div>", unsafe_allow_html=True)

section_label("Model Status")
with panel():
    for name, label in [("stage1", "Stage 1 — Fatal Event Detection"), ("stage2", "Stage 2 — Severity Classification"),
                         ("label_encoder", "Stage 2 Label Encoder"), ("feature_lists", "Feature Schema")]:
        ok = name not in model_errors
        row = st.columns([2, 1, 4])
        with row[0]:
            st.markdown(f"**{label}**")
        with row[1]:
            st.markdown(status_pill("Online" if ok else "Offline"), unsafe_allow_html=True)
        with row[2]:
            if not ok:
                st.caption(model_errors[name])
            else:
                st.caption("Loaded successfully")

    if not models_ok:
        st.write("")
        st.warning(
            "One or more models failed to load. Event Scorer, SHAP explanations, and About page "
            "metadata will be degraded or unavailable until this is fixed. See README, 'Model version "
            "compatibility' for how to re-export models under a compatible numpy/scikit-learn version."
        )

section_label("News Connection")
with panel():
    try:
        test_articles = fetch_kenya_conflict_news(limit=1)
        news_ok = True
    except Exception:
        news_ok = False
        test_articles = []

    row = st.columns([2, 1, 4])
    with row[0]:
        st.markdown("**Google News RSS**")
    with row[1]:
        st.markdown(status_pill("Connected" if news_ok else "Offline"), unsafe_allow_html=True)
    with row[2]:
        if news_ok:
            st.caption(f"Test query returned {len(test_articles)} result(s)" if test_articles else "Connected, but no results for the test query - this can be normal.")
        else:
            st.caption("Could not reach Google News RSS. Check network access from this environment.")

section_label("File Inventory")
with panel():
    for label, path in [
        ("Conflict data", DATA_DIR / "kenya_conflict_full.parquet"),
        ("Vulnerability index", DATA_DIR / "county_vulnerability_index.csv"),
        ("County boundaries", DATA_DIR / "kenya_counties.geojson"),
        ("Model comparison table", DATA_DIR / "master_model_comparison.csv"),
        ("Stage 1 model", MODELS_DIR / "stage1_final_model.joblib"),
        ("Stage 2 model", MODELS_DIR / "stage2_final_model.joblib"),
    ]:
        exists = Path(path).exists()
        size = f"{os.path.getsize(path) / 1024:.0f} KB" if exists else "—"
        row = st.columns([2, 1, 1])
        with row[0]:
            st.markdown(f"**{label}**")
        with row[1]:
            st.markdown(status_pill("OK" if exists else "Missing"), unsafe_allow_html=True)
        with row[2]:
            st.caption(size)

st.write("")
disclaimer(
    "This page reflects the state of the currently running deployment only. Refresh the page after "
    "swapping any data or model file to see updated status."
)
