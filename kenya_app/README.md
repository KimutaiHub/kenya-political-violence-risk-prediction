# Kenya Conflict Intelligence System (v2)

Intelligence-dashboard companion app for the Kenya Political Violence Risk
Predictor capstone. Built with Streamlit, styled as a single cohesive
platform rather than a set of independent pages. Reads the models and data
produced by the capstone notebook — it does not retrain anything.

## Pages

- **Overview** (`app.py`) — executive dashboard: national KPIs, top vulnerable
  counties, national intelligence brief, recent/ongoing activity
- **County Intelligence** — the primary page. County overview, risk
  assessment, historical trends, SHAP explanation, intelligence brief,
  rule-based recommendations, embedded news, county statistics — everything
  needed to understand one county without leaving the page
- **Event Scorer** — score a hypothetical or real event: inputs, prediction
  summary, operational assessment, confidence, recommendations, county
  context, embedded news
- **Historical Analytics** — timeline slider, county × year heatmap, national
  trend, county comparison
- **Risk Map** — county choropleth, toggle between predicted vulnerability
  and live event status
- **About** — methodology, live model metadata (auto-generated from whatever
  model is actually loaded), performance table, limitations
- **System Status** — dataset/model/news connection health, file inventory

There is no standalone News or Recommendations page by design — both appear
as contextual panels inside County Intelligence and Event Scorer instead.

## Running locally

```bash
cd kenya_app
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501` and auto-launches a browser tab locally.
Multi-page nav is a persistent top bar (`utils/ui.py: top_nav()`), not the
default Streamlit sidebar list (hidden via CSS) — the sidebar is currently
unused space, free for you to repurpose later if useful.

Python 3.8+ is supported; avoid `X | Y` union type hints anywhere in this
codebase if editing on 3.8 — use `typing.Optional`/`Union` instead.

## Model version compatibility (numpy._core bug — read this if a model fails to load)

**Symptom:** `ModuleNotFoundError: No module named 'numpy._core'` when
loading a `.joblib` model file.

**Cause:** numpy 2.0 renamed internal modules. A model pickled under
numpy≥2.0 cannot be loaded by numpy<2.0, and Python 3.8 cannot install
numpy≥2.0 at all — so if the model files were exported from an environment
with numpy≥2.0, anyone on Python 3.8 (or numpy<2.0 for any reason) hits this.

**The app degrades gracefully now** (`utils/model_loader.py`) instead of
crashing: a failed model load shows a clear diagnostic on the **System
Status** page and disables just the affected features (Event Scorer, SHAP
explanations), rather than taking down the whole app. But the actual fix is
re-exporting the models under a numpy<2.0 environment:

```bash
python -m venv legacy_env
source legacy_env/bin/activate        # or legacy_env\Scripts\activate on Windows
pip install "numpy<2" "pandas<2.2" "scikit-learn<1.4" "xgboost<2.1" pyarrow joblib
python pipeline/refit_models.py
```

This re-fits the models fresh (not just re-pickles) using the exact
hyperparameters already selected during the notebook's hyperparameter
search, so results are functionally identical — only the serialization
environment changes. Re-run this any time a teammate's environment can't
load the shipped model files.

**Keeping deployment model-agnostic:** nothing in the app hardcodes which
algorithm is deployed. `utils/model_loader.py` and `utils/features.py` only
assume the loaded object exposes `.predict` / `.predict_proba` /
`.named_steps`, the way any scikit-learn `Pipeline` does. Swap a `.joblib`
file for a different (compatible) model and the app — including the
auto-generated model metadata on the About page — adapts without code
changes.

## Deploying to Streamlit Community Cloud

1. Push this folder to a GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo, point it at `app.py`.
3. Free hosting, auto-redeploys on every push to main.

## Data pipeline architecture

The models were trained on ACLED's **Aggregated Africa workbook**
(downloaded by hand, opened in Excel, filtered to `COUNTRY == "Kenya"`) -
not on raw event-level data from the ACLED Event API. An earlier version of
this refresh pipeline used the Event API, which is a *different* upstream
source with its own aggregation logic - using it for inference while the
models were trained on the aggregated workbook is a dataset-drift risk by
itself, independent of whether either source is individually correct.

`pipeline/refresh_acled.py` now reproduces the original manual workflow
exactly, automated:

```
pipeline/download_acled.py
    Login to acleddata.com (Drupal session, not the Event API's OAuth2)
            |
            v
    Find + download the latest Aggregated Africa workbook (.xlsx)
            |
            v
    Filter COUNTRY == "Kenya", parse WEEK, sort chronologically
            |
            v
    data/Africa_aggregated_data.csv   <- matches the original training input exactly
            |
            v
pipeline/build_features.py (unchanged)
    Population merge + feature engineering
            |
            v
pipeline/refresh_acled.py
    Validation -> data/kenya_conflict_full.parquet -> inference -> data/county_vulnerability_index.csv
```

Every run is a **full rebuild** from a fresh workbook download - not an
incremental append to the previous run's output. The workbook download
itself already contains the complete history, so there's no "new rows
since last time" concept the way the old Event-API pipeline had.

`pipeline/refit_models.py` is unaffected by any of this - it only reads the
final `kenya_conflict_full.parquet`, whose schema is identical regardless
of which pipeline built it (verified: rebuilding from the original training
CSV through the new pipeline reproduces the notebook's output exactly,
column-for-column and value-for-value).

## ACLED authentication

`pipeline/download_acled.py` authenticates against `acleddata.com`'s
Drupal login form directly (GET the login page, extract a `form_build_id`
hidden field, POST credentials against it) - not the Event API's OAuth2
flow. Same environment variables as before:

```bash
export ACLED_EMAIL="you@example.com"
export ACLED_PASSWORD="yourpassword"
```

**This has not been tested against the live acleddata.com** - it isn't
reachable from the development environment this was built in. The HTML
parsing (form field extraction, finding the `.xlsx` download link) is
tested against synthetic HTML matching the documented page structure, and
the whole pipeline is verified end-to-end using your original training CSV
in place of a fresh download - but the actual login/download HTTP round
trip has never run successfully for real. Treat the first real run as a
genuine first test; run it with `--dry-run` first (below) and read the
output carefully rather than assuming it'll just work.

## Testing locally, before ever touching GitHub Actions

**1. Dry run - fully local, hits the real ACLED website, writes nothing:**

```bash
cd kenya_app
export ACLED_EMAIL="you@example.com"
export ACLED_PASSWORD="yourpassword"
python pipeline/refresh_acled.py --dry-run
```

Runs the real login, real download, real filtering, real validation, and
prints exactly what would happen - but never writes to `data/` or
`models/`. **Always do this first**, especially the very first time this
runs against the real site.

**2. Real local run - writes files, still no git/deployment needed:**

```bash
python pipeline/refresh_acled.py
```

Check `git diff` on the changed files afterward before committing anything.

**3. GitHub Actions itself - requires the repo pushed to GitHub, not deployed:**

"Pushing to git" and "deploying the app" are separate things - you never
need to deploy the Streamlit app to test the refresh workflow. Push to
GitHub, add `ACLED_EMAIL`/`ACLED_PASSWORD` as repo secrets, then use the
**Actions** tab -> "Weekly ACLED refresh" -> **Run workflow** button
(`workflow_dispatch` in the yaml) to trigger it on demand instead of
waiting for Monday.

## What happens if ACLED changes their site or workbook structure

Validation runs at two points and refuses to write anything if either
fails (`RefreshValidationError` in `pipeline/refresh_acled.py`,
`AcledDownloadError` in `pipeline/download_acled.py`):

**At download time:**
- login failed (wrong credentials, or the login page's HTML structure changed)
- no `.xlsx` link found on the aggregated-data page
- downloaded file suspiciously small (likely an error page, not real data)
- workbook is missing expected columns (ACLED changed the workbook schema)
- filtering to Kenya produced zero rows (country value format changed, e.g. casing)

**After download, on the Kenya data itself:**
- missing expected columns, unparseable `WEEK` values, negative fatalities
- duplicate (county, week, event type, sub-event type) rows
- a gap of more than 21 days between consecutive weeks *in the most recent
  ~26 weeks* (this is deliberately NOT checked across the full 28-year
  history - the early ACLED years have genuine multi-week quiet stretches
  that are real data, not a stalled feed; only recent gaps indicate a
  live problem)

**After feature engineering:**
- wrong county count, negative values, null weeks, an implausible
  single-week fatality spike (duplication-bug detector)

**After inference:**
- a model fails to load or predict, or the output doesn't cover all 47 counties

If any check fails, the script exits non-zero and writes nothing - the
live app keeps serving last week's already-validated data. In GitHub
Actions, a non-zero exit on "Run refresh pipeline" means "Commit updated
data" never runs (`if: success()`, explicit in the workflow file). Check
the **Actions** tab for a failed run.

## Data products and what's committed to git

```
data/raw/*.xlsx                       <- NOT committed (see .gitignore) - large, re-downloaded every run
data/Africa_aggregated_data.csv       <- committed - the Kenya-filtered derived artifact
data/kenya_conflict_full.parquet      <- committed - the full feature-engineered panel
data/county_vulnerability_index.csv   <- committed - re-scored county rankings
```

## News

Google News RSS via `feedparser`, no API key required
(`utils/news_feed.py`). Replaces an earlier GDELT integration entirely.
Queries are county-scoped when called from County Intelligence or Event
Scorer, national otherwise. Failures return an empty list rather than
raising, so a quiet news day and a broken connection both degrade to the
same friendly "no headlines" state in the UI — check **System Status** to
tell them apart.

## Project structure

```
kenya_app/
├── app.py                        # Overview / Executive Dashboard
├── diagnose_news.py               # standalone news-feed diagnostic (bypasses Streamlit caching)
├── pages/
│   ├── 1_County_Intelligence.py  # primary page
│   ├── 2_Event_Scorer.py
│   ├── 3_Historical_Analytics.py
│   ├── 4_Risk_Map.py
│   ├── 5_About.py
│   └── 6_System_Status.py
├── utils/
│   ├── ui.py                     # design system - panels, KPI cards, status pills, top nav
│   ├── sidebar.py                # dynamic left-panel content, page-aware
│   ├── data_loader.py            # cached data + model loading
│   ├── model_loader.py           # robust, model-agnostic loading + metadata introspection
│   ├── features.py                # single-event feature construction + scoring
│   ├── recommendations.py        # rule-based action lists
│   ├── brief.py                  # auto-generated intelligence brief text
│   ├── shap_explain.py           # SHAP explanation of the live deployed model
│   └── news_feed.py              # Google News RSS
├── pipeline/
│   ├── build_features.py         # population extrapolation + feature engineering (mirrors the notebook)
│   ├── build_geojson.py          # regenerate kenya_counties.geojson from a GADM shapefile
│   ├── download_acled.py         # ACLED website login + Aggregated Africa workbook download/filter
│   ├── refresh_acled.py          # weekly refresh orchestrator, with validation circuit breaker
│   └── refit_models.py           # re-export models under numpy<2 (see above)
├── models/                       # trained model artifacts
├── data/                         # processed data the app reads
└── .github/workflows/            # scheduled refresh automation
```

## Data & model provenance

| File | Source | Reproducible via |
|---|---|---|
| `models/*.joblib` | Notebook Section 6.9–6.10 (final tuned models) | `pipeline/refit_models.py`, or the notebook export cells (see below) |
| `data/kenya_conflict_full.parquet` | Notebook `df` after Section 4 (feature engineering) | Notebook export cell (see below) |
| `data/county_vulnerability_index.csv` | Notebook `county_agg` from Section 6.15 | Notebook export cell (see below) |
| `data/vulnerability_stats.json` | Notebook Section 6.15 Spearman calculation | Notebook export cell (see below) - **not part of the original notebook**, added specifically for this app |
| `data/master_model_comparison.csv` | Notebook `master_model_comparison` from Section 6.17 | Notebook export cell (see below) |
| `data/raw_pwd_reference.csv` | Verbatim copy of the notebook's `PWD_2020-2000_sub_national_100m.csv` input | Copy the same source file |
| `data/kenya_counties.geojson` | GADM v4.1 Kenya administrative level 1 boundaries (not from the notebook or ACLED at all) | `pipeline/build_geojson.py path/to/gadm41_KEN_1.shp` - see that file's docstring for where to download the source shapefile |

### Notebook export cells

Add these to the master notebook, after Section 6.17 (Master Model Comparison),
alongside the model-export cells already given separately. Together they
regenerate every file in `kenya_app/data/` from a live notebook run.

```python
# --- Export data files for deployment (kenya_app/data/) ---
import os
import json

OUTPUT_DIR = "../kenya-app/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df.to_parquet(os.path.join(OUTPUT_DIR, "kenya_conflict_full.parquet"), index=False)
county_agg.to_csv(os.path.join(OUTPUT_DIR, "county_vulnerability_index.csv"), index=False)
master_model_comparison.to_csv(os.path.join(OUTPUT_DIR, "master_model_comparison.csv"), index=False)

json.dump(
    {"rho": float(rho), "pval": float(pval)},
    open(os.path.join(OUTPUT_DIR, "vulnerability_stats.json"), "w"),
)

print(f"Exported 4 data files to {OUTPUT_DIR}")
```

`raw_pwd_reference.csv` isn't generated by the notebook - it's a direct copy
of whichever file the notebook loads as its WorldPop population reference
(`PWD_2020-2000_sub_national_100m.csv` at time of writing). Copy it manually
if that source file ever changes.
