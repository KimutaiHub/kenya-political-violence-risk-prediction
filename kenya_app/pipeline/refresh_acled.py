"""
Weekly refresh job - full rebuild from ACLED's Aggregated Africa workbook.

This does NOT use the ACLED Event API. The models were trained on ACLED's
Aggregated Africa workbook (originally downloaded and Kenya-filtered by hand
in Excel), so inference has to use the identical upstream source - using a
different source (even one that's individually reasonable, like the Event
API) risks dataset drift between what the models learned and what they see
in production. See README, "Data pipeline architecture".

This always rebuilds kenya_conflict_full.parquet from scratch, from a fresh
workbook download - it does not append incrementally to the previous run's
output. It does not retrain models; that's pipeline/refit_models.py, a
fully separate, manual process.

Pipeline:
    download_acled.py: login -> find workbook -> download -> filter Kenya
            |
            v
    validate the raw Kenya data
            |
            v
    build_features.py: population merge -> feature engineering
            |
            v
    validate the rebuilt panel
            |
            v
    write kenya_conflict_full.parquet
            |
            v
    run inference -> county_vulnerability_index.csv

Requires a myACLED account (free): https://acleddata.com/register
Set as environment variables (or GitHub Actions secrets):
    ACLED_EMAIL
    ACLED_PASSWORD

Usage:
    python pipeline/refresh_acled.py [--dry-run]
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

APP_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = APP_ROOT / "data"
MODELS_DIR = APP_ROOT / "models"

sys.path.insert(0, str(APP_ROOT))
from pipeline.build_features import (  # noqa: E402
    build_population_reference, merge_population, clean_and_engineer_features,
)
from pipeline.download_acled import download_and_filter_kenya, AcledDownloadError  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EXPECTED_COUNTIES = 47
EXPECTED_KENYA_COLUMNS = {
    "WEEK", "ADMIN1", "EVENT_TYPE", "SUB_EVENT_TYPE", "EVENTS", "FATALITIES",
    "POPULATION_EXPOSURE", "CENTROID_LATITUDE", "CENTROID_LONGITUDE",
}
# A generous ceiling on the gap between consecutive distinct WEEK values
# across the whole dataset - this exists to catch a stalled/broken feed
# (ACLED not publishing, or the filter silently returning almost nothing),
# not to enforce a strict weekly cadence at the individual-county level,
# which is expected to be sparse (most counties are quiet most weeks).
MAX_PLAUSIBLE_WEEK_GAP_DAYS = 21
# Only the most recent slice of history is checked for gaps (see
# validate_kenya_csv) - the full 28-year record legitimately has multi-week
# quiet stretches in its earliest years that are real, not a bug.
RECENT_WINDOW_FOR_GAP_CHECK_WEEKS = 26


def _max_recent_week_gap(week_series: pd.Series) -> Optional[float]:
    """Largest gap (in days) between consecutive distinct weeks, restricted
    to roughly the most recent RECENT_WINDOW_FOR_GAP_CHECK_WEEKS weeks of
    the dataset. Returns None if there's not enough recent data to check."""
    distinct_weeks = week_series.drop_duplicates().sort_values()
    cutoff = distinct_weeks.max() - pd.Timedelta(weeks=RECENT_WINDOW_FOR_GAP_CHECK_WEEKS)
    recent_weeks = distinct_weeks[distinct_weeks >= cutoff]
    if len(recent_weeks) < 2:
        return None
    return recent_weeks.diff().dt.days.max()


class RefreshValidationError(Exception):
    """Raised when data fails a sanity check at any stage. The circuit
    breaker: if this is raised, main() must not write any output file -
    callers (and the GitHub Actions workflow) should treat a non-zero exit
    as 'do not trust or commit this run's output', not as a transient
    glitch to retry blindly."""


def validate_kenya_csv(df: pd.DataFrame) -> None:
    """Checks on the freshly downloaded-and-filtered Kenya data, before any
    feature engineering runs on it."""
    if df.empty:
        raise RefreshValidationError("Kenya dataset is empty after download and filtering.")

    missing_cols = EXPECTED_KENYA_COLUMNS - set(df.columns)
    if missing_cols:
        raise RefreshValidationError(f"Kenya dataset is missing expected columns: {missing_cols}")

    if not pd.api.types.is_datetime64_any_dtype(df["WEEK"]):
        raise RefreshValidationError("WEEK column did not parse to a datetime dtype.")

    if df["WEEK"].isna().any():
        raise RefreshValidationError(f"{df['WEEK'].isna().sum()} null WEEK values after parsing.")

    if (df["FATALITIES"] < 0).any():
        raise RefreshValidationError("Negative FATALITIES values in the downloaded data - not physically meaningful.")

    dup_key = ["ADMIN1", "WEEK", "EVENT_TYPE", "SUB_EVENT_TYPE"]
    duplicates = df.duplicated(subset=dup_key).sum()
    if duplicates > 0:
        raise RefreshValidationError(
            f"{duplicates} duplicate (county, week, event type, sub-event type) rows found. "
            f"Expected this combination to be unique per the workbook's own aggregation."
        )

    week_gap_days = _max_recent_week_gap(df["WEEK"])
    if week_gap_days is not None and week_gap_days > MAX_PLAUSIBLE_WEEK_GAP_DAYS:
        raise RefreshValidationError(
            f"Largest gap between consecutive weeks in the most recent "
            f"{RECENT_WINDOW_FOR_GAP_CHECK_WEEKS} weeks of data is {week_gap_days:.0f} days, "
            f"more than the {MAX_PLAUSIBLE_WEEK_GAP_DAYS}-day threshold. This usually means "
            f"ACLED's feed has stalled or the download/filter step silently returned partial data. "
            f"(Only recent data is checked for gaps - sparse coverage in the early historical "
            f"record, e.g. the late 1990s/2000s, is normal and not flagged.)"
        )

    logger.info(f"Kenya CSV validation passed: {len(df):,} rows, {df['ADMIN1'].nunique()} counties")


def validate_feature_engineered_output(full_df: pd.DataFrame) -> None:
    """Checks on the fully rebuilt, feature-engineered panel, immediately
    before it's allowed to overwrite the file the live app reads."""
    n_counties = full_df["COUNTY"].nunique()
    if n_counties != EXPECTED_COUNTIES:
        raise RefreshValidationError(
            f"Expected {EXPECTED_COUNTIES} counties after rebuild, got {n_counties}. "
            f"A county name likely changed upstream, or the population merge silently dropped rows."
        )

    for col in ["EVENTS", "FATALITIES"]:
        if (full_df[col] < 0).any():
            raise RefreshValidationError(f"Negative values found in {col} after feature engineering.")

    if full_df["WEEK"].isna().any():
        raise RefreshValidationError("Null WEEK values after feature engineering.")

    weekly_national_fatalities = full_df.groupby("WEEK")["FATALITIES"].sum()
    if (weekly_national_fatalities > 2000).any():
        raise RefreshValidationError(
            "At least one week shows over 2,000 national fatalities - implausible and far outside "
            "anything in the historical record. Likely a duplication bug, not real data."
        )

    logger.info(f"Feature-engineered panel validation passed: {full_df.shape}")


def build_dataset(kenya_csv_path: Path) -> pd.DataFrame:
    """Runs the full population-merge + feature-engineering pipeline
    (build_features.py) over a freshly downloaded Kenya dataset. Always a
    full rebuild - there is no 'existing data' concept here, since the
    workbook download itself already contains the complete history."""
    acled = pd.read_csv(kenya_csv_path)
    acled["WEEK"] = pd.to_datetime(acled["WEEK"])
    acled["YEAR"] = acled["WEEK"].dt.year

    validate_kenya_csv(acled)

    pwd_raw = pd.read_csv(DATA_DIR / "raw_pwd_reference.csv")
    target_max_year = max(int(acled["YEAR"].max()), pd.Timestamp.today().year + 4)

    try:
        pwd_filled = build_population_reference(pwd_raw, target_max_year)
        merged = merge_population(acled, pwd_filled)
    except AssertionError as e:
        raise RefreshValidationError(f"Population merge failed: {e}") from e

    try:
        full_df = clean_and_engineer_features(merged)
    except Exception as e:
        raise RefreshValidationError(f"Feature engineering failed: {type(e).__name__}: {e}") from e

    validate_feature_engineered_output(full_df)
    return full_df


def rescore(df: pd.DataFrame) -> pd.DataFrame:
    """Score every county-week with the already-trained models (inference
    only - see pipeline/refit_models.py for the separate, manual retraining
    process) and rebuild the county vulnerability index from the result."""
    try:
        stage1_model = joblib.load(MODELS_DIR / "stage1_final_model.joblib")
        stage2_model = joblib.load(MODELS_DIR / "stage2_final_model.joblib")
        stage2_label_encoder = joblib.load(MODELS_DIR / "stage2_label_encoder.joblib")
        feature_lists = joblib.load(MODELS_DIR / "model_feature_lists.joblib")
    except Exception as e:
        raise RefreshValidationError(f"Could not load models for inference: {type(e).__name__}: {e}") from e

    feature_cols = feature_lists["categorical_features"] + feature_lists["numeric_features"]
    X = df[feature_cols].copy()

    try:
        stage1_pred = stage1_model.predict(X)
    except Exception as e:
        raise RefreshValidationError(f"Stage 1 inference failed: {type(e).__name__}: {e}") from e

    severity_map = feature_lists["severity_numeric_map"]
    predicted_class = np.full(len(df), "Non-fatal", dtype=object)
    fatal_mask = stage1_pred == 1

    if fatal_mask.sum() > 0:
        try:
            model_step = stage2_model.named_steps.get("model") if hasattr(stage2_model, "named_steps") else stage2_model
            if hasattr(model_step, "classes_") and set(model_step.classes_) <= {0, 1, 2}:
                pred_encoded = stage2_model.predict(X.loc[fatal_mask]).astype(int)
                predicted_class[fatal_mask] = stage2_label_encoder.inverse_transform(pred_encoded)
            else:
                predicted_class[fatal_mask] = stage2_model.predict(X.loc[fatal_mask])
        except Exception as e:
            raise RefreshValidationError(f"Stage 2 inference failed: {type(e).__name__}: {e}") from e

    df = df.copy()
    df["PREDICTED_SEVERITY_SCORE"] = pd.Series(predicted_class).map(severity_map).values

    if df["PREDICTED_SEVERITY_SCORE"].isna().any():
        raise RefreshValidationError(
            "Inference produced unmapped severity classes - predicted_class contains a value "
            "not in severity_numeric_map. Model/feature schema may be out of sync."
        )

    county_agg = (
        df.groupby("COUNTY")
        .agg(
            PREDICTED_TOTAL_SEVERITY=("PREDICTED_SEVERITY_SCORE", "sum"),
            TEST_ROWS=("COUNTY", "size"),
        )
        .reset_index()
        .sort_values("PREDICTED_TOTAL_SEVERITY", ascending=False)
    )
    county_agg["RANK"] = range(1, len(county_agg) + 1)
    county_agg["PREDICTED_MEAN_SEVERITY"] = (
        county_agg["PREDICTED_TOTAL_SEVERITY"] / county_agg["TEST_ROWS"]
    ).round(3)
    # ACTUAL_* columns only exist for the notebook's historical test-period
    # snapshot (2022+, where outcomes are already known). Live refreshed data
    # is scoring the present, so there's no "actual" to compare against yet -
    # kept as NaN so the app's table columns stay consistent either way.
    county_agg["ACTUAL_TOTAL_SEVERITY"] = np.nan
    county_agg["ACTUAL_TOTAL_FATALITIES"] = np.nan

    if county_agg.empty or county_agg["COUNTY"].nunique() != EXPECTED_COUNTIES:
        raise RefreshValidationError(
            f"Inference output covers {county_agg['COUNTY'].nunique()} counties, expected {EXPECTED_COUNTIES}."
        )

    logger.info("Inference validation passed.")
    return county_agg


def main(dry_run: bool = False, email: Optional[str] = None, password: Optional[str] = None) -> None:
    email = email or os.environ.get("ACLED_EMAIL")
    password = password or os.environ.get("ACLED_PASSWORD")
    if not email or not password:
        logger.error("Set ACLED_EMAIL and ACLED_PASSWORD environment variables.")
        sys.exit(1)

    if dry_run:
        logger.info("DRY RUN - no files will be written, regardless of validation outcome.")

    try:
        kenya_csv_path = download_and_filter_kenya(email, password, DATA_DIR)
    except AcledDownloadError as e:
        raise RefreshValidationError(f"Download/filter stage failed: {e}") from e

    full_df = build_dataset(kenya_csv_path)
    county_agg = rescore(full_df)

    if dry_run:
        logger.info("DRY RUN complete - all validation passed, but no files were written.")
        logger.info("Re-run without --dry-run to actually apply this refresh.")
        return

    full_df.to_parquet(DATA_DIR / "kenya_conflict_full.parquet", index=False)
    county_agg.to_csv(DATA_DIR / "county_vulnerability_index.csv", index=False)
    logger.info("Refresh complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weekly ACLED refresh for the Kenya conflict monitor.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run the full pipeline including live download and validation, but never write to "
             "data/ or models/. Use this to test before trusting a real run.",
    )
    args = parser.parse_args()

    try:
        main(dry_run=args.dry_run)
    except RefreshValidationError as e:
        logger.error(f"REFRESH ABORTED - validation failed: {e}")
        logger.error(
            "No data files were modified. The live app continues serving last week's "
            "(validated) data. Fix the underlying issue and re-run manually."
        )
        sys.exit(1)
