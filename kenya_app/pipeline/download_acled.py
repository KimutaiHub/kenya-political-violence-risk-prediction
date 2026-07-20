"""
Downloads ACLED's Aggregated Africa workbook and filters it to Kenya.

This exists because the models were trained on ACLED's Aggregated Africa
workbook (downloaded by hand, opened in Excel, filtered to Kenya) - NOT on
raw event-level data from the ACLED Event API. Using a different upstream
source for inference than was used for training is a dataset-drift risk in
itself, independent of whether either source is individually correct. This
module reproduces the original manual workflow exactly, automated:

    Login to acleddata.com
            |
            v
    Find the latest workbook download link
            |
            v
    Download the .xlsx (keep a copy in data/raw/)
            |
            v
    Filter COUNTRY == "Kenya", parse WEEK, sort chronologically
            |
            v
    Save data/Africa_aggregated_data.csv (matches the original training input)

Authentication note: acleddata.com uses a Drupal login form with a
form_build_id field (not a CSRF token, not the OAuth2 flow used by the
Event API). This has never been tested against the live site from this
development environment (acleddata.com is not reachable from here) - the
implementation follows the documented login flow exactly, but the first
real run should be watched closely, ideally via --dry-run in
refresh_acled.py, before being trusted unattended.
"""

import logging
import re
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

LOGIN_PAGE_URL = "https://acleddata.com/user/login"
AGGREGATED_PAGE_URL = "https://acleddata.com/aggregated/aggregated-data-africa"

# The exact schema of the original training input (Africa_aggregated_data.csv,
# already filtered to Kenya). Used both to select/validate columns from the
# freshly downloaded workbook and to catch it early if ACLED ever changes
# the workbook's structure.
EXPECTED_WORKBOOK_COLUMNS = [
    "WEEK", "REGION", "COUNTRY", "ADMIN1", "EVENT_TYPE", "SUB_EVENT_TYPE",
    "EVENTS", "FATALITIES", "POPULATION_EXPOSURE", "DISORDER_TYPE", "ID",
    "CENTROID_LATITUDE", "CENTROID_LONGITUDE",
]


class AcledDownloadError(Exception):
    """Raised for any failure in the login/download/filter chain. Distinct
    from RefreshValidationError in refresh_acled.py - this is specifically
    about not being able to obtain trustworthy source data at all, before
    any of the modelling pipeline's own validation gets a chance to run."""


def login(session: requests.Session, email: str, password: str) -> requests.Session:
    """Authenticate against acleddata.com's Drupal login form and return the
    same session, now holding an authenticated cookie. Reuse this session
    for every subsequent request."""
    login_page = session.get(LOGIN_PAGE_URL, timeout=30)
    login_page.raise_for_status()

    soup = BeautifulSoup(login_page.text, "html.parser")
    form_build_id_input = soup.find("input", {"name": "form_build_id"})
    if form_build_id_input is None or not form_build_id_input.get("value"):
        raise AcledDownloadError(
            "Could not find form_build_id on the ACLED login page. The login page's HTML "
            "structure may have changed - inspect https://acleddata.com/user/login manually."
        )
    form_build_id = form_build_id_input["value"]

    # Prefer the form's own action attribute if present (handles ACLED ever
    # pointing the form somewhere other than the page it's rendered on);
    # fall back to the login page URL itself, which is the documented behavior.
    form_tag = soup.find("form", {"id": "user-login-form"}) or soup.find("form", {"id": re.compile("user-login")})
    post_url = urljoin(LOGIN_PAGE_URL, form_tag["action"]) if form_tag and form_tag.get("action") else LOGIN_PAGE_URL

    response = session.post(
        post_url,
        data={
            "name": email,
            "pass": password,
            "form_build_id": form_build_id,
            "form_id": "user_login_form",
            "op": "Log in",
        },
        timeout=30,
    )
    response.raise_for_status()

    # Drupal doesn't return a clean success/failure status code for a login
    # POST - a failed login re-renders the login form (HTTP 200) rather than
    # erroring. Detecting the form still being present is the most reliable
    # signal available without a documented API contract to check against.
    if "user-login-form" in response.text or "user_login_form" in response.text:
        raise AcledDownloadError(
            "ACLED login appears to have failed - the login form was still present in the "
            "response. Check ACLED_EMAIL/ACLED_PASSWORD are correct."
        )

    logger.info("Logged in to acleddata.com")
    return session


def find_latest_workbook_url(session: requests.Session) -> str:
    """Find the first .xlsx hyperlink on the Aggregated Africa data page.
    Deliberately does not hardcode a filename - ACLED publishes a new file
    (with the publish date in its name) on their own schedule."""
    page = session.get(AGGREGATED_PAGE_URL, timeout=30)
    page.raise_for_status()

    soup = BeautifulSoup(page.text, "html.parser")
    links = soup.find_all("a", href=True)
    xlsx_links = [a["href"] for a in links if a["href"].lower().endswith(".xlsx")]

    if not xlsx_links:
        raise AcledDownloadError(
            f"No .xlsx download link found on {AGGREGATED_PAGE_URL}. The page structure may "
            f"have changed, or this session isn't actually authenticated - inspect the page "
            f"manually before re-running."
        )

    workbook_url = urljoin(AGGREGATED_PAGE_URL, xlsx_links[0])
    logger.info(f"Found workbook link: {workbook_url}")
    return workbook_url


def download_workbook(session: requests.Session, url: str, raw_dir: Path) -> Path:
    """Download the workbook, streamed (these files can be large), and keep
    a permanent copy in data/raw/ for debugging and reproducibility."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1] or "Africa_aggregated_data.xlsx"
    output_path = raw_dir / filename

    with session.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Downloaded workbook to {output_path} ({size_mb:.1f} MB)")

    if size_mb < 0.01:
        raise AcledDownloadError(
            f"Downloaded file is suspiciously small ({size_mb:.3f} MB) - likely an error page "
            f"rather than the real workbook. Check {output_path} manually before proceeding."
        )

    return output_path


def filter_kenya(workbook_path: Path) -> pd.DataFrame:
    """Read the full-Africa workbook, filter to Kenya, parse WEEK, sort
    chronologically. Mirrors the manual Excel step from the original
    training workflow exactly."""
    df = pd.read_excel(workbook_path, engine="openpyxl")

    missing_cols = set(EXPECTED_WORKBOOK_COLUMNS) - set(df.columns)
    if missing_cols:
        raise AcledDownloadError(
            f"Downloaded workbook is missing expected columns: {missing_cols}. "
            f"ACLED may have changed the workbook's structure - check "
            f"https://acleddata.com/aggregated/aggregated-data-africa manually. "
            f"Columns actually present: {sorted(df.columns)}"
        )

    if "COUNTRY" not in df.columns:
        raise AcledDownloadError("No COUNTRY column in the downloaded workbook - cannot filter to Kenya.")

    kenya = df[df["COUNTRY"] == "Kenya"].copy()
    if kenya.empty:
        raise AcledDownloadError(
            "Filtering COUNTRY == 'Kenya' produced zero rows. Either the workbook's COUNTRY "
            "values changed format (check for whitespace/casing differences) or something is "
            "wrong with the download itself."
        )

    kenya["WEEK"] = _parse_week_column(kenya["WEEK"])
    kenya = kenya.sort_values("WEEK").reset_index(drop=True)

    logger.info(f"Filtered to {len(kenya):,} Kenya rows, {kenya['WEEK'].min().date()} to {kenya['WEEK'].max().date()}")
    return kenya


def _parse_week_column(week_series: pd.Series) -> pd.Series:
    """Robust WEEK parsing. ACLED's exports have used at least one
    non-ISO format historically ("24-January-1998", day-Month-Year) - try
    that explicitly before falling back to pandas' general inference, so a
    format pandas might otherwise mis-guess (e.g. month/day ambiguity) is
    handled correctly rather than silently."""
    if pd.api.types.is_datetime64_any_dtype(week_series):
        return week_series

    try:
        return pd.to_datetime(week_series, format="%d-%B-%Y")
    except (ValueError, TypeError):
        pass

    parsed = pd.to_datetime(week_series, errors="coerce", dayfirst=True)
    if parsed.isna().any():
        bad_count = parsed.isna().sum()
        raise AcledDownloadError(
            f"Could not parse {bad_count} WEEK values (out of {len(week_series)}) with any "
            f"known format. Sample unparsed values: {week_series[parsed.isna()].head(3).tolist()}"
        )
    return parsed


def save_kenya_csv(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {output_path} ({len(df):,} rows)")
    return output_path


def download_and_filter_kenya(email: str, password: str, data_dir: Path) -> Path:
    """Full pipeline: login, find and download the latest workbook, filter
    to Kenya, save the CSV. Returns the path to the saved Kenya CSV, matching
    what the original training workflow's manual Excel export produced."""
    session = requests.Session()
    login(session, email, password)

    workbook_url = find_latest_workbook_url(session)
    workbook_path = download_workbook(session, workbook_url, data_dir / "raw")

    kenya_df = filter_kenya(workbook_path)
    return save_kenya_csv(kenya_df, data_dir / "Africa_aggregated_data.csv")
