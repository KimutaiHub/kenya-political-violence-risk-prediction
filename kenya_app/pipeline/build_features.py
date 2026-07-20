"""
This mirrors Sections 2 (Dataset Creation) and 4 (Cleaning & Feature
Engineering) of the capstone notebook exactly. It exists as a standalone
module so the weekly refresh job can rebuild the full feature set without
re-running the notebook, while staying logically identical to it.

If the notebook's feature engineering ever changes, update it here too -
these two are meant to never drift apart.
"""

import numpy as np
import pandas as pd

PWD_VALUE_COLS = ["Pop", "Density", "Area", "PWD_G", "PWD_D10"]
CENSUS_YEARS = [2000, 2005, 2010, 2015, 2020]
NATIONAL_GROWTH_RATE = 0.022  # KNBS 2019 Census inter-censal rate, 2009-2019
ELECTION_YEARS = [2002, 2007, 2013, 2017, 2022]

COUNTY_NAME_MAP = {
    "Elgeyo-Marakwet": "Elgeyo Marakwet",
    "Murang'a": "Muranga",
}


def build_population_reference(pwd_raw: pd.DataFrame, target_max_year: int) -> pd.DataFrame:
    """Interpolate, backfill, and extrapolate the WorldPop reference table
    out to target_max_year using the national-total / trending-county-share method."""
    pwd_raw = pwd_raw.copy()
    pwd_raw["Adm_N"] = pwd_raw["Adm_N"].replace(COUNTY_NAME_MAP)

    all_counties = sorted(pwd_raw["Adm_N"].unique())
    all_years = list(range(1997, target_max_year + 1))
    grid = pd.MultiIndex.from_product([all_counties, all_years], names=["Adm_N", "year"]).to_frame(index=False)
    pwd_grid = grid.merge(pwd_raw, on=["Adm_N", "year"], how="left")

    def interpolate_and_backfill_column(series):
        return series.interpolate(method="linear", limit_area="inside").bfill()

    pwd_grid = pwd_grid.sort_values(["Adm_N", "year"]).reset_index(drop=True)
    pwd_filled = pwd_grid.copy()
    for col in PWD_VALUE_COLS:
        pwd_filled[col] = pwd_filled.groupby("Adm_N")[col].transform(interpolate_and_backfill_column)

    census = pwd_filled[pwd_filled["year"].isin(CENSUS_YEARS)][["Adm_N", "year", "Pop"]].copy()
    national_totals = census.groupby("year")["Pop"].sum()

    shares = census.merge(national_totals.rename("national_pop"), on="year")
    shares["share"] = shares["Pop"] / shares["national_pop"]

    future_years = np.array([y for y in all_years if y > 2020])
    if len(future_years) > 0:
        share_trend = {}
        for county, g in shares.groupby("Adm_N"):
            coeffs = np.polyfit(g["year"], g["share"], 1)
            share_trend[county] = np.polyval(coeffs, future_years)

        share_future = pd.DataFrame(share_trend, index=future_years).T
        share_future = share_future.clip(lower=1e-6)
        share_future = share_future.div(share_future.sum(axis=0), axis=1)

        national_2020 = national_totals.loc[2020]
        national_future = pd.Series({
            y: national_2020 * (1 + NATIONAL_GROWTH_RATE) ** (y - 2020) for y in future_years
        })

        pop_future = share_future.mul(national_future, axis=1)
        pop_2020 = pwd_filled[pwd_filled["year"] == 2020].set_index("Adm_N")["Pop"]
        growth_factor = pop_future.div(pop_2020, axis=0)

        base_2020 = pwd_filled[pwd_filled["year"] == 2020].set_index("Adm_N")
        for county in growth_factor.index:
            for y in future_years:
                mask = (pwd_filled["Adm_N"] == county) & (pwd_filled["year"] == y)
                gf = growth_factor.loc[county, y]
                pwd_filled.loc[mask, "Pop"] = pop_future.loc[county, y]
                pwd_filled.loc[mask, "Density"] = base_2020.loc[county, "Density"] * gf
                pwd_filled.loc[mask, "PWD_G"] = base_2020.loc[county, "PWD_G"] * gf
                pwd_filled.loc[mask, "PWD_D10"] = base_2020.loc[county, "PWD_D10"] * gf
                pwd_filled.loc[mask, "Area"] = base_2020.loc[county, "Area"]

    assert pwd_filled[PWD_VALUE_COLS].isna().sum().sum() == 0, "population fill incomplete"
    return pwd_filled


def merge_population(acled: pd.DataFrame, pwd_filled: pd.DataFrame) -> pd.DataFrame:
    pwd_merge = pwd_filled[["Adm_N", "year"] + PWD_VALUE_COLS].copy()
    pwd_merge = pwd_merge.rename(columns={"Adm_N": "ADMIN1", "year": "YEAR"})
    pwd_merge = pwd_merge.rename(columns={
        "Pop": "PWD_POPULATION", "Density": "PWD_DENSITY", "Area": "PWD_AREA_KM2",
    })
    df = acled.merge(pwd_merge, on=["ADMIN1", "YEAR"], how="left")
    assert df["PWD_POPULATION"].isna().sum() == 0, "unmatched county/year after population merge"
    return df


def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Section 4 of the notebook: cleaning, renaming, and feature construction."""
    df = df.copy()
    drop_cols = [c for c in ["REGION", "COUNTRY", "ID", "DISORDER_TYPE"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    df["POPULATION_EXPOSURE_MISSING"] = df["POPULATION_EXPOSURE"].isna().astype(int)
    df["POPULATION_EXPOSURE"] = df["POPULATION_EXPOSURE"].fillna(0)

    df = df.rename(columns={
        "ADMIN1": "COUNTY",
        "PWD_POPULATION": "COUNTY_POPULATION",
        "PWD_DENSITY": "COUNTY_RAW_DENSITY",
        "PWD_AREA_KM2": "COUNTY_AREA_KM2",
        "PWD_G": "COUNTY_WEIGHTED_DENSITY",
        "PWD_D10": "COUNTY_URBAN_CORE_DENSITY",
    })

    df["MONTH"] = df["WEEK"].dt.month
    df["QUARTER"] = df["WEEK"].dt.quarter
    df["WEEK_OF_YEAR"] = df["WEEK"].dt.isocalendar().week.astype(int)
    df["IS_ELECTION_YEAR"] = df["YEAR"].isin(ELECTION_YEARS).astype(int)

    df["LOG_POPULATION_EXPOSURE"] = np.log1p(df["POPULATION_EXPOSURE"])
    df["LOG_PWD_POPULATION"] = np.log1p(df["COUNTY_POPULATION"])
    df["LOG_PWD_DENSITY"] = np.log1p(df["COUNTY_RAW_DENSITY"])
    df["LOG_PWD_AREA_KM2"] = np.log1p(df["COUNTY_AREA_KM2"])
    df["LOG_PWD_G"] = np.log1p(df["COUNTY_WEIGHTED_DENSITY"])
    df["LOG_PWD_D10"] = np.log1p(df["COUNTY_URBAN_CORE_DENSITY"])

    df = df.sort_values(["COUNTY", "WEEK"]).reset_index(drop=True)

    weekly = df.groupby(["COUNTY", "WEEK"])[["EVENTS", "FATALITIES"]].sum().reset_index()
    all_weeks = pd.date_range(df["WEEK"].min(), df["WEEK"].max(), freq="W-SAT")
    panel_index = pd.MultiIndex.from_product(
        [sorted(df["COUNTY"].unique()), all_weeks], names=["COUNTY", "WEEK"]
    )
    panel = (
        weekly.set_index(["COUNTY", "WEEK"])
        .reindex(panel_index, fill_value=0)
        .reset_index()
        .sort_values(["COUNTY", "WEEK"])
    )

    panel["EVENTS_LAG_1W"] = panel.groupby("COUNTY")["EVENTS"].shift(1).fillna(0)
    panel["FATALITIES_LAG_1W"] = panel.groupby("COUNTY")["FATALITIES"].shift(1).fillna(0)
    panel["EVENTS_ROLLING_4W"] = (
        panel.groupby("COUNTY")["EVENTS"].transform(lambda x: x.shift(1).rolling(4, min_periods=1).sum()).fillna(0)
    )
    panel["FATALITIES_ROLLING_4W"] = (
        panel.groupby("COUNTY")["FATALITIES"].transform(lambda x: x.shift(1).rolling(4, min_periods=1).sum()).fillna(0)
    )

    lag_cols = ["EVENTS_LAG_1W", "FATALITIES_LAG_1W", "EVENTS_ROLLING_4W", "FATALITIES_ROLLING_4W"]
    df = df.merge(panel[["COUNTY", "WEEK"] + lag_cols], on=["COUNTY", "WEEK"], how="left", validate="many_to_one")

    df["IS_FATAL"] = (df["FATALITIES"] > 0).astype(int)

    def severity_3class(x):
        if x <= 2:
            return "Low"
        elif x <= 5:
            return "Medium"
        return "High"

    df["FATALITY_SEVERITY"] = df["FATALITIES"].apply(severity_3class)

    return df
