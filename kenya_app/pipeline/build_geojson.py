"""
Regenerates kenya_app/data/kenya_counties.geojson from a GADM v4.1 Kenya
administrative-level-1 shapefile.

Where the shapefile came from: GADM (https://gadm.org/download_country.html)
- select Kenya, download the shapefile format, use the ADM_1 (county) level
  file - in the original download this was named gadm41_KEN_1.shp (plus its
  .dbf/.shx/.prj/.cpg sibling files, all required together).

This is NOT derived from ACLED or the notebook - it's a one-time
administrative boundary reference. Re-run this only if you get an updated
GADM extract or need to change the simplification tolerance.

Usage:
    python pipeline/build_geojson.py path/to/gadm41_KEN_1.shp
"""

import sys
from pathlib import Path

import geopandas as gpd

APP_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = APP_ROOT / "data"

# Two counties have spelling differences between GADM's naming and this
# project's convention (which follows the ACLED admin1 field) - same map
# used throughout the notebook and pipeline/build_features.py, kept
# identical here deliberately so a county name is spelled one way,
# everywhere, always.
COUNTY_NAME_MAP = {
    "Elgeyo-Marakwet": "Elgeyo Marakwet",
    "Murang'a": "Muranga",
}

EXPECTED_COUNTIES = {
    "Baringo", "Bomet", "Bungoma", "Busia", "Elgeyo Marakwet", "Embu", "Garissa",
    "Homa Bay", "Isiolo", "Kajiado", "Kakamega", "Kericho", "Kiambu", "Kilifi",
    "Kirinyaga", "Kisii", "Kisumu", "Kitui", "Kwale", "Laikipia", "Lamu", "Machakos",
    "Makueni", "Mandera", "Marsabit", "Meru", "Migori", "Mombasa", "Muranga",
    "Nairobi", "Nakuru", "Nandi", "Narok", "Nyamira", "Nyandarua", "Nyeri", "Samburu",
    "Siaya", "Taita Taveta", "Tana River", "Tharaka-Nithi", "Trans Nzoia", "Turkana",
    "Uasin Gishu", "Vihiga", "Wajir", "West Pokot",
}


def main():
    if len(sys.argv) < 2:
        print("Usage: python pipeline/build_geojson.py path/to/gadm41_KEN_1.shp")
        sys.exit(1)

    shapefile_path = sys.argv[1]
    gdf = gpd.read_file(shapefile_path)

    if "NAME_1" not in gdf.columns:
        print(f"ERROR: expected a 'NAME_1' column (GADM's county-name field), found: {list(gdf.columns)}")
        sys.exit(1)

    gdf["COUNTY"] = gdf["NAME_1"].replace(COUNTY_NAME_MAP)
    gdf = gdf[["COUNTY", "geometry"]].copy()

    # Simplify for web/browser rendering performance. 0.005 degrees (~500m at
    # this latitude) is a good balance - visibly identical at national-map
    # zoom, much smaller file. Increase this number for a smaller file at
    # the cost of coarser boundaries; set to 0 to keep full GADM precision.
    gdf["geometry"] = gdf["geometry"].simplify(0.005, preserve_topology=True)

    # Circuit breaker, same principle as refresh_acled.py: don't silently
    # write a broken map file if GADM's naming or county count doesn't
    # match what the rest of the app expects.
    actual_counties = set(gdf["COUNTY"])
    mismatch = EXPECTED_COUNTIES.symmetric_difference(actual_counties)
    if mismatch:
        print(f"ERROR: county name mismatch after mapping - {mismatch}")
        print("Check COUNTY_NAME_MAP above against the shapefile's actual NAME_1 values.")
        sys.exit(1)

    out_path = DATA_DIR / "kenya_counties.geojson"
    gdf.to_file(out_path, driver="GeoJSON")

    print(f"Wrote {out_path}")
    print(f"Counties: {len(gdf)} (all matched expected set)")
    print(f"File size: {out_path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
