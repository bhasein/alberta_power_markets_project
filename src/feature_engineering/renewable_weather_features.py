# src/feature_engineering/renewable_weather_features.py

"""
Build hourly capacity-weighted weather features for Alberta wind and solar.

Pipeline
--------
1. Load the renewable-project registry.
2. Open one standardized ERA5 file to obtain the weather grid.
3. Map every project to its nearest ERA5 latitude/longitude.
4. Open each monthly spatial ERA5 NetCDF file.
5. Extract weather only at grid cells used by renewable projects.
6. Determine which projects existed during each hour.
7. Weight project weather by active installed capacity.
8. Save one hourly wind/solar weather-feature table.

Historical weighting
--------------------
For project i at timestamp t:

    active_capacity(i, t) =
        capacity_mw(i), if:
            commissioning_date(i) <= t
            and retirement_date(i) > t

        0 otherwise

The normalized project weight is:

    weight(i, t) =
        active_capacity(i, t)
        / total_active_capacity(t)

Projects therefore contribute nothing before commissioning or after
retirement.

Project phases should be represented as separate project rows. For example,
a wind facility expanded in 2019 and 2022 should have separate rows for each
capacity phase, each with its own commissioning date.

Expected project registry columns
---------------------------------
project_id
project_name
fuel_type
latitude
longitude
capacity_mw
commissioning_date
retirement_date       optional

Supported fuel_type values:
wind
solar

Outputs
-------
data/features/weather/renewable_weather_features_hourly.parquet
data/features/weather/renewable_project_weather_mapping.csv
data/audits/weather_features_monthly_summary.csv
data/audits/weather_features_audit_checks.csv

Run
---
python src/feature_engineering/renewable_weather_features.py

or:

.venv/bin/python src/feature_engineering/renewable_weather_features.py
"""

# Imports
from __future__ import annotations
import argparse
import calendar
import logging
import time
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
import xarray as xr


# ============================================================================
# Logging
# ============================================================================

LOGGER = logging.getLogger(__name__)


def configure_logging(
    verbose: bool = False,
) -> None:
    """Configure console logging for the pipeline."""

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


# ============================================================================
# Paths
# ============================================================================

# This file is expected to live at:
#
#     PROJECT_ROOT/src/feature_engineering/renewable_weather_features.py
#
# parents[0] -> feature_engineering
# parents[1] -> src
# parents[2] -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREPROCESSING_DIR = PROJECT_ROOT / "data" / "preprocessing"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
OUTPUT_DIR = FEATURES_DIR / "weather"
AUDIT_DIR = PROJECT_ROOT / "data" / "audits"

ERA5_MONTHLY_DIR = (
    PREPROCESSING_DIR
    / "weather"
    / "era5"
    / "monthly_standardized"
)

WIND_PROJECTS_FILE = (
    PREPROCESSING_DIR
    / "wind_projects_preprocessed.csv"
)

SOLAR_PROJECTS_FILE = (
    PREPROCESSING_DIR
    / "solar_projects_preprocessed.csv"
)

OUTPUT_PARQUET = (
    OUTPUT_DIR
    / "renewable_weather_features_hourly.parquet"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "renewable_weather_features_hourly.csv"
)

# Backward-compatible alias retained for code importing OUTPUT_FILE.
OUTPUT_FILE = OUTPUT_PARQUET

PROJECT_MAPPING_FILE = (
    OUTPUT_DIR
    / "renewable_project_weather_mapping.csv"
)

MONTHLY_SUMMARY_FILE = (
    AUDIT_DIR
    / "weather_features_monthly_summary.csv"
)

AUDIT_FILE = (
    AUDIT_DIR
    / "weather_features_audit_checks.csv"
)


# Configuration
VALID_FUEL_TYPES = {
    "wind",
    "solar",
}

# Variables used directly or in engineered features for wind.
WIND_SOURCE_VARIABLES = [
    "u_wind_100m",
    "v_wind_100m",
    "u_wind_10m",
    "v_wind_10m",
    "instantaneous_10m_wind_gust",
    "max_10m_wind_gust",
    "temperature_2m",
    "surface_pressure",
    "mean_sea_level_pressure",
    "temperature_850hpa",
    "u_wind_850hpa",
    "v_wind_850hpa",
    "boundary_layer_height",
    "total_cloud_cover",
    "total_precipitation",
    "snowfall",
    "snow_depth",
]

# Variables useful for solar-generation conditions.
SOLAR_SOURCE_VARIABLES = [
    "surface_solar_radiation_downwards",
    "surface_solar_radiation_downwards_clear_sky",
    "total_sky_direct_solar_radiation",
    "temperature_2m",
    "skin_temperature",
    "total_cloud_cover",
    "high_cloud_cover",
    "medium_cloud_cover",
    "low_cloud_cover",
    "total_precipitation",
    "snowfall",
    "snow_depth",
]

# Create a sorted set of wind and solar source variables
ALL_SOURCE_VARIABLES = sorted(set(WIND_SOURCE_VARIABLES + SOLAR_SOURCE_VARIABLES))

WIND_DIRECT_WEIGHT_VARIABLES = [
    "u_wind_100m",
    "v_wind_100m",
    "u_wind_10m",
    "v_wind_10m",
    "instantaneous_10m_wind_gust",
    "max_10m_wind_gust",
    "temperature_2m",
    "surface_pressure",
    "mean_sea_level_pressure",
    "temperature_850hpa",
    "u_wind_850hpa",
    "v_wind_850hpa",
    "boundary_layer_height",
    "total_cloud_cover",
    "total_precipitation",
    "snowfall",
    "snow_depth",
]

SOLAR_DIRECT_WEIGHT_VARIABLES = SOLAR_SOURCE_VARIABLES.copy()

AIR_GAS_CONSTANT = 287.05



# General helpers
def add_check(
    rows: list[dict],
    period: str,
    check: str,
    passed: bool,
    observed=None,
    expected=None,
    severity: str = "error",
    notes: str = "",
) -> None:
    """Append one audit check."""

    # Append dictionary to rows
    rows.append(
        {
            "period": period,
            "check": check,
            "pass": bool(passed),
            "severity": severity,
            "observed": observed,
            "expected": expected,
            "notes": notes,
        }
    )


def require_columns(
    dataframe: pd.DataFrame,
    required_columns: Iterable[str],
    dataset_name: str,
) -> None:
    """Validate that a dataframe contains required columns."""

    # store missing columns
    missing = sorted(
        set(required_columns) - set(dataframe.columns)
    )

    # if columns are missing, raise an error and print them
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing}"
        )


def monthly_file_period(path: Path) -> str:
    """
    Extract YYYY-MM from a standardized monthly filename.

    Expected filename:
        era5_alberta_standardized_2020_01.nc
    """

    # stem returns file name without the extension, and splits the file name between the "_"
    parts = path.stem.split("_")

    # raise error if the length of the resulting list is not long enough
    if len(parts) < 2:
        raise ValueError(
            f"Cannot identify year and month from {path.name}"
        )

    # store year and month value (last and second last values)
    year = parts[-2]
    month = parts[-1]

    # return string of year-month
    return f"{year}-{month}"


def expected_month_hours(period: str) -> int:
    """Return expected number of hourly timestamps in a month."""

    # split and store year/month in a list, converted to ints
    year, month = map(int, period.split("-"))

    # monthrange returns a tuple
    # pick second value of the typle (days) and multiply by 24
    return (
        calendar.monthrange(year, month)[1]
        * 24
    )


# Project registry
def normalize_project_columns(
    projects: pd.DataFrame,
    fuel_type: str,
) -> pd.DataFrame:
    """
    Normalize the existing wind/solar project files to the schema expected
    by the weather-feature pipeline.

    Current source columns
    ----------------------
    Wind:
        project_name
        project_identifier
        capacity_mw
        turbines
        rotor_diameter_m
        hub_height_m
        commissioning_year
        latitude
        longitude

    Solar:
        project_name
        capacity_mw
        commissioning_year
        latitude
        longitude
    """

    # create new copy of projects file
    projects = projects.copy()

    # strip whitespace, convert to lowercase, replace separation with "_", stip "_" on the edge
    projects.columns = (
        projects.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    # Wind file already has an identifier.
    if "project_identifier" in projects.columns:
        projects = projects.rename(
            columns={
                "project_identifier": "project_id",
            }
        )

    # Solar has no identifier, so generate one from project_name.
    if "project_id" not in projects.columns:
        normalized_name = (
            projects["project_name"]
            .astype(str)
            .str.strip()
            .str.upper()
            .str.replace(
                r"[^A-Z0-9]+",
                "_",
                regex=True,
            )
            .str.strip("_")
        )

        projects["project_id"] = (
            fuel_type.upper()
            + "_"
            + normalized_name
        )

    projects["fuel_type"] = fuel_type

    projects["commissioning_year"] = pd.to_numeric(
        projects["commissioning_year"],
        errors="coerce",
    )

    projects["commissioning_date"] = pd.to_datetime(
        projects["commissioning_year"]
        .astype("Int64")
        .astype("string")
        + "-01-01",
        errors="coerce",
        utc=True,
    )

    if "retirement_date" not in projects.columns:
        projects["retirement_date"] = pd.NaT

    return projects


def load_project_file(
    path: Path,
    fuel_type: str,
) -> pd.DataFrame:
    """
    Load and validate one renewable-project file.

    The fuel type is assigned from the file being loaded rather than
    requiring a fuel_type column to already exist.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"{fuel_type.capitalize()} project file does not exist: {path}"
        )

    projects = pd.read_csv(path)

    projects = normalize_project_columns(
        projects=projects,
        fuel_type=fuel_type,
    )

    required_columns = {
        "project_id",
        "project_name",
        "latitude",
        "longitude",
        "capacity_mw",
        "commissioning_date",
    }

    require_columns(
        projects,
        required_columns,
        f"{fuel_type.capitalize()} project file",
    )

    projects = projects.copy()

    projects["project_id"] = (
        projects["project_id"]
        .astype(str)
        .str.strip()
    )

    projects["project_name"] = (
        projects["project_name"]
        .astype(str)
        .str.strip()
    )

    for column in [
        "latitude",
        "longitude",
        "capacity_mw",
    ]:
        projects[column] = pd.to_numeric(
            projects[column],
            errors="raise",
        )

    if projects["project_id"].duplicated().any():
        duplicate_ids = projects.loc[
            projects["project_id"].duplicated(keep=False),
            "project_id",
        ].tolist()

        raise ValueError(
            f"{fuel_type.capitalize()} project_id values must be unique. "
            f"Duplicates: {duplicate_ids[:20]}"
        )

    if projects["latitude"].isna().any():
        raise ValueError(
            f"{fuel_type.capitalize()} project latitude contains missing values."
        )

    if projects["longitude"].isna().any():
        raise ValueError(
            f"{fuel_type.capitalize()} project longitude contains missing values."
        )

    if projects["capacity_mw"].isna().any():
        raise ValueError(
            f"{fuel_type.capitalize()} project capacity_mw contains missing values."
        )

    if (projects["capacity_mw"] <= 0).any():
        invalid = projects.loc[
            projects["capacity_mw"] <= 0,
            [
                "project_id",
                "capacity_mw",
            ],
        ]

        raise ValueError(
            f"Every {fuel_type} project capacity must be greater than zero:\n"
            f"{invalid.to_string(index=False)}"
        )

    # Commissioning year is necessary for historical weighting.
    if projects["commissioning_date"].isna().any():
        invalid = projects.loc[
            projects["commissioning_date"].isna(),
            [
                "project_id",
                "project_name",
                "capacity_mw",
                "commissioning_year",
            ],
        ]

        raise ValueError(
            f"{fuel_type.capitalize()} projects are missing commissioning years. "
            "Fill these before building historical weather weights:\n"
            f"{invalid.to_string(index=False)}"
        )

    projects["retirement_date"] = pd.to_datetime(
        projects["retirement_date"],
        errors="coerce",
        utc=True,
    )

    invalid_dates = (
        projects["retirement_date"].notna()
        & (
            projects["retirement_date"]
            <= projects["commissioning_date"]
        )
    )

    if invalid_dates.any():
        invalid = projects.loc[
            invalid_dates,
            [
                "project_id",
                "commissioning_date",
                "retirement_date",
            ],
        ]

        raise ValueError(
            f"{fuel_type.capitalize()} retirement_date must be later "
            "than commissioning_date:\n"
            f"{invalid.to_string(index=False)}"
        )

    return projects

def load_projects(
    wind_path: Path,
    solar_path: Path,
) -> pd.DataFrame:
    """
    Load the wind and solar project files and combine them into one table.
    """

    wind_projects = load_project_file(
        path=wind_path,
        fuel_type="wind",
    )

    solar_projects = load_project_file(
        path=solar_path,
        fuel_type="solar",
    )

    projects = pd.concat(
        [
            wind_projects,
            solar_projects,
        ],
        ignore_index=True,
    )

    duplicate_mask = projects[
        "project_id"
    ].duplicated(keep=False)

    if duplicate_mask.any():
        duplicates = projects.loc[
            duplicate_mask,
            [
                "project_id",
                "project_name",
                "fuel_type",
            ],
        ]

        raise ValueError(
            "project_id values must be unique across the wind and "
            "solar project files:\n"
            f"{duplicates.to_string(index=False)}"
        )

    projects = (
        projects
        .sort_values(
            [
                "fuel_type",
                "commissioning_date",
                "project_id",
            ]
        )
        .reset_index(drop=True)
    )

    return projects

# ============================================================================
# ERA5 spatial grid mapping
# ============================================================================

def get_monthly_files(
    monthly_dir: Path,
) -> list[Path]:
    """Return standardized monthly ERA5 files in chronological order."""

    files = sorted(
        monthly_dir.glob(
            "era5_alberta_standardized_*.nc"
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No standardized monthly ERA5 files found in "
            f"{monthly_dir}"
        )

    return files


def load_grid(reference_file: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load latitude and longitude arrays from one ERA5 file."""

    with xr.open_dataset(reference_file) as ds:
        required_coordinates = {
            "timestamp",
            "lat",
            "lon",
        }

        missing = (
            required_coordinates
            - set(ds.coords)
            - set(ds.dims)
        )

        if missing:
            raise ValueError(
                f"{reference_file.name} is missing coordinates: "
                f"{sorted(missing)}"
            )

        latitudes = np.asarray(
            ds["lat"].values,
            dtype=float,
        )

        longitudes = np.asarray(
            ds["lon"].values,
            dtype=float,
        )

    return latitudes, longitudes


def nearest_coordinate(
    value: float,
    available_coordinates: np.ndarray,
) -> tuple[float, int]:
    """Return the nearest coordinate value and its array position."""

    position = int(
        np.abs(
            available_coordinates - value
        ).argmin()
    )

    return (
        float(available_coordinates[position]),
        position,
    )


def approximate_distance_km(
    project_latitude: float,
    project_longitude: float,
    grid_latitude: float,
    grid_longitude: float,
) -> float:
    """
    Calculate approximate great-circle distance using the haversine formula.
    """

    earth_radius_km = 6371.0088

    lat1 = np.radians(project_latitude)
    lon1 = np.radians(project_longitude)
    lat2 = np.radians(grid_latitude)
    lon2 = np.radians(grid_longitude)

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    haversine_value = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(delta_lon / 2.0) ** 2
    )

    central_angle = (
        2.0
        * np.arcsin(
            np.sqrt(haversine_value)
        )
    )

    return float(
        earth_radius_km * central_angle
    )


def map_projects_to_grid(
    projects: pd.DataFrame,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> pd.DataFrame:
    """Map every project to its nearest ERA5 grid coordinate."""

    mapped = projects.copy()

    weather_latitudes = []
    weather_longitudes = []
    latitude_positions = []
    longitude_positions = []
    distances_km = []

    for row in mapped.itertuples(index=False):
        weather_lat, lat_position = nearest_coordinate(
            float(row.latitude),
            latitudes,
        )

        weather_lon, lon_position = nearest_coordinate(
            float(row.longitude),
            longitudes,
        )

        distance = approximate_distance_km(
            project_latitude=float(row.latitude),
            project_longitude=float(row.longitude),
            grid_latitude=weather_lat,
            grid_longitude=weather_lon,
        )

        weather_latitudes.append(weather_lat)
        weather_longitudes.append(weather_lon)
        latitude_positions.append(lat_position)
        longitude_positions.append(lon_position)
        distances_km.append(distance)

    mapped["weather_lat"] = weather_latitudes
    mapped["weather_lon"] = weather_longitudes
    mapped["weather_lat_position"] = latitude_positions
    mapped["weather_lon_position"] = longitude_positions
    mapped["weather_distance_km"] = distances_km

    unique_grid = (
        mapped[
            [
                "weather_lat",
                "weather_lon",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "weather_lat",
                "weather_lon",
            ]
        )
        .reset_index(drop=True)
    )

    unique_grid["weather_site_id"] = np.arange(
        len(unique_grid),
        dtype=int,
    )

    mapped = mapped.merge(
        unique_grid,
        on=[
            "weather_lat",
            "weather_lon",
        ],
        how="left",
        validate="many_to_one",
    )

    mapped["weather_site_id"] = (
        mapped["weather_site_id"]
        .astype(int)
    )

    return mapped


# ============================================================================
# Monthly weather extraction
# ============================================================================

def extract_project_sites(
    ds: xr.Dataset,
    project_mapping: pd.DataFrame,
) -> xr.Dataset:
    """
    Extract weather at each unique grid cell used by the project fleet.

    Xarray vectorized indexing is used so latitude and longitude are paired
    by weather_site_id rather than creating their Cartesian product.
    """

    unique_sites = (
        project_mapping[
            [
                "weather_site_id",
                "weather_lat",
                "weather_lon",
            ]
        ]
        .drop_duplicates()
        .sort_values("weather_site_id")
        .reset_index(drop=True)
    )

    site_ids = unique_sites["weather_site_id"].to_numpy(
        dtype=int
    )

    lat_indexer = xr.DataArray(
        unique_sites["weather_lat"].to_numpy(dtype=float),
        dims="weather_site_id",
        coords={
            "weather_site_id": site_ids,
        },
    )

    lon_indexer = xr.DataArray(
        unique_sites["weather_lon"].to_numpy(dtype=float),
        dims="weather_site_id",
        coords={
            "weather_site_id": site_ids,
        },
    )

    available_variables = [
        variable
        for variable in ALL_SOURCE_VARIABLES
        if variable in ds.data_vars
    ]

    if not available_variables:
        raise ValueError(
            "Monthly ERA5 file contains none of the configured "
            "weather variables."
        )

    extracted = ds[available_variables].sel(
        lat=lat_indexer,
        lon=lon_indexer,
        method="nearest",
    )

    return extracted


def weather_array(
    site_dataset: xr.Dataset,
    variable: str,
) -> np.ndarray | None:
    """
    Return a weather variable as timestamp × site.

    Returns None when the variable is unavailable.
    """

    if variable not in site_dataset.data_vars:
        return None

    array = (
        site_dataset[variable]
        .transpose(
            "timestamp",
            "weather_site_id",
        )
        .values
    )

    return np.asarray(
        array,
        dtype=float,
    )


# ============================================================================
# Weighting
# ============================================================================

def active_capacity_matrix(
    timestamps: pd.DatetimeIndex,
    projects: pd.DataFrame,
) -> np.ndarray:
    """
    Build timestamp × project active-capacity matrix.

    Commissioning is inclusive.
    Retirement is exclusive.
    """

    timestamp_values = timestamps.to_numpy(
        dtype="datetime64[ns]"
    )[:, None]

    commissioning = (
        projects["commissioning_date"]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .to_numpy(dtype="datetime64[ns]")
    )[None, :]

    retirement_series = projects["retirement_date"]

    retirement = (
        retirement_series
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .fillna(
            pd.Timestamp.max.tz_localize(None)
        )
        .to_numpy(dtype="datetime64[ns]")
    )[None, :]

    capacity = projects["capacity_mw"].to_numpy(
        dtype=float
    )[None, :]

    active = (
        (timestamp_values >= commissioning)
        & (timestamp_values < retirement)
    )

    return active.astype(float) * capacity


def project_values_from_sites(
    site_values: np.ndarray,
    projects: pd.DataFrame,
) -> np.ndarray:
    """
    Expand timestamp × site weather into timestamp × project weather.

    Projects sharing one ERA5 grid cell share the same extracted weather.
    """

    project_site_positions = (
        projects["weather_site_id"]
        .to_numpy(dtype=int)
    )

    return site_values[:, project_site_positions]


def weighted_project_average(
    project_values: np.ndarray,
    active_capacity: np.ndarray,
) -> np.ndarray:
    """
    Compute hourly capacity-weighted average while handling missing weather.

    Missing project weather is removed from both numerator and denominator
    for the corresponding hour.
    """

    valid = np.isfinite(project_values)

    valid_capacity = np.where(
        valid,
        active_capacity,
        0.0,
    )

    numerator = np.nansum(
        project_values * valid_capacity,
        axis=1,
    )

    denominator = np.sum(
        valid_capacity,
        axis=1,
    )

    result = np.full(
        project_values.shape[0],
        np.nan,
        dtype=float,
    )

    np.divide(
        numerator,
        denominator,
        out=result,
        where=denominator > 0,
    )

    return result


def weighted_project_standard_deviation(
    project_values: np.ndarray,
    active_capacity: np.ndarray,
) -> np.ndarray:
    """
    Compute hourly capacity-weighted cross-project standard deviation.

    This indicates how geographically uniform or dispersed wind conditions
    are across the active fleet.
    """

    weighted_mean = weighted_project_average(
        project_values,
        active_capacity,
    )

    valid = np.isfinite(project_values)

    valid_capacity = np.where(
        valid,
        active_capacity,
        0.0,
    )

    squared_deviation = (
        project_values
        - weighted_mean[:, None]
    ) ** 2

    numerator = np.nansum(
        squared_deviation * valid_capacity,
        axis=1,
    )

    denominator = np.sum(
        valid_capacity,
        axis=1,
    )

    result = np.full(
        project_values.shape[0],
        np.nan,
        dtype=float,
    )

    np.divide(
        numerator,
        denominator,
        out=result,
        where=denominator > 0,
    )

    return np.sqrt(result)


def wind_speed(
    u_component: np.ndarray,
    v_component: np.ndarray,
) -> np.ndarray:
    """Calculate wind speed from u and v components."""

    return np.sqrt(
        u_component ** 2
        + v_component ** 2
    )


def meteorological_wind_direction(
    u_component: np.ndarray,
    v_component: np.ndarray,
) -> np.ndarray:
    """
    Calculate meteorological wind direction in degrees.

    Direction describes where the wind comes from:
        0 degrees   = north
        90 degrees  = east
        180 degrees = south
        270 degrees = west
    """

    return (
        np.degrees(
            np.arctan2(
                -u_component,
                -v_component,
            )
        )
        + 360.0
    ) % 360.0


def safe_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> np.ndarray:
    """Divide two arrays while avoiding division by zero."""

    result = np.full(
        numerator.shape,
        np.nan,
        dtype=float,
    )

    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator != 0)
    )

    np.divide(
        numerator,
        denominator,
        out=result,
        where=valid,
    )

    return result


# ============================================================================
# Fuel-specific feature construction
# ============================================================================

def add_direct_weighted_variables(
    output: pd.DataFrame,
    site_dataset: xr.Dataset,
    projects: pd.DataFrame,
    active_capacity: np.ndarray,
    fuel_prefix: str,
    variables: list[str],
) -> None:
    """Add direct capacity-weighted weather variables to output."""

    for variable in variables:
        site_values = weather_array(
            site_dataset,
            variable,
        )

        if site_values is None:
            continue

        project_values = project_values_from_sites(
            site_values,
            projects,
        )

        output[
            f"{fuel_prefix}_capacity_weighted_{variable}"
        ] = weighted_project_average(
            project_values,
            active_capacity,
        )


def build_wind_features(
    timestamps: pd.DatetimeIndex,
    site_dataset: xr.Dataset,
    projects: pd.DataFrame,
) -> pd.DataFrame:
    """Build capacity-weighted wind-fleet weather features."""

    output = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
        }
    )

    if projects.empty:
        output["wind_installed_capacity_mw"] = 0.0
        output["wind_active_project_count"] = 0
        output["wind_active_weather_site_count"] = 0

        return output

    active_capacity = active_capacity_matrix(
        timestamps,
        projects,
    )

    output["wind_installed_capacity_mw"] = (
        active_capacity.sum(axis=1)
    )

    output["wind_active_project_count"] = (
        (active_capacity > 0)
        .sum(axis=1)
        .astype(int)
    )

    site_ids = projects[
        "weather_site_id"
    ].to_numpy(dtype=int)

    active_site_count = np.zeros(
        len(timestamps),
        dtype=int,
    )

    for row_index in range(len(timestamps)):
        active_projects = active_capacity[
            row_index
        ] > 0

        if active_projects.any():
            active_site_count[row_index] = np.unique(
                site_ids[active_projects]
            ).size

    output[
        "wind_active_weather_site_count"
    ] = active_site_count

    add_direct_weighted_variables(
        output=output,
        site_dataset=site_dataset,
        projects=projects,
        active_capacity=active_capacity,
        fuel_prefix="wind",
        variables=WIND_DIRECT_WEIGHT_VARIABLES,
    )

    u100_sites = weather_array(
        site_dataset,
        "u_wind_100m",
    )

    v100_sites = weather_array(
        site_dataset,
        "v_wind_100m",
    )

    if (
        u100_sites is not None
        and v100_sites is not None
    ):
        u100_projects = project_values_from_sites(
            u100_sites,
            projects,
        )

        v100_projects = project_values_from_sites(
            v100_sites,
            projects,
        )

        speed_100m_projects = wind_speed(
            u100_projects,
            v100_projects,
        )

        weighted_u100 = weighted_project_average(
            u100_projects,
            active_capacity,
        )

        weighted_v100 = weighted_project_average(
            v100_projects,
            active_capacity,
        )

        output[
            "wind_capacity_weighted_speed_100m"
        ] = weighted_project_average(
            speed_100m_projects,
            active_capacity,
        )

        output[
            "wind_capacity_weighted_speed_100m_squared"
        ] = weighted_project_average(
            speed_100m_projects ** 2,
            active_capacity,
        )

        output[
            "wind_capacity_weighted_speed_100m_cubed"
        ] = weighted_project_average(
            speed_100m_projects ** 3,
            active_capacity,
        )

        output[
            "wind_speed_100m_spatial_std"
        ] = weighted_project_standard_deviation(
            speed_100m_projects,
            active_capacity,
        )

        output[
            "wind_vector_speed_100m"
        ] = wind_speed(
            weighted_u100,
            weighted_v100,
        )

        output[
            "wind_vector_direction_100m_degrees"
        ] = meteorological_wind_direction(
            weighted_u100,
            weighted_v100,
        )

    u10_sites = weather_array(
        site_dataset,
        "u_wind_10m",
    )

    v10_sites = weather_array(
        site_dataset,
        "v_wind_10m",
    )

    if (
        u10_sites is not None
        and v10_sites is not None
    ):
        u10_projects = project_values_from_sites(
            u10_sites,
            projects,
        )

        v10_projects = project_values_from_sites(
            v10_sites,
            projects,
        )

        speed_10m_projects = wind_speed(
            u10_projects,
            v10_projects,
        )

        weighted_u10 = weighted_project_average(
            u10_projects,
            active_capacity,
        )

        weighted_v10 = weighted_project_average(
            v10_projects,
            active_capacity,
        )

        output[
            "wind_capacity_weighted_speed_10m"
        ] = weighted_project_average(
            speed_10m_projects,
            active_capacity,
        )

        output[
            "wind_speed_10m_spatial_std"
        ] = weighted_project_standard_deviation(
            speed_10m_projects,
            active_capacity,
        )

        output[
            "wind_vector_speed_10m"
        ] = wind_speed(
            weighted_u10,
            weighted_v10,
        )

        output[
            "wind_vector_direction_10m_degrees"
        ] = meteorological_wind_direction(
            weighted_u10,
            weighted_v10,
        )

    if (
        "wind_capacity_weighted_speed_100m"
        in output.columns
        and "wind_capacity_weighted_speed_10m"
        in output.columns
    ):
        output[
            "wind_speed_shear_100m_minus_10m"
        ] = (
            output[
                "wind_capacity_weighted_speed_100m"
            ]
            - output[
                "wind_capacity_weighted_speed_10m"
            ]
        )

        output[
            "wind_speed_ratio_100m_to_10m"
        ] = safe_ratio(
            output[
                "wind_capacity_weighted_speed_100m"
            ].to_numpy(dtype=float),
            output[
                "wind_capacity_weighted_speed_10m"
            ].to_numpy(dtype=float),
        )

    if (
        "wind_capacity_weighted_surface_pressure"
        in output.columns
        and "wind_capacity_weighted_temperature_2m"
        in output.columns
    ):
        output[
            "wind_capacity_weighted_air_density"
        ] = (
            output[
                "wind_capacity_weighted_surface_pressure"
            ]
            / (
                AIR_GAS_CONSTANT
                * output[
                    "wind_capacity_weighted_temperature_2m"
                ]
            )
        )

    return output


def build_solar_features(
    timestamps: pd.DatetimeIndex,
    site_dataset: xr.Dataset,
    projects: pd.DataFrame,
) -> pd.DataFrame:
    """Build capacity-weighted solar-fleet weather features."""

    output = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
        }
    )

    if projects.empty:
        output["solar_installed_capacity_mw"] = 0.0
        output["solar_active_project_count"] = 0
        output["solar_active_weather_site_count"] = 0

        return output

    active_capacity = active_capacity_matrix(
        timestamps,
        projects,
    )

    output["solar_installed_capacity_mw"] = (
        active_capacity.sum(axis=1)
    )

    output["solar_active_project_count"] = (
        (active_capacity > 0)
        .sum(axis=1)
        .astype(int)
    )

    site_ids = projects[
        "weather_site_id"
    ].to_numpy(dtype=int)

    active_site_count = np.zeros(
        len(timestamps),
        dtype=int,
    )

    for row_index in range(len(timestamps)):
        active_projects = active_capacity[
            row_index
        ] > 0

        if active_projects.any():
            active_site_count[row_index] = np.unique(
                site_ids[active_projects]
            ).size

    output[
        "solar_active_weather_site_count"
    ] = active_site_count

    add_direct_weighted_variables(
        output=output,
        site_dataset=site_dataset,
        projects=projects,
        active_capacity=active_capacity,
        fuel_prefix="solar",
        variables=SOLAR_DIRECT_WEIGHT_VARIABLES,
    )

    radiation_column = (
        "solar_capacity_weighted_"
        "surface_solar_radiation_downwards"
    )

    clear_sky_column = (
        "solar_capacity_weighted_"
        "surface_solar_radiation_downwards_clear_sky"
    )

    if radiation_column in output.columns:
        # ERA5 accumulated radiation is normally J/m² over the hour.
        # Dividing by 3600 gives the hourly mean W/m².
        output[
            "solar_capacity_weighted_radiation_wm2"
        ] = (
            output[radiation_column]
            / 3600.0
        )

    if (
        radiation_column in output.columns
        and clear_sky_column in output.columns
    ):
        output[
            "solar_clear_sky_radiation_ratio"
        ] = safe_ratio(
            output[radiation_column].to_numpy(dtype=float),
            output[clear_sky_column].to_numpy(dtype=float),
        )

    return output


# ============================================================================
# Monthly processing
# ============================================================================

def process_month(
    path: Path,
    project_mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Build weighted renewable weather features for one month."""

    period = monthly_file_period(path)
    start = time.perf_counter()
    audit_rows: list[dict] = []

    with xr.open_dataset(path) as ds:
        required_dimensions = {
            "timestamp",
            "lat",
            "lon",
        }

        available_dimensions = (
            set(ds.dims)
            | set(ds.coords)
        )

        add_check(
            audit_rows,
            period,
            "required_spatial_dimensions",
            required_dimensions.issubset(
                available_dimensions
            ),
            observed=sorted(available_dimensions),
            expected=sorted(required_dimensions),
        )

        timestamps = pd.DatetimeIndex(
            pd.to_datetime(
                ds["timestamp"].values,
                utc=True,
            )
        )

        add_check(
            audit_rows,
            period,
            "expected_hour_count",
            len(timestamps)
            == expected_month_hours(period),
            observed=len(timestamps),
            expected=expected_month_hours(period),
        )

        add_check(
            audit_rows,
            period,
            "timestamps_unique",
            timestamps.is_unique,
            observed=int(
                timestamps.duplicated().sum()
            ),
            expected=0,
        )

        add_check(
            audit_rows,
            period,
            "timestamps_monotonic",
            timestamps.is_monotonic_increasing,
            observed=(
                timestamps.is_monotonic_increasing
            ),
            expected=True,
        )

        site_dataset = extract_project_sites(
            ds,
            project_mapping,
        ).load()

    wind_projects = project_mapping.loc[
        project_mapping["fuel_type"].eq("wind")
    ].copy()

    solar_projects = project_mapping.loc[
        project_mapping["fuel_type"].eq("solar")
    ].copy()

    wind_features = build_wind_features(
        timestamps=timestamps,
        site_dataset=site_dataset,
        projects=wind_projects,
    )

    solar_features = build_solar_features(
        timestamps=timestamps,
        site_dataset=site_dataset,
        projects=solar_projects,
    )

    features = wind_features.merge(
        solar_features,
        on="timestamp_utc",
        how="outer",
        validate="one_to_one",
    )

    features = (
        features
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )

    add_check(
        audit_rows,
        period,
        "output_row_count",
        len(features) == len(timestamps),
        observed=len(features),
        expected=len(timestamps),
    )

    add_check(
        audit_rows,
        period,
        "output_timestamp_unique",
        not features["timestamp_utc"].duplicated().any(),
        observed=int(
            features[
                "timestamp_utc"
            ].duplicated().sum()
        ),
        expected=0,
    )

    add_check(
        audit_rows,
        period,
        "wind_capacity_non_negative",
        (
            features[
                "wind_installed_capacity_mw"
            ]
            >= 0
        ).all(),
        observed=float(
            features[
                "wind_installed_capacity_mw"
            ].min()
        ),
        expected=">= 0",
    )

    add_check(
        audit_rows,
        period,
        "solar_capacity_non_negative",
        (
            features[
                "solar_installed_capacity_mw"
            ]
            >= 0
        ).all(),
        observed=float(
            features[
                "solar_installed_capacity_mw"
            ].min()
        ),
        expected=">= 0",
    )

    summary = {
        "period": period,
        "status": "processed",
        "pass": bool(
            pd.DataFrame(audit_rows)
            .loc[
                lambda frame: frame["severity"].eq("error"),
                "pass",
            ]
            .all()
        ),
        "rows": len(features),
        "columns": len(features.columns),
        "start": str(
            features["timestamp_utc"].min()
        ),
        "end": str(
            features["timestamp_utc"].max()
        ),
        "maximum_wind_capacity_mw": float(
            features[
                "wind_installed_capacity_mw"
            ].max()
        ),
        "maximum_solar_capacity_mw": float(
            features[
                "solar_installed_capacity_mw"
            ].max()
        ),
        "processing_seconds": round(
            time.perf_counter() - start,
            3,
        ),
        "source_file": str(path),
    }

    return (
        features,
        summary,
        pd.DataFrame(audit_rows),
    )


# ============================================================================
# Output and final-audit helpers
# ============================================================================

def ensure_output_directories() -> None:
    """Create feature and audit output directories."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def error_checks_pass(audit: pd.DataFrame) -> bool:
    """Return True when all error-level audit checks pass."""

    if audit.empty or "severity" not in audit.columns:
        return True

    checks = audit.loc[audit["severity"].eq("error"), "pass"]
    return bool(checks.all()) if not checks.empty else True


def audit_final_weather_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Run final checks on the concatenated hourly feature table."""

    rows: list[dict] = []
    timestamps = pd.DatetimeIndex(
        pd.to_datetime(frame["timestamp_utc"], utc=True)
    )

    add_check(rows, "final", "row_count_positive", len(frame) > 0, len(frame), "> 0")
    add_check(
        rows,
        "final",
        "timestamps_unique",
        not timestamps.has_duplicates,
        int(timestamps.duplicated().sum()),
        0,
    )
    add_check(
        rows,
        "final",
        "timestamps_monotonic",
        timestamps.is_monotonic_increasing,
        timestamps.is_monotonic_increasing,
        True,
    )

    if len(timestamps):
        expected = pd.date_range(
            timestamps.min(),
            timestamps.max(),
            freq="h",
            tz="UTC",
        )
        missing = expected.difference(timestamps)
    else:
        missing = pd.DatetimeIndex([])

    add_check(rows, "final", "hourly_continuity", len(missing) == 0, len(missing), 0)

    required_columns = {
        "timestamp_utc",
        "wind_installed_capacity_mw",
        "wind_active_project_count",
        "wind_active_weather_site_count",
        "solar_installed_capacity_mw",
        "solar_active_project_count",
        "solar_active_weather_site_count",
    }
    missing_columns = sorted(required_columns - set(frame.columns))
    add_check(
        rows,
        "final",
        "required_output_columns",
        not missing_columns,
        missing_columns,
        [],
    )

    for fuel in ["wind", "solar"]:
        capacity_column = f"{fuel}_installed_capacity_mw"
        project_count_column = f"{fuel}_active_project_count"
        site_count_column = f"{fuel}_active_weather_site_count"

        if capacity_column in frame.columns:
            negative = int((frame[capacity_column] < 0).sum())
            add_check(rows, "final", f"{fuel}_capacity_non_negative", negative == 0, negative, 0)

        if project_count_column in frame.columns:
            negative = int((frame[project_count_column] < 0).sum())
            add_check(rows, "final", f"{fuel}_project_count_non_negative", negative == 0, negative, 0)

        if site_count_column in frame.columns:
            negative = int((frame[site_count_column] < 0).sum())
            add_check(rows, "final", f"{fuel}_site_count_non_negative", negative == 0, negative, 0)

        if {project_count_column, site_count_column}.issubset(frame.columns):
            invalid = int((frame[site_count_column] > frame[project_count_column]).sum())
            add_check(rows, "final", f"{fuel}_site_count_not_above_project_count", invalid == 0, invalid, 0)

    return pd.DataFrame(rows)


def save_supporting_outputs(
    project_mapping: pd.DataFrame,
    monthly_summary: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    """Save mapping and audit outputs before canonical feature output."""

    project_mapping.to_csv(PROJECT_MAPPING_FILE, index=False)
    monthly_summary.to_csv(MONTHLY_SUMMARY_FILE, index=False)
    audit.to_csv(AUDIT_FILE, index=False)

    LOGGER.info("Saved project mapping: %s", PROJECT_MAPPING_FILE)
    LOGGER.info("Saved monthly summary: %s", MONTHLY_SUMMARY_FILE)
    LOGGER.info("Saved audit checks: %s", AUDIT_FILE)


def save_feature_outputs(
    frame: pd.DataFrame,
    write_csv: bool,
) -> None:
    """Save canonical Parquet and optional CSV outputs."""

    frame.to_parquet(OUTPUT_PARQUET, index=False)
    LOGGER.info("Saved Parquet output: %s", OUTPUT_PARQUET)

    if write_csv:
        frame.to_csv(OUTPUT_CSV, index=False)
        LOGGER.info("Saved CSV output: %s", OUTPUT_CSV)


def existing_output_result(
    write_csv: bool,
) -> dict[str, Any] | None:
    """Handle an existing canonical output without rebuilding NetCDF months."""

    if not OUTPUT_PARQUET.exists():
        return None

    frame: pd.DataFrame | None = None

    if write_csv and not OUTPUT_CSV.exists():
        LOGGER.info("Parquet exists; creating the requested CSV without rebuilding features.")
        frame = pd.read_parquet(OUTPUT_PARQUET)
        frame.to_csv(OUTPUT_CSV, index=False)

    if frame is None:
        frame = pd.read_parquet(OUTPUT_PARQUET, columns=["timestamp_utc"])

    timestamps = pd.to_datetime(frame["timestamp_utc"], utc=True)

    return {
        "dataset": "renewable_weather_features",
        "status": "skipped_existing",
        "pass": True,
        "rows": len(frame),
        "start": str(timestamps.min()),
        "end": str(timestamps.max()),
        "output_file": str(OUTPUT_PARQUET),
        "output_csv": str(OUTPUT_CSV) if OUTPUT_CSV.exists() else None,
    }


# ============================================================================
# Pipeline
# ============================================================================

def build_weather_features(
    overwrite: bool = False,
    write_csv: bool = False,
) -> dict[str, Any]:
    """Run the complete renewable-weather feature pipeline."""

    pipeline_start = time.perf_counter()
    ensure_output_directories()

    if not overwrite:
        existing = existing_output_result(write_csv=write_csv)
        if existing is not None:
            return existing

    monthly_files = get_monthly_files(ERA5_MONTHLY_DIR)
    projects = load_projects(
        wind_path=WIND_PROJECTS_FILE,
        solar_path=SOLAR_PROJECTS_FILE,
    )

    latitudes, longitudes = load_grid(monthly_files[0])
    project_mapping = map_projects_to_grid(
        projects=projects,
        latitudes=latitudes,
        longitudes=longitudes,
    )

    LOGGER.info("Projects loaded: %s", f"{len(project_mapping):,}")
    LOGGER.info("Wind projects: %s", f"{project_mapping['fuel_type'].eq('wind').sum():,}")
    LOGGER.info("Solar projects: %s", f"{project_mapping['fuel_type'].eq('solar').sum():,}")
    LOGGER.info(
        "Maximum project-to-grid distance: %.2f km",
        project_mapping["weather_distance_km"].max(),
    )

    monthly_outputs: list[pd.DataFrame] = []
    monthly_summaries: list[dict] = []
    monthly_audits: list[pd.DataFrame] = []

    for path in monthly_files:
        period = monthly_file_period(path)
        LOGGER.info("Building renewable weather features for %s", period)

        try:
            features, summary, audit = process_month(
                path=path,
                project_mapping=project_mapping,
            )
            monthly_outputs.append(features)
            monthly_summaries.append(summary)
            monthly_audits.append(audit)
        except Exception as exc:
            LOGGER.exception("Failed renewable-weather processing for %s", period)
            monthly_summaries.append(
                {
                    "period": period,
                    "status": "error",
                    "pass": False,
                    "error": repr(exc),
                    "source_file": str(path),
                }
            )
            monthly_audits.append(
                pd.DataFrame(
                    [{
                        "period": period,
                        "check": "process_month",
                        "pass": False,
                        "severity": "error",
                        "observed": repr(exc),
                        "expected": "successful monthly weather feature construction",
                        "notes": "",
                    }]
                )
            )

    monthly_summary_df = pd.DataFrame(monthly_summaries)
    audit_df = (
        pd.concat(monthly_audits, ignore_index=True)
        if monthly_audits
        else pd.DataFrame()
    )

    if not monthly_outputs:
        save_supporting_outputs(project_mapping, monthly_summary_df, audit_df)
        raise RuntimeError("No monthly weather-feature outputs were created.")

    master = pd.concat(monthly_outputs, ignore_index=True)
    master["timestamp_utc"] = pd.to_datetime(master["timestamp_utc"], utc=True)
    master = (
        master
        .drop_duplicates(subset=["timestamp_utc"], keep="last")
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )

    final_audit = audit_final_weather_features(master)
    audit_df = pd.concat([audit_df, final_audit], ignore_index=True)
    overall_pass = error_checks_pass(audit_df)

    save_supporting_outputs(project_mapping, monthly_summary_df, audit_df)

    if not overall_pass:
        LOGGER.error("Renewable-weather audit failed; canonical feature outputs were not written.")
        return {
            "dataset": "renewable_weather_features",
            "status": "audit_failed",
            "pass": False,
            "rows": len(master),
            "columns": len(master.columns),
            "project_mapping_file": str(PROJECT_MAPPING_FILE),
            "monthly_summary_file": str(MONTHLY_SUMMARY_FILE),
            "audit_file": str(AUDIT_FILE),
            "processing_seconds": round(time.perf_counter() - pipeline_start, 3),
        }

    save_feature_outputs(master, write_csv=write_csv)

    return {
        "dataset": "renewable_weather_features",
        "status": "saved",
        "pass": True,
        "rows": len(master),
        "columns": len(master.columns),
        "start": str(master["timestamp_utc"].min()),
        "end": str(master["timestamp_utc"].max()),
        "projects": len(project_mapping),
        "wind_projects": int(project_mapping["fuel_type"].eq("wind").sum()),
        "solar_projects": int(project_mapping["fuel_type"].eq("solar").sum()),
        "maximum_project_grid_distance_km": float(project_mapping["weather_distance_km"].max()),
        "output_file": str(OUTPUT_PARQUET),
        "output_csv": str(OUTPUT_CSV) if write_csv else None,
        "project_mapping_file": str(PROJECT_MAPPING_FILE),
        "monthly_summary_file": str(MONTHLY_SUMMARY_FILE),
        "audit_file": str(AUDIT_FILE),
        "processing_seconds": round(time.perf_counter() - pipeline_start, 3),
    }


def print_result(result: dict[str, Any]) -> None:
    """Print a compact final pipeline report."""

    print("\n" + "=" * 80)
    print("RENEWABLE WEATHER FEATURE RESULT")
    print("=" * 80)

    for key, value in result.items():
        print(f"{key}: {value}")

    print("=" * 80)


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build capacity-weighted renewable weather features "
            "from spatial monthly ERA5 files."
        )
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild outputs even when the canonical Parquet already exists.",
    )
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Write a full CSV copy in addition to canonical Parquet.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )

    args = parser.parse_args()
    configure_logging(verbose=args.verbose)

    try:
        result = build_weather_features(
            overwrite=args.overwrite,
            write_csv=args.write_csv,
        )
    except Exception:
        LOGGER.exception("Renewable-weather feature pipeline failed.")
        raise SystemExit(1)

    print_result(result)

    if not result.get("pass", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
