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

Project phases are represented as separate rows. Exact commissioning dates
are used when available. A source that contains only ``commissioning_year``
is retained as an explicitly flagged year-level estimate beginning January 1;
the output records how many active projects use estimated dates.

Expected project registry columns
---------------------------------
project_id
project_name
fuel_type
latitude
longitude
capacity_mw
commissioning_date       preferred
commissioning_year       accepted fallback
retirement_date       optional

Supported fuel_type values:
wind
solar

Outputs
-------
data/processed/feature_engineering/weather/renewable_weather_features_hourly.parquet
data/audits/feature_engineering/renewable_project_weather_mapping.csv
data/audits/feature_engineering/weather_features_monthly_summary.csv
data/audits/feature_engineering/weather_features_audit_checks.csv

Run
---
python src/feature_engineering/renewable_weather_features.py

or:

.venv/bin/python src/feature_engineering/renewable_weather_features.py

Information timing
------------------
ERA5 values are contemporaneous observed weather. They are explanatory
features unless replaced with weather forecasts for operational modeling.
"""

# ============================================================================
# Imports
# ============================================================================

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

try:
    from .shared import (
        add_period_check as add_check,
        audit_passed,
        build_manifest,
        configure_logging,
        ensure_directories,
        feature_code_paths,
        ensure_src_on_path,
        existing_outputs_satisfy_request as outputs_satisfy_request,
        expected_month_hours,
        get_monthly_files,
        haversine_distance_km,
        load_grid,
        monthly_file_period,
        nearest_coordinate,
        output_is_current,
        require_columns,
        safe_divide as safe_ratio,
        save_feature_outputs as write_feature_outputs,
        save_tables,
        weather_array,
        weighted_average as weighted_project_average,
        weighted_standard_deviation as weighted_project_standard_deviation,
        write_manifest,
    )
except ImportError:  # Support direct execution of this file.
    from shared import (
        add_period_check as add_check,
        audit_passed,
        build_manifest,
        configure_logging,
        ensure_directories,
        feature_code_paths,
        ensure_src_on_path,
        existing_outputs_satisfy_request as outputs_satisfy_request,
        expected_month_hours,
        get_monthly_files,
        haversine_distance_km,
        load_grid,
        monthly_file_period,
        nearest_coordinate,
        output_is_current,
        require_columns,
        safe_divide as safe_ratio,
        save_feature_outputs as write_feature_outputs,
        save_tables,
        weather_array,
        weighted_average as weighted_project_average,
        weighted_standard_deviation as weighted_project_standard_deviation,
        write_manifest,
    )

ensure_src_on_path(__file__)


# ============================================================================
# Logging
# ============================================================================

LOGGER = logging.getLogger(__name__)

# ============================================================================
# Project paths
# ============================================================================
from config import (
    PROJECT_ROOT,
    PREPROCESSING_DIR,
    FEATURES_DIR,
    FEATURE_ENGINEERING_AUDITS_DIR as AUDIT_DIR,
    ERA5_MONTHLY_STANDARDIZED_DIR as ERA5_MONTHLY_DIR,
)

OUTPUT_DIR = FEATURES_DIR / "weather"

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

# ============================================================================
# Configuration
# ============================================================================

VALID_FUEL_TYPES = {
    "wind",
    "solar",
}
FEATURE_INFORMATION_POLICY = "contemporaneous_observed_weather"

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

# Union of variables required by either fleet-specific feature builder.
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



# ============================================================================
# Project registry
# ============================================================================

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

    projects = projects.copy()

    # Normalize source headings once at the ingestion boundary.
    projects.columns = (
        projects.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    # Wind generally has an identifier; solar identifiers are generated from
    # names. A stable fuel prefix prevents cross-fuel collisions.
    if "project_identifier" in projects.columns:
        projects = projects.rename(
            columns={
                "project_identifier": "project_id",
            }
        )

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

    projects["facility_id"] = projects["project_id"].astype("string")

    projects["fuel_type"] = fuel_type

    exact_dates = (
        pd.to_datetime(
            projects["commissioning_date"],
            errors="coerce",
            utc=True,
        )
        if "commissioning_date" in projects.columns
        else pd.Series(pd.NaT, index=projects.index, dtype="datetime64[ns, UTC]")
    )

    commissioning_year = pd.to_numeric(
        projects["commissioning_year"]
        if "commissioning_year" in projects.columns
        else pd.Series(np.nan, index=projects.index),
        errors="coerce",
    )
    estimated_dates = pd.to_datetime(
        commissioning_year.astype("Int64").astype("string") + "-01-01",
        errors="coerce",
        utc=True,
    )
    projects["commissioning_date"] = exact_dates.fillna(estimated_dates)
    projects["commissioning_date_precision"] = np.where(
        exact_dates.notna(),
        "exact_date",
        "year_estimate",
    )

    # Make every phase independently addressable without rejecting multiple
    # capacity additions for the same facility.
    phase_number = projects.groupby("facility_id", sort=False).cumcount() + 1
    phase_count = projects.groupby("facility_id", sort=False)["facility_id"].transform(
        "size"
    )
    projects["project_id"] = np.where(
        phase_count.gt(1),
        projects["facility_id"].astype(str)
        + "__PHASE_"
        + phase_number.astype(str),
        projects["facility_id"].astype(str),
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

    # Historical weighting requires either an exact date or a source year.
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
            f"{fuel_type.capitalize()} projects are missing commissioning dates "
            "and years. Fill one before building historical weather weights:\n"
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

        distance = haversine_distance_km(
            float(row.latitude),
            float(row.longitude),
            weather_lat,
            weather_lon,
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

    missing_variables = sorted(
        set(ALL_SOURCE_VARIABLES) - set(ds.data_vars)
    )
    if missing_variables:
        raise ValueError(
            "Monthly ERA5 file is missing required renewable-weather "
            f"variables: {missing_variables}"
        )

    extracted = ds[ALL_SOURCE_VARIABLES].sel(
        lat=lat_indexer,
        lon=lon_indexer,
        method="nearest",
    )

    return extracted


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
        pd.to_datetime(
            projects["commissioning_date"],
            utc=True,
            errors="raise",
        )
        .dt.tz_localize(None)
        .to_numpy(dtype="datetime64[ns]")
    )[None, :]

    retirement_series = projects["retirement_date"]

    retirement = (
        pd.to_datetime(
            retirement_series,
            utc=True,
            errors="coerce",
        )
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

    # Direct fields are weighted over the active fleet independently for each
    # hour. Missing project weather is removed from both numerator and weight.
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
        output["wind_estimated_commissioning_project_count"] = 0

        return output

    # Historical capacity determines both the fleet totals and every weather
    # weight. Projects contribute nothing outside their operating interval.
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

    # Retain visible evidence of projects whose activation is estimated from
    # commissioning year rather than known to an exact source date.
    estimated_commissioning = projects[
        "commissioning_date_precision"
    ].eq("year_estimate").to_numpy()
    output["wind_estimated_commissioning_project_count"] = (
        ((active_capacity > 0) & estimated_commissioning[None, :])
        .sum(axis=1)
        .astype(int)
    )

    # Several projects can share one ERA5 grid cell, so site count is measured
    # separately from project count.
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

    # Compute speed at each project before capacity weighting. This preserves
    # wind-speed magnitude and supports squared/cubed generation-response terms.
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

    # Ten-metre fields provide near-surface context and enable a simple shear
    # comparison with the generation-relevant 100-metre level.
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

    # Speed shear and ratios describe vertical wind-profile differences.
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

    # The ideal-gas approximation uses pressure in Pa and temperature in K.
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
        output["solar_estimated_commissioning_project_count"] = 0

        return output

    # Use the same historical activation and audit conventions as wind.
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

    estimated_commissioning = projects[
        "commissioning_date_precision"
    ].eq("year_estimate").to_numpy()
    output["solar_estimated_commissioning_project_count"] = (
        ((active_capacity > 0) & estimated_commissioning[None, :])
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

    # ERA5 accumulated radiation is J/m² over the hour; dividing by 3,600
    # converts it to the hourly mean W/m².
    if radiation_column in output.columns:
        output[
            "solar_capacity_weighted_radiation_wm2"
        ] = (
            output[radiation_column]
            / 3600.0
        )

    # The clear-sky ratio is dimensionless and missing when its denominator is
    # zero, including nighttime hours.
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
# Audit and output helpers
# ============================================================================

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
        "wind_estimated_commissioning_project_count",
        "solar_estimated_commissioning_project_count",
        *{
            f"wind_capacity_weighted_{variable}"
            for variable in WIND_DIRECT_WEIGHT_VARIABLES
        },
        *{
            f"solar_capacity_weighted_{variable}"
            for variable in SOLAR_DIRECT_WEIGHT_VARIABLES
        },
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
            add_check(
                rows, "final", f"{fuel}_capacity_non_negative", negative == 0,
                negative, 0,
            )

        if project_count_column in frame.columns:
            negative = int((frame[project_count_column] < 0).sum())
            add_check(
                rows, "final", f"{fuel}_project_count_non_negative",
                negative == 0, negative, 0,
            )

        if site_count_column in frame.columns:
            negative = int((frame[site_count_column] < 0).sum())
            add_check(
                rows, "final", f"{fuel}_site_count_non_negative",
                negative == 0, negative, 0,
            )

        if {project_count_column, site_count_column}.issubset(frame.columns):
            invalid = int(
                (frame[site_count_column] > frame[project_count_column]).sum()
            )
            add_check(
                rows, "final", f"{fuel}_site_count_not_above_project_count",
                invalid == 0, invalid, 0,
            )

        active_mask = frame[capacity_column].gt(0)
        direct_prefix = f"{fuel}_capacity_weighted_"
        direct_columns = [
            column
            for column in frame.columns
            if column.startswith(direct_prefix)
            and not column.endswith(("_squared", "_cubed", "_air_density"))
        ]
        missing_active_weather = int(
            frame.loc[active_mask, direct_columns].isna().sum().sum()
        )
        add_check(
            rows,
            "final",
            f"{fuel}_active_weather_complete",
            missing_active_weather == 0,
            missing_active_weather,
            0,
        )

    return pd.DataFrame(rows)


def existing_output_result(
    write_csv: bool,
    expected_manifest: dict[str, Any],
) -> dict[str, Any] | None:
    """Handle an existing canonical output without rebuilding NetCDF months."""

    output_complete = outputs_satisfy_request(
        OUTPUT_PARQUET,
        OUTPUT_CSV,
        write_csv,
        expected_manifest,
        required_artifacts=[
            PROJECT_MAPPING_FILE,
            MONTHLY_SUMMARY_FILE,
            AUDIT_FILE,
        ],
    )
    evidence_paths = [PROJECT_MAPPING_FILE, MONTHLY_SUMMARY_FILE, AUDIT_FILE]
    evidence_current = all(
        output_is_current(path, expected_manifest)
        for path in evidence_paths
    )
    if not output_complete and (
        not output_is_current(OUTPUT_PARQUET, expected_manifest)
        or not evidence_current
    ):
        return None

    frame: pd.DataFrame | None = None

    if write_csv and not output_is_current(OUTPUT_CSV, expected_manifest):
        LOGGER.info(
            "Parquet exists; creating the requested CSV without rebuilding."
        )
        frame = pd.read_parquet(OUTPUT_PARQUET)
        frame.to_csv(OUTPUT_CSV, index=False)
        write_manifest(OUTPUT_CSV, expected_manifest)

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
    ensure_directories(OUTPUT_DIR, AUDIT_DIR)

    monthly_files = get_monthly_files(ERA5_MONTHLY_DIR)
    expected_manifest = build_manifest(
        dataset="renewable_weather_features",
        source_paths=[
            WIND_PROJECTS_FILE,
            SOLAR_PROJECTS_FILE,
            *monthly_files,
        ],
        code_paths=feature_code_paths(Path(__file__)),
        configuration={
            "feature_information_policy": FEATURE_INFORMATION_POLICY,
            "source_variables": ALL_SOURCE_VARIABLES,
            "commissioning_year_fallback": "january_1_flagged_estimate",
        },
    )

    if not overwrite:
        existing = existing_output_result(
            write_csv=write_csv,
            expected_manifest=expected_manifest,
        )
        if existing is not None:
            return existing

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
    LOGGER.info(
        "Wind projects: %s",
        f"{project_mapping['fuel_type'].eq('wind').sum():,}",
    )
    LOGGER.info(
        "Solar projects: %s",
        f"{project_mapping['fuel_type'].eq('solar').sum():,}",
    )
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
        save_tables(
            {
                PROJECT_MAPPING_FILE: project_mapping,
                MONTHLY_SUMMARY_FILE: monthly_summary_df,
                AUDIT_FILE: audit_df,
            }
        )
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
    overall_pass = audit_passed(audit_df)

    save_tables(
        {
            PROJECT_MAPPING_FILE: project_mapping,
            MONTHLY_SUMMARY_FILE: monthly_summary_df,
            AUDIT_FILE: audit_df,
        }
    )

    if not overall_pass:
        LOGGER.error(
            "Renewable-weather audit failed; canonical outputs were not written."
        )
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

    write_feature_outputs(
        master,
        OUTPUT_PARQUET,
        OUTPUT_CSV,
        write_csv,
        "renewable-weather feature",
        manifest=expected_manifest,
    )
    for artifact in [PROJECT_MAPPING_FILE, MONTHLY_SUMMARY_FILE, AUDIT_FILE]:
        write_manifest(artifact, expected_manifest)
    provenance_file = write_manifest(OUTPUT_PARQUET, expected_manifest)

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
        "maximum_project_grid_distance_km": float(
            project_mapping["weather_distance_km"].max()
        ),
        "output_file": str(OUTPUT_PARQUET),
        "output_csv": str(OUTPUT_CSV) if write_csv else None,
        "project_mapping_file": str(PROJECT_MAPPING_FILE),
        "monthly_summary_file": str(MONTHLY_SUMMARY_FILE),
        "audit_file": str(AUDIT_FILE),
        "manifest_file": str(provenance_file),
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
    """Run the renewable-weather pipeline from the command line."""
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
