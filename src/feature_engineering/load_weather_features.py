# src/feature_engineering/load_weather_features.py

"""
Build hourly load-relevant weather features for Alberta using actual AESO
hourly regional load.

Pipeline
--------
1. Load preprocessed AESO hourly regional load.
2. Define one representative weather location for each AESO region.
3. Map each regional location to the nearest ERA5 grid cell.
4. Extract hourly weather at those regional grid cells.
5. Join each ERA5 month to the corresponding hourly AESO regional loads.
6. Weight regional weather by actual hourly regional load shares.
7. Apply reusable load-weather feature builders.
8. Save one canonical hourly Parquet file.

Hourly regional weighting
-------------------------
For region r and timestamp t:

    load_share(r, t) =
        regional_load_mw(r, t)
        / total_region_load_mw(t)

For weather variable x:

    load_weighted_x(t) =
        sum_r[x(r, t) * load_share(r, t)]

The load shares therefore change every hour and reflect the actual spatial
distribution of Alberta load rather than provisional static estimates.

Input
-----
data/preprocessing/area_load_preprocessed.parquet

Required regional load columns:
    calgary_load_mw
    central_load_mw
    edmonton_load_mw
    northeast_load_mw
    northwest_load_mw
    south_load_mw
    total_region_load_mw
    area_load_imputed

Output
------
data/features/weather/load_weather_features_hourly.parquet
data/features/weather/load_region_weather_mapping.csv
data/audits/load_weather_features_monthly_summary.csv
data/audits/load_weather_features_audit_checks.csv
"""

from __future__ import annotations

import argparse
import calendar
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import xarray as xr


# ============================================================================
# Paths
# ============================================================================

PROJECT_ROOT = Path(
    "/Users/brodiehasein/alberta_power_markets_project"
)

ERA5_MONTHLY_DIR = (
    PROJECT_ROOT
    / "data/preprocessing/weather/era5/monthly_standardized"
)

AREA_LOAD_FILE = (
    PROJECT_ROOT
    / "data/preprocessing/area_load_preprocessed.parquet"
)

LOAD_REGIONS_FILE = (
    PROJECT_ROOT
    / "data/preprocessing/load_regions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/features/weather"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "load_weather_features_hourly.parquet"
)

LOAD_REGION_MAPPING_FILE = (
    OUTPUT_DIR
    / "load_region_weather_mapping.csv"
)

AUDIT_DIR = PROJECT_ROOT / "data/audits"

MONTHLY_SUMMARY_FILE = (
    AUDIT_DIR
    / "load_weather_features_monthly_summary.csv"
)

AUDIT_FILE = (
    AUDIT_DIR
    / "load_weather_features_audit_checks.csv"
)


# ============================================================================
# Configuration
# ============================================================================

REGION_LOAD_COLUMNS = {
    "calgary": "calgary_load_mw",
    "central": "central_load_mw",
    "edmonton": "edmonton_load_mw",
    "northeast": "northeast_load_mw",
    "northwest": "northwest_load_mw",
    "south": "south_load_mw",
}

# Representative weather points for the six AESO regions.
# A user-maintained data/preprocessing/load_regions.csv overrides these values.
DEFAULT_LOAD_REGIONS = pd.DataFrame(
    [
        {
            "region": "calgary",
            "representative_location": "Calgary",
            "latitude": 51.0447,
            "longitude": -114.0719,
        },
        {
            "region": "central",
            "representative_location": "Red Deer",
            "latitude": 52.2681,
            "longitude": -113.8112,
        },
        {
            "region": "edmonton",
            "representative_location": "Edmonton",
            "latitude": 53.5461,
            "longitude": -113.4938,
        },
        {
            "region": "northeast",
            "representative_location": "Fort McMurray",
            "latitude": 56.7264,
            "longitude": -111.3803,
        },
        {
            "region": "northwest",
            "representative_location": "Grande Prairie",
            "latitude": 55.1707,
            "longitude": -118.7884,
        },
        {
            "region": "south",
            "representative_location": "Lethbridge",
            "latitude": 49.6956,
            "longitude": -112.8451,
        },
    ]
)

SOURCE_VARIABLES = [
    "temperature_2m",
    "dewpoint_2m",
    "u_wind_10m",
    "v_wind_10m",
    "surface_solar_radiation_downwards",
    "total_cloud_cover",
    "total_precipitation",
    "snowfall",
    "snow_depth",
    "surface_pressure",
    "mean_sea_level_pressure",
]


# ============================================================================
# General helpers
# ============================================================================

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


def monthly_file_period(
    path: Path,
) -> str:
    parts = path.stem.split("_")
    return f"{parts[-2]}-{parts[-1]}"


def expected_month_hours(
    period: str,
) -> int:
    year, month = map(
        int,
        period.split("-"),
    )

    return (
        calendar.monthrange(
            year,
            month,
        )[1]
        * 24
    )


# ============================================================================
# AESO hourly regional load
# ============================================================================

def load_hourly_area_load(
    path: Path = AREA_LOAD_FILE,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing preprocessed AESO area-load file: {path}"
        )

    load = pd.read_parquet(path)

    required_columns = {
        "timestamp_utc",
        "total_region_load_mw",
        "area_load_imputed",
        *REGION_LOAD_COLUMNS.values(),
    }

    missing = sorted(
        required_columns
        - set(load.columns)
    )

    if missing:
        raise ValueError(
            f"AESO area-load data is missing columns: {missing}"
        )

    load = load[
        [
            "timestamp_utc",
            *REGION_LOAD_COLUMNS.values(),
            "total_region_load_mw",
            "area_load_imputed",
        ]
    ].copy()

    load["timestamp_utc"] = pd.to_datetime(
        load["timestamp_utc"],
        utc=True,
    )

    numeric_columns = [
        *REGION_LOAD_COLUMNS.values(),
        "total_region_load_mw",
        "area_load_imputed",
    ]

    for column in numeric_columns:
        load[column] = pd.to_numeric(
            load[column],
            errors="raise",
        )

    load = (
        load
        .drop_duplicates(
            subset=["timestamp_utc"],
            keep="last",
        )
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )

    if load["timestamp_utc"].duplicated().any():
        raise ValueError(
            "AESO area-load table contains duplicate timestamps."
        )

    if load[numeric_columns].isna().any().any():
        missing_columns = (
            load[numeric_columns]
            .columns[
                load[numeric_columns]
                .isna()
                .any()
            ]
            .tolist()
        )

        raise ValueError(
            "AESO area-load table contains missing values in: "
            f"{missing_columns}"
        )

    if (
        load["total_region_load_mw"]
        <= 0
    ).any():
        raise ValueError(
            "total_region_load_mw must be positive for every hour."
        )

    return add_region_load_shares(
        load
    )


def add_region_load_shares(
    load: pd.DataFrame,
) -> pd.DataFrame:
    load = load.copy()

    share_columns = []

    for (
        region,
        load_column,
    ) in REGION_LOAD_COLUMNS.items():
        share_column = (
            f"{region}_load_share"
        )

        load[share_column] = (
            load[load_column]
            / load["total_region_load_mw"]
        )

        share_columns.append(
            share_column
        )

    share_sum = load[
        share_columns
    ].sum(axis=1)

    maximum_error = float(
        np.abs(
            share_sum - 1.0
        ).max()
    )

    if not np.allclose(
        share_sum,
        1.0,
        atol=1e-6,
    ):
        raise ValueError(
            "Hourly regional load shares do not sum to one. "
            f"Maximum error: {maximum_error}"
        )

    if (
        load[share_columns]
        < 0
    ).any().any():
        raise ValueError(
            "Regional load shares contain negative values."
        )

    return load


# ============================================================================
# Regional weather locations
# ============================================================================

def load_load_regions(
    path: Path = LOAD_REGIONS_FILE,
) -> pd.DataFrame:
    regions = (
        pd.read_csv(path)
        if path.exists()
        else DEFAULT_LOAD_REGIONS.copy()
    )

    source = (
        str(path)
        if path.exists()
        else "built-in representative regional locations"
    )

    regions = regions.copy()

    regions.columns = (
        regions.columns
        .astype(str)
        .str.strip()
        .str.lower()
    )

    required_columns = {
        "region",
        "latitude",
        "longitude",
    }

    missing = sorted(
        required_columns
        - set(regions.columns)
    )

    if missing:
        raise ValueError(
            f"Load-region configuration is missing columns: {missing}"
        )

    if (
        "representative_location"
        not in regions.columns
    ):
        regions[
            "representative_location"
        ] = regions["region"]

    regions["region"] = (
        regions["region"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    regions[
        "representative_location"
    ] = (
        regions[
            "representative_location"
        ]
        .astype(str)
        .str.strip()
    )

    for column in [
        "latitude",
        "longitude",
    ]:
        regions[column] = pd.to_numeric(
            regions[column],
            errors="raise",
        )

    expected_regions = set(
        REGION_LOAD_COLUMNS
    )

    observed_regions = set(
        regions["region"]
    )

    missing_regions = sorted(
        expected_regions
        - observed_regions
    )

    extra_regions = sorted(
        observed_regions
        - expected_regions
    )

    if missing_regions:
        raise ValueError(
            "Load-region configuration is missing regions: "
            f"{missing_regions}"
        )

    if extra_regions:
        raise ValueError(
            "Load-region configuration has unexpected regions: "
            f"{extra_regions}"
        )

    if regions["region"].duplicated().any():
        duplicates = (
            regions.loc[
                regions["region"]
                .duplicated(
                    keep=False
                ),
                "region",
            ]
            .tolist()
        )

        raise ValueError(
            f"Load-region names must be unique: {duplicates}"
        )

    regions = (
        regions
        .set_index("region")
        .loc[
            list(
                REGION_LOAD_COLUMNS
                .keys()
            )
        ]
        .reset_index()
    )

    regions["load_column"] = (
        regions["region"]
        .map(
            REGION_LOAD_COLUMNS
        )
    )

    regions["share_column"] = (
        regions["region"]
        + "_load_share"
    )

    regions["location_source"] = source

    return regions


# ============================================================================
# ERA5 spatial mapping
# ============================================================================

def get_monthly_files(
    monthly_dir: Path,
) -> list[Path]:
    files = sorted(
        monthly_dir.glob(
            "era5_alberta_standardized_*.nc"
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No monthly ERA5 files found in {monthly_dir}"
        )

    return files


def load_grid(
    reference_file: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    with xr.open_dataset(
        reference_file
    ) as ds:
        required = {
            "lat",
            "lon",
            "timestamp",
        }

        available = (
            set(ds.coords)
            | set(ds.dims)
        )

        missing = sorted(
            required
            - available
        )

        if missing:
            raise ValueError(
                f"{reference_file.name} is missing coordinates: {missing}"
            )

        latitudes = np.asarray(
            ds["lat"].values,
            dtype=float,
        )

        longitudes = np.asarray(
            ds["lon"].values,
            dtype=float,
        )

    return (
        latitudes,
        longitudes,
    )


def nearest_coordinate(
    value: float,
    coordinates: np.ndarray,
) -> tuple[
    float,
    int,
]:
    position = int(
        np.abs(
            coordinates - value
        ).argmin()
    )

    return (
        float(
            coordinates[position]
        ),
        position,
    )


def haversine_distance_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    radius = 6371.0088

    p1 = np.radians(lat1)
    p2 = np.radians(lat2)

    delta_latitude = p2 - p1
    delta_longitude = np.radians(
        lon2 - lon1
    )

    a = (
        np.sin(
            delta_latitude / 2
        ) ** 2
        + np.cos(p1)
        * np.cos(p2)
        * np.sin(
            delta_longitude / 2
        ) ** 2
    )

    return float(
        radius
        * 2
        * np.arcsin(
            np.sqrt(a)
        )
    )


def map_load_regions_to_grid(
    regions: pd.DataFrame,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> pd.DataFrame:
    mapped = regions.copy()

    mapping_rows = []

    for row in mapped.itertuples(
        index=False
    ):
        (
            weather_latitude,
            latitude_position,
        ) = nearest_coordinate(
            float(row.latitude),
            latitudes,
        )

        (
            weather_longitude,
            longitude_position,
        ) = nearest_coordinate(
            float(row.longitude),
            longitudes,
        )

        distance = haversine_distance_km(
            float(row.latitude),
            float(row.longitude),
            weather_latitude,
            weather_longitude,
        )

        mapping_rows.append(
            (
                weather_latitude,
                weather_longitude,
                latitude_position,
                longitude_position,
                distance,
            )
        )

    mapped[
        [
            "weather_lat",
            "weather_lon",
            "weather_lat_position",
            "weather_lon_position",
            "weather_distance_km",
        ]
    ] = mapping_rows

    unique_sites = (
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
        .reset_index(
            drop=True
        )
    )

    unique_sites[
        "weather_site_id"
    ] = np.arange(
        len(unique_sites),
        dtype=int,
    )

    mapped = mapped.merge(
        unique_sites,
        on=[
            "weather_lat",
            "weather_lon",
        ],
        how="left",
        validate="many_to_one",
    )

    mapped[
        "weather_site_id"
    ] = (
        mapped[
            "weather_site_id"
        ]
        .astype(int)
    )

    return mapped


def extract_load_region_sites(
    ds: xr.Dataset,
    mapping: pd.DataFrame,
) -> xr.Dataset:
    unique_sites = (
        mapping[
            [
                "weather_site_id",
                "weather_lat",
                "weather_lon",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "weather_site_id"
        )
        .reset_index(
            drop=True
        )
    )

    site_ids = (
        unique_sites[
            "weather_site_id"
        ]
        .to_numpy(
            dtype=int
        )
    )

    latitude_indexer = xr.DataArray(
        unique_sites[
            "weather_lat"
        ].to_numpy(
            dtype=float
        ),
        dims="weather_site_id",
        coords={
            "weather_site_id": site_ids,
        },
    )

    longitude_indexer = xr.DataArray(
        unique_sites[
            "weather_lon"
        ].to_numpy(
            dtype=float
        ),
        dims="weather_site_id",
        coords={
            "weather_site_id": site_ids,
        },
    )

    variables = [
        variable
        for variable in SOURCE_VARIABLES
        if variable in ds.data_vars
    ]

    if not variables:
        raise ValueError(
            "Monthly ERA5 file contains none of the configured "
            "load-weather variables."
        )

    return ds[
        variables
    ].sel(
        lat=latitude_indexer,
        lon=longitude_indexer,
        method="nearest",
    )


def weather_array(
    site_dataset: xr.Dataset,
    variable: str,
) -> np.ndarray | None:
    if variable not in site_dataset.data_vars:
        return None

    return np.asarray(
        site_dataset[
            variable
        ]
        .transpose(
            "timestamp",
            "weather_site_id",
        )
        .values,
        dtype=float,
    )


def region_values_from_sites(
    site_values: np.ndarray,
    mapping: pd.DataFrame,
) -> np.ndarray:
    site_positions = (
        mapping[
            "weather_site_id"
        ]
        .to_numpy(
            dtype=int
        )
    )

    return site_values[
        :,
        site_positions,
    ]

# ============================================================================
# Derived meteorological calculations
# ============================================================================

def relative_humidity_from_temperature_dewpoint(
    temperature_c: np.ndarray,
    dewpoint_c: np.ndarray,
) -> np.ndarray:
    """
    Calculate relative humidity from air temperature and dew point.

    Uses the Magnus approximation. Output is clipped to 0–100 percent.
    """

    saturation_vapour_pressure = np.exp(
        (17.625 * temperature_c)
        / (243.04 + temperature_c)
    )

    actual_vapour_pressure = np.exp(
        (17.625 * dewpoint_c)
        / (243.04 + dewpoint_c)
    )

    relative_humidity = (
        100.0
        * actual_vapour_pressure
        / saturation_vapour_pressure
    )

    return np.clip(
        relative_humidity,
        0.0,
        100.0,
    )


def wet_bulb_temperature_c(
    temperature_c: np.ndarray,
    relative_humidity_pct: np.ndarray,
) -> np.ndarray:
    """
    Approximate wet-bulb temperature using the Stull formula.

    Inputs
    ------
    temperature_c:
        Air temperature in degrees Celsius.

    relative_humidity_pct:
        Relative humidity in percent.

    Returns
    -------
    Approximate wet-bulb temperature in degrees Celsius.
    """

    rh = np.clip(
        relative_humidity_pct,
        0.0,
        100.0,
    )

    wet_bulb = (
        temperature_c
        * np.arctan(
            0.151977
            * np.sqrt(
                rh + 8.313659
            )
        )
        + np.arctan(
            temperature_c
            + rh
        )
        - np.arctan(
            rh - 1.676331
        )
        + 0.00391838
        * rh ** 1.5
        * np.arctan(
            0.023101
            * rh
        )
        - 4.686035
    )

    return wet_bulb


def heat_index_c(
    temperature_c: np.ndarray,
    relative_humidity_pct: np.ndarray,
) -> np.ndarray:
    """
    Calculate heat index using the Rothfusz regression.

    Heat index is returned only when:
        temperature >= 26.7 C
        relative humidity >= 40 percent

    Values outside those conditions are NaN because the index is not
    meteorologically applicable.
    """

    temperature_f = (
        temperature_c
        * 9.0
        / 5.0
        + 32.0
    )

    rh = relative_humidity_pct

    heat_index_f = (
        -42.379
        + 2.04901523
        * temperature_f
        + 10.14333127
        * rh
        - 0.22475541
        * temperature_f
        * rh
        - 0.00683783
        * temperature_f ** 2
        - 0.05481717
        * rh ** 2
        + 0.00122874
        * temperature_f ** 2
        * rh
        + 0.00085282
        * temperature_f
        * rh ** 2
        - 0.00000199
        * temperature_f ** 2
        * rh ** 2
    )

    heat_index_celsius = (
        heat_index_f - 32.0
    ) * 5.0 / 9.0

    applicable = (
        np.isfinite(
            temperature_c
        )
        & np.isfinite(
            rh
        )
        & (
            temperature_c
            >= 26.7
        )
        & (
            rh
            >= 40.0
        )
    )

    return np.where(
        applicable,
        heat_index_celsius,
        np.nan,
    )


def wind_chill_temperature_c(
    temperature_c: np.ndarray,
    wind_speed_ms: np.ndarray,
) -> np.ndarray:
    """
    Calculate Canadian wind-chill temperature.

    Wind chill is returned only when:
        temperature <= 10 C
        wind speed >= 4.8 km/h

    Values outside those conditions are NaN.
    """

    wind_speed_kmh = (
        wind_speed_ms
        * 3.6
    )

    wind_chill = (
        13.12
        + 0.6215
        * temperature_c
        - 11.37
        * wind_speed_kmh ** 0.16
        + 0.3965
        * temperature_c
        * wind_speed_kmh ** 0.16
    )

    applicable = (
        np.isfinite(
            temperature_c
        )
        & np.isfinite(
            wind_speed_kmh
        )
        & (
            temperature_c
            <= 10.0
        )
        & (
            wind_speed_kmh
            >= 4.8
        )
    )

    return np.where(
        applicable,
        wind_chill,
        np.nan,
    )

# ============================================================================
# Hourly weighting
# ============================================================================

def hourly_weighted_average(
    values: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    if values.shape != weights.shape:
        raise ValueError(
            "Weather values and hourly weights must have the same shape. "
            f"Observed values={values.shape}, weights={weights.shape}"
        )

    valid = (
        np.isfinite(values)
        & np.isfinite(weights)
    )

    valid_weights = np.where(
        valid,
        weights,
        0.0,
    )

    numerator = np.nansum(
        values
        * valid_weights,
        axis=1,
    )

    denominator = np.sum(
        valid_weights,
        axis=1,
    )

    result = np.full(
        values.shape[0],
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


def hourly_weighted_standard_deviation(
    values: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    weighted_mean = hourly_weighted_average(
        values,
        weights,
    )

    valid = (
        np.isfinite(values)
        & np.isfinite(weights)
    )

    valid_weights = np.where(
        valid,
        weights,
        0.0,
    )

    squared_deviation = (
        values
        - weighted_mean[:, None]
    ) ** 2

    numerator = np.nansum(
        squared_deviation
        * valid_weights,
        axis=1,
    )

    denominator = np.sum(
        valid_weights,
        axis=1,
    )

    result = np.full(
        values.shape[0],
        np.nan,
        dtype=float,
    )

    np.divide(
        numerator,
        denominator,
        out=result,
        where=denominator > 0,
    )

    return np.sqrt(
        result
    )


def build_monthly_base_weather(
    timestamps: pd.DatetimeIndex,
    site_dataset: xr.Dataset,
    mapping: pd.DataFrame,
    monthly_load: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build hourly load-weighted weather features.

    Direct ERA5 variables and nonlinear derived variables are calculated
    regionally first and then weighted by actual hourly regional load shares.
    """

    output = monthly_load.copy()

    expected_timestamps = pd.DatetimeIndex(
        timestamps
    )

    observed_timestamps = pd.DatetimeIndex(
        output[
            "timestamp_utc"
        ]
    )

    if not expected_timestamps.equals(
        observed_timestamps
    ):
        raise ValueError(
            "Monthly AESO load timestamps do not exactly match "
            "the ERA5 timestamps."
        )

    share_columns = (
        mapping[
            "share_column"
        ]
        .tolist()
    )

    hourly_weights = (
        output[
            share_columns
        ]
        .to_numpy(
            dtype=float
        )
    )

    regional_arrays: dict[
        str,
        np.ndarray,
    ] = {}

    for variable in SOURCE_VARIABLES:
        site_values = weather_array(
            site_dataset,
            variable,
        )

        if site_values is None:
            continue

        region_values = (
            region_values_from_sites(
                site_values,
                mapping,
            )
        )

        regional_arrays[
            variable
        ] = region_values

        output[
            f"load_weighted_{variable}"
        ] = hourly_weighted_average(
            region_values,
            hourly_weights,
        )

        if variable == "temperature_2m":
            output[
                "load_temperature_spatial_std_k"
            ] = (
                hourly_weighted_standard_deviation(
                    region_values,
                    hourly_weights,
                )
            )

            output[
                "load_temperature_min_k"
            ] = np.nanmin(
                region_values,
                axis=1,
            )

            output[
                "load_temperature_max_k"
            ] = np.nanmax(
                region_values,
                axis=1,
            )

    # ---------------------------------------------------------------------
    # Regional temperature and dew point
    # ---------------------------------------------------------------------

    if {
        "temperature_2m",
        "dewpoint_2m",
    }.issubset(
        regional_arrays
    ):
        regional_temperature_c = (
            regional_arrays[
                "temperature_2m"
            ]
            - 273.15
        )

        regional_dewpoint_c = (
            regional_arrays[
                "dewpoint_2m"
            ]
            - 273.15
        )

        regional_relative_humidity = (
            relative_humidity_from_temperature_dewpoint(
                temperature_c=regional_temperature_c,
                dewpoint_c=regional_dewpoint_c,
            )
        )

        regional_wet_bulb_c = (
            wet_bulb_temperature_c(
                temperature_c=regional_temperature_c,
                relative_humidity_pct=regional_relative_humidity,
            )
        )

        regional_heat_index_c = (
            heat_index_c(
                temperature_c=regional_temperature_c,
                relative_humidity_pct=regional_relative_humidity,
            )
        )

        output[
            "load_weighted_relative_humidity_pct"
        ] = hourly_weighted_average(
            regional_relative_humidity,
            hourly_weights,
        )

        output[
            "load_weighted_wetbulb_c"
        ] = hourly_weighted_average(
            regional_wet_bulb_c,
            hourly_weights,
        )

        output[
            "load_weighted_heat_index_c"
        ] = hourly_weighted_average(
            regional_heat_index_c,
            hourly_weights,
        )

        output[
            "load_relative_humidity_spatial_std_pct"
        ] = (
            hourly_weighted_standard_deviation(
                regional_relative_humidity,
                hourly_weights,
            )
        )

        output[
            "load_wetbulb_spatial_std_c"
        ] = (
            hourly_weighted_standard_deviation(
                regional_wet_bulb_c,
                hourly_weights,
            )
        )

    # ---------------------------------------------------------------------
    # Regional wind speed and wind chill
    # ---------------------------------------------------------------------

    if {
        "u_wind_10m",
        "v_wind_10m",
    }.issubset(
        regional_arrays
    ):
        regional_wind_speed_ms = np.sqrt(
            regional_arrays[
                "u_wind_10m"
            ] ** 2
            + regional_arrays[
                "v_wind_10m"
            ] ** 2
        )

        output[
            "load_weighted_wind_speed_10m"
        ] = hourly_weighted_average(
            regional_wind_speed_ms,
            hourly_weights,
        )

        output[
            "load_wind_speed_spatial_std_10m"
        ] = (
            hourly_weighted_standard_deviation(
                regional_wind_speed_ms,
                hourly_weights,
            )
        )

        if "temperature_2m" in regional_arrays:
            regional_temperature_c = (
                regional_arrays[
                    "temperature_2m"
                ]
                - 273.15
            )

            regional_wind_chill_c = (
                wind_chill_temperature_c(
                    temperature_c=regional_temperature_c,
                    wind_speed_ms=regional_wind_speed_ms,
                )
            )

            output[
                "load_weighted_wind_chill_c"
            ] = hourly_weighted_average(
                regional_wind_chill_c,
                hourly_weights,
            )

    return output


# ============================================================================
# Reusable feature functions
# ============================================================================

def add_temperature_unit_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    frame = frame.copy()

    frame[
        "load_weighted_temperature_c"
    ] = (
        frame[
            "load_weighted_temperature_2m"
        ]
        - 273.15
    )

    if (
        "load_weighted_dewpoint_2m"
        in frame.columns
    ):
        frame[
            "load_weighted_dewpoint_c"
        ] = (
            frame[
                "load_weighted_dewpoint_2m"
            ]
            - 273.15
        )

    if (
        "load_temperature_spatial_std_k"
        in frame.columns
    ):
        frame[
            "load_temperature_spatial_std_c"
        ] = frame[
            "load_temperature_spatial_std_k"
        ]

    if (
        "load_temperature_min_k"
        in frame.columns
    ):
        frame[
            "load_temperature_min_c"
        ] = (
            frame[
                "load_temperature_min_k"
            ]
            - 273.15
        )

    if (
        "load_temperature_max_k"
        in frame.columns
    ):
        frame[
            "load_temperature_max_c"
        ] = (
            frame[
                "load_temperature_max_k"
            ]
            - 273.15
        )

    return frame


def add_degree_hour_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    frame = frame.copy()

    temperature = frame[
        "load_weighted_temperature_c"
    ]

    for base in [
        15.0,
        18.0,
        20.0,
    ]:
        suffix = str(
            base
        ).replace(
            ".",
            "_",
        )

        frame[
            f"heating_degree_hours_{suffix}c"
        ] = (
            base
            - temperature
        ).clip(
            lower=0.0
        )

        frame[
            f"cooling_degree_hours_{suffix}c"
        ] = (
            temperature
            - base
        ).clip(
            lower=0.0
        )

    return frame


def add_temperature_nonlinearity_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    frame = frame.copy()

    temperature = frame[
        "load_weighted_temperature_c"
    ]

    frame[
        "load_temperature_c_squared"
    ] = temperature ** 2

    frame[
        "load_temperature_c_cubed"
    ] = temperature ** 3

    return frame


def add_temperature_change_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    frame = frame.copy()

    temperature = frame[
        "load_weighted_temperature_c"
    ]

    for lag in [
        1,
        3,
        6,
        12,
        24,
    ]:
        frame[
            f"load_temperature_change_{lag}h_c"
        ] = temperature.diff(
            lag
        )

    return frame


def add_temperature_rolling_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    frame = frame.copy()

    temperature = frame[
        "load_weighted_temperature_c"
    ]

    for window in [
        3,
        6,
        12,
        24,
        72,
    ]:
        frame[
            f"load_temperature_mean_{window}h_c"
        ] = (
            temperature
            .rolling(
                window,
                min_periods=1,
            )
            .mean()
        )

    frame[
        "load_temperature_min_24h_c"
    ] = (
        temperature
        .rolling(
            24,
            min_periods=1,
        )
        .min()
    )

    frame[
        "load_temperature_max_24h_c"
    ] = (
        temperature
        .rolling(
            24,
            min_periods=1,
        )
        .max()
    )

    return frame


def add_extreme_temperature_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    frame = frame.copy()

    temperature = frame[
        "load_weighted_temperature_c"
    ]

    thresholds = {
        "extreme_cold_below_minus_30c": (
            temperature < -30.0
        ),
        "extreme_cold_below_minus_25c": (
            temperature < -25.0
        ),
        "extreme_cold_below_minus_20c": (
            temperature < -20.0
        ),
        "extreme_heat_above_25c": (
            temperature > 25.0
        ),
        "extreme_heat_above_30c": (
            temperature > 30.0
        ),
    }

    for (
        column,
        condition,
    ) in thresholds.items():
        frame[column] = (
            condition.astype(
                "int8"
            )
        )

    return frame


def add_wind_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add provincial weighted-vector wind diagnostics.

    The canonical load_weighted_wind_speed_10m feature is calculated
    regionally before weighting in build_monthly_base_weather().
    """

    frame = frame.copy()

    required = {
        "load_weighted_u_wind_10m",
        "load_weighted_v_wind_10m",
    }

    if required.issubset(
        frame.columns
    ):
        frame[
            "load_weighted_wind_vector_speed_10m"
        ] = np.sqrt(
            frame[
                "load_weighted_u_wind_10m"
            ] ** 2
            + frame[
                "load_weighted_v_wind_10m"
            ] ** 2
        )

        frame[
            "load_weighted_wind_vector_direction_10m_degrees"
        ] = (
            np.degrees(
                np.arctan2(
                    -frame[
                        "load_weighted_u_wind_10m"
                    ],
                    -frame[
                        "load_weighted_v_wind_10m"
                    ],
                )
            )
            + 360.0
        ) % 360.0

    return frame


def add_solar_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    frame = frame.copy()

    radiation = (
        "load_weighted_"
        "surface_solar_radiation_downwards"
    )

    if radiation in frame.columns:
        frame[
            "load_weighted_solar_radiation_wm2"
        ] = (
            frame[radiation]
            / 3600.0
        )

        frame[
            "load_daylight_indicator"
        ] = (
            frame[
                "load_weighted_solar_radiation_wm2"
            ]
            > 0
        ).astype(
            "int8"
        )

    return frame


def add_local_time_reference_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    frame = frame.copy()

    local = (
        frame[
            "timestamp_utc"
        ]
        .dt.tz_convert(
            "America/Edmonton"
        )
    )

    frame[
        "hour_alberta"
    ] = local.dt.hour.astype(
        "int8"
    )

    frame[
        "month_alberta"
    ] = local.dt.month.astype(
        "int8"
    )

    return frame


FEATURE_BUILDERS: list[
    Callable[
        [
            pd.DataFrame
        ],
        pd.DataFrame,
    ]
] = [
    add_temperature_unit_features,
    add_degree_hour_features,
    add_temperature_nonlinearity_features,
    add_temperature_change_features,
    add_temperature_rolling_features,
    add_extreme_temperature_features,
    add_wind_features,
    add_solar_features,
    add_local_time_reference_features,
]


def apply_feature_builders(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    output = frame.copy()

    for builder in FEATURE_BUILDERS:
        output = builder(
            output
        )

    return output


# ============================================================================
# Monthly processing
# ============================================================================

def process_month(
    path: Path,
    mapping: pd.DataFrame,
    hourly_load: pd.DataFrame,
) -> tuple[
    pd.DataFrame | None,
    dict,
    pd.DataFrame,
]:
    period = monthly_file_period(
        path
    )

    started = time.perf_counter()
    audit_rows: list[dict] = []

    with xr.open_dataset(
        path
    ) as ds:
        timestamps = pd.DatetimeIndex(
            pd.to_datetime(
                ds[
                    "timestamp"
                ].values,
                utc=True,
            )
        )

        add_check(
            audit_rows,
            period,
            "expected_era5_hour_count",
            len(timestamps)
            == expected_month_hours(
                period
            ),
            len(timestamps),
            expected_month_hours(
                period
            ),
        )

        add_check(
            audit_rows,
            period,
            "era5_timestamps_unique",
            timestamps.is_unique,
            int(
                timestamps
                .duplicated()
                .sum()
            ),
            0,
        )

        add_check(
            audit_rows,
            period,
            "era5_timestamps_monotonic",
            timestamps.is_monotonic_increasing,
            timestamps.is_monotonic_increasing,
            True,
        )

        monthly_load = hourly_load.loc[
            hourly_load[
                "timestamp_utc"
            ].isin(
                timestamps
            )
        ].copy()

        monthly_load = (
            monthly_load
            .sort_values(
                "timestamp_utc"
            )
            .reset_index(
                drop=True
            )
        )

        # Months outside AESO area-load coverage are skipped cleanly.
        # Months without complete AESO load coverage are skipped cleanly.
        #
        # This handles:
        # - months entirely outside load coverage;
        # - edge months with only partial load coverage, such as January 2025.
        if len(monthly_load) != len(timestamps):
            coverage_status = (
                "skipped_no_load_coverage"
                if monthly_load.empty
                else "skipped_partial_load_coverage"
            )

            add_check(
                audit_rows,
                period,
                "aeso_complete_month_available",
                True,
                observed=len(monthly_load),
                expected=(
                    f"{len(timestamps)} rows required; "
                    f"month skipped because complete load coverage is unavailable"
                ),
                severity="info",
            )

            summary = {
                "period": period,
                "status": coverage_status,
                "pass": True,
                "rows": 0,
                "columns": 0,
                "available_load_hours": len(monthly_load),
                "expected_month_hours": len(timestamps),
                "source_file": str(path),
                "processing_seconds": round(
                    time.perf_counter() - started,
                    3,
                ),
            }

            return (
                None,
                summary,
                pd.DataFrame(audit_rows),
            )

        add_check(
            audit_rows,
            period,
            "aeso_load_hour_count_matches_era5",
            len(monthly_load)
            == len(timestamps),
            len(monthly_load),
            len(timestamps),
        )

        load_timestamps = pd.DatetimeIndex(
            monthly_load[
                "timestamp_utc"
            ]
        )

        add_check(
            audit_rows,
            period,
            "aeso_load_timestamps_match_era5",
            load_timestamps.equals(
                timestamps
            ),
            observed=(
                f"load_start={load_timestamps.min()}, "
                f"load_end={load_timestamps.max()}"
            ),
            expected=(
                f"era5_start={timestamps.min()}, "
                f"era5_end={timestamps.max()}"
            ),
        )

        share_columns = (
            mapping[
                "share_column"
            ]
            .tolist()
        )

        share_sum = monthly_load[
            share_columns
        ].sum(axis=1)

        maximum_share_error = float(
            np.abs(
                share_sum - 1.0
            ).max()
        )

        add_check(
            audit_rows,
            period,
            "regional_load_shares_sum_to_one",
            np.allclose(
                share_sum,
                1.0,
                atol=1e-6,
            ),
            observed=(
                f"max_abs_error="
                f"{maximum_share_error:.6g}"
            ),
            expected="0",
        )

        add_check(
            audit_rows,
            period,
            "regional_load_values_complete",
            not monthly_load[
                [
                    *REGION_LOAD_COLUMNS.values(),
                    "total_region_load_mw",
                    *share_columns,
                ]
            ].isna().any().any(),
            observed=int(
                monthly_load[
                    [
                        *REGION_LOAD_COLUMNS.values(),
                        "total_region_load_mw",
                        *share_columns,
                    ]
                ]
                .isna()
                .sum()
                .sum()
            ),
            expected=0,
        )

        site_dataset = extract_load_region_sites(
            ds,
            mapping,
        ).load()

    base = build_monthly_base_weather(
        timestamps=timestamps,
        site_dataset=site_dataset,
        mapping=mapping,
        monthly_load=monthly_load,
    )

    add_check(
        audit_rows,
        period,
        "output_row_count",
        len(base)
        == len(timestamps),
        len(base),
        len(timestamps),
    )

    add_check(
        audit_rows,
        period,
        "output_timestamps_unique",
        not base[
            "timestamp_utc"
        ].duplicated().any(),
        int(
            base[
                "timestamp_utc"
            ]
            .duplicated()
            .sum()
        ),
        0,
    )

    audit_df = pd.DataFrame(
        audit_rows
    )

    error_checks = audit_df.loc[
        audit_df[
            "severity"
        ].eq(
            "error"
        ),
        "pass",
    ]

    summary = {
        "period": period,
        "status": "processed",
        "pass": (
            bool(
                error_checks.all()
            )
            if not error_checks.empty
            else True
        ),
        "rows": len(base),
        "columns": len(base.columns),
        "start": str(
            base[
                "timestamp_utc"
            ].min()
        ),
        "end": str(
            base[
                "timestamp_utc"
            ].max()
        ),
        "imputed_load_hours": int(
            base[
                "area_load_imputed"
            ].sum()
        ),
        "processing_seconds": round(
            time.perf_counter()
            - started,
            3,
        ),
        "source_file": str(path),
    }

    return (
        base,
        summary,
        audit_df,
    )


# ============================================================================
# Full pipeline
# ============================================================================

def build_load_weather_features() -> dict:
    started = time.perf_counter()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    monthly_files = get_monthly_files(
        ERA5_MONTHLY_DIR
    )

    hourly_load = load_hourly_area_load(
        AREA_LOAD_FILE
    )

    regions = load_load_regions(
        LOAD_REGIONS_FILE
    )

    (
        latitudes,
        longitudes,
    ) = load_grid(
        monthly_files[0]
    )

    mapping = map_load_regions_to_grid(
        regions,
        latitudes,
        longitudes,
    )

    mapping.to_csv(
        LOAD_REGION_MAPPING_FILE,
        index=False,
    )

    print(
        f"Load regions: {len(mapping):,}"
    )

    print(
        "AESO load coverage: "
        f"{hourly_load['timestamp_utc'].min()} "
        f"to {hourly_load['timestamp_utc'].max()}"
    )

    print(
        "Maximum region-to-grid distance: "
        f"{mapping['weather_distance_km'].max():.2f} km"
    )

    print(
        "Location source: "
        f"{mapping['location_source'].iloc[0]}"
    )

    outputs = []
    summaries = []
    audits = []

    for path in monthly_files:
        period = monthly_file_period(
            path
        )

        print(
            f"Building load weather base for {period}"
        )

        try:
            (
                monthly,
                summary,
                audit,
            ) = process_month(
                path,
                mapping,
                hourly_load,
            )

            if monthly is not None:
                outputs.append(
                    monthly
                )

            summaries.append(
                summary
            )

            audits.append(
                audit
            )

        except Exception as exc:
            summaries.append(
                {
                    "period": period,
                    "status": "error",
                    "pass": False,
                    "error": repr(exc),
                    "source_file": str(
                        path
                    ),
                }
            )

            audits.append(
                pd.DataFrame(
                    [
                        {
                            "period": period,
                            "check": "process_month",
                            "pass": False,
                            "severity": "error",
                            "observed": repr(
                                exc
                            ),
                            "expected": (
                                "successful monthly "
                                "load-weather construction"
                            ),
                            "notes": "",
                        }
                    ]
                )
            )

    monthly_summary_df = pd.DataFrame(
        summaries
    )

    monthly_summary_df.to_csv(
        MONTHLY_SUMMARY_FILE,
        index=False,
    )

    audit_df = (
        pd.concat(
            audits,
            ignore_index=True,
        )
        if audits
        else pd.DataFrame()
    )

    audit_df.to_csv(
        AUDIT_FILE,
        index=False,
    )

    if not outputs:
        raise RuntimeError(
            "No monthly load-weather outputs were created."
        )

    master = pd.concat(
        outputs,
        ignore_index=True,
    )

    master[
        "timestamp_utc"
    ] = pd.to_datetime(
        master[
            "timestamp_utc"
        ],
        utc=True,
    )

    master = (
        master
        .drop_duplicates(
            subset=[
                "timestamp_utc"
            ],
            keep="last",
        )
        .sort_values(
            "timestamp_utc"
        )
        .reset_index(
            drop=True
        )
    )

    expected = pd.date_range(
        master[
            "timestamp_utc"
        ].min(),
        master[
            "timestamp_utc"
        ].max(),
        freq="h",
        tz="UTC",
    )

    missing = expected.difference(
        pd.DatetimeIndex(
            master[
                "timestamp_utc"
            ]
        )
    )

    if len(missing):
        raise ValueError(
            "Final load-weather table is missing "
            f"{len(missing):,} hourly timestamps."
        )

    if master[
        "timestamp_utc"
    ].duplicated().any():
        raise ValueError(
            "Final load-weather table contains duplicate timestamps."
        )

    master = apply_feature_builders(
        master
    )

    master.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    error_checks = audit_df.loc[
        audit_df[
            "severity"
        ].eq(
            "error"
        ),
        "pass",
    ]

    overall_pass = (
        bool(
            error_checks.all()
        )
        if not error_checks.empty
        else True
    )

    return {
        "dataset": "load_weather_features",
        "status": "saved",
        "pass": overall_pass,
        "rows": len(master),
        "columns": len(master.columns),
        "start": str(
            master[
                "timestamp_utc"
            ].min()
        ),
        "end": str(
            master[
                "timestamp_utc"
            ].max()
        ),
        "load_regions": len(
            mapping
        ),
        "imputed_load_hours": int(
            master[
                "area_load_imputed"
            ].sum()
        ),
        "output_file": str(
            OUTPUT_FILE
        ),
        "mapping_file": str(
            LOAD_REGION_MAPPING_FILE
        ),
        "monthly_summary_file": str(
            MONTHLY_SUMMARY_FILE
        ),
        "audit_file": str(
            AUDIT_FILE
        ),
        "processing_seconds": round(
            time.perf_counter()
            - started,
            3,
        ),
    }


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build Alberta load-relevant weather features "
            "using actual hourly AESO regional load."
        )
    )

    parser.parse_args()

    result = build_load_weather_features()

    print("\n" + "=" * 80)
    print("LOAD WEATHER FEATURE RESULT")
    print("=" * 80)

    for (
        key,
        value,
    ) in result.items():
        print(
            f"{key}: {value}"
        )

    print("=" * 80)


if __name__ == "__main__":
    main()
