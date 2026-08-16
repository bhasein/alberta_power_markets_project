"""Standardize and audit monthly ERA5 weather grids for Alberta.

Raw single- and pressure-level NetCDF files are merged into one canonical
hourly monthly dataset. Full audits validate file presence, grid geometry,
timeline, variables, and meteorological ranges; quick audits retain only the
structural checks and write to separate evidence files.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import calendar
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import xarray as xr

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    PROJECT_ROOT,
    ERA5_RAW_DIR,
    ERA5_SINGLE_LEVEL_DIR as SINGLE_DIR,
    ERA5_PRESSURE_LEVEL_DIR as PRESSURE_DIR,
    ERA5_PREPROCESSING_DIR as PREPROCESSING_DIR,
    ERA5_MONTHLY_STANDARDIZED_DIR as MONTHLY_NC_DIR,
    PREPROCESSING_AUDITS_DIR as AUDIT_DIR,
    PIPELINE_START_YEAR as START_YEAR,
    PIPELINE_END_YEAR as END_YEAR,
    PIPELINE_END_MONTH as END_MONTH,
)
from preprocessing.shared import (
    add_check as append_audit_check,
    audit_passes,
    build_manifest,
    outputs_are_current,
    preprocessing_code_paths,
    write_audit_artifacts,
    write_manifests,
)

AUDIT_FILE = AUDIT_DIR / "era5_preprocessing_audit_checks.csv"
MONTHLY_SUMMARY_FILE = AUDIT_DIR / "era5_preprocessing_monthly_summary.csv"
FEATURE_SUMMARY_FILE = AUDIT_DIR / "era5_preprocessing_feature_summary.csv"
QUICK_AUDIT_FILE = AUDIT_DIR / "era5_preprocessing_quick_audit_checks.csv"
QUICK_MONTHLY_SUMMARY_FILE = (
    AUDIT_DIR / "era5_preprocessing_quick_monthly_summary.csv"
)
QUICK_FEATURE_SUMMARY_FILE = (
    AUDIT_DIR / "era5_preprocessing_quick_feature_summary.csv"
)

EXPECTED_LAT_COUNT = 47
EXPECTED_LON_COUNT = 47
EXPECTED_LAT_MIN = 48.5
EXPECTED_LAT_MAX = 60.0
EXPECTED_LON_MIN = -120.5
EXPECTED_LON_MAX = -109.0
EXPECTED_GRID_SIZE = EXPECTED_LAT_COUNT * EXPECTED_LON_COUNT

SINGLE_FILES = [
    "data_stream-oper_stepType-accum.nc",
    "data_stream-oper_stepType-instant.nc",
    "data_stream-oper_stepType-max.nc",
]

PRESSURE_FILES = {
    "850": "era5_pressure_850_temp_wind_rh_alberta_{year}_{month}.nc",
    "700": "era5_pressure_700_temp_wind_rh_alberta_{year}_{month}.nc",
    "500": "era5_pressure_500_geopotential_wind_alberta_{year}_{month}.nc",
}

SINGLE_RENAME_MAP = {
    "ssrd": "surface_solar_radiation_downwards",
    "ssrdc": "surface_solar_radiation_downwards_clear_sky",
    "fdir": "total_sky_direct_solar_radiation",
    "tp": "total_precipitation",
    "sf": "snowfall",
    "t2m": "temperature_2m",
    "d2m": "dewpoint_2m",
    "msl": "mean_sea_level_pressure",
    "sp": "surface_pressure",
    "skt": "skin_temperature",
    "u100": "u_wind_100m",
    "v100": "v_wind_100m",
    "u10": "u_wind_10m",
    "v10": "v_wind_10m",
    "i10fg": "instantaneous_10m_wind_gust",
    "tcc": "total_cloud_cover",
    "hcc": "high_cloud_cover",
    "mcc": "medium_cloud_cover",
    "lcc": "low_cloud_cover",
    "sd": "snow_depth",
    "tcwv": "total_column_water_vapour",
    "blh": "boundary_layer_height",
    "fg10": "max_10m_wind_gust",
}

EXPECTED_VARIABLES = [
    "surface_solar_radiation_downwards",
    "surface_solar_radiation_downwards_clear_sky",
    "total_sky_direct_solar_radiation",
    "total_precipitation",
    "snowfall",
    "temperature_2m",
    "dewpoint_2m",
    "mean_sea_level_pressure",
    "surface_pressure",
    "skin_temperature",
    "u_wind_100m",
    "v_wind_100m",
    "u_wind_10m",
    "v_wind_10m",
    "instantaneous_10m_wind_gust",
    "total_cloud_cover",
    "high_cloud_cover",
    "medium_cloud_cover",
    "low_cloud_cover",
    "snow_depth",
    "total_column_water_vapour",
    "boundary_layer_height",
    "max_10m_wind_gust",
    "temperature_850hpa",
    "u_wind_850hpa",
    "v_wind_850hpa",
    "relative_humidity_850hpa",
    "temperature_700hpa",
    "u_wind_700hpa",
    "v_wind_700hpa",
    "relative_humidity_700hpa",
    "geopotential_500hpa",
    "u_wind_500hpa",
    "v_wind_500hpa",
]

RANGE_EXPECTATIONS = {
    "temperature_2m": (180, 330),
    "dewpoint_2m": (180, 330),
    "skin_temperature": (180, 340),
    "temperature_850hpa": (180, 330),
    "temperature_700hpa": (180, 330),
    "mean_sea_level_pressure": (85000, 110000),
    "surface_pressure": (60000, 110000),
    "surface_solar_radiation_downwards": (0, 5_000_000),
    "surface_solar_radiation_downwards_clear_sky": (0, 5_000_000),
    "total_sky_direct_solar_radiation": (0, 5_000_000),
    "total_precipitation": (0, 1),
    "snowfall": (0, 1),
    "snow_depth": (0, 10),
    "total_cloud_cover": (0, 1),
    "high_cloud_cover": (0, 1),
    "medium_cloud_cover": (0, 1),
    "low_cloud_cover": (0, 1),
    "relative_humidity_850hpa": (-5, 150),
    "relative_humidity_700hpa": (-5, 150),
    "total_column_water_vapour": (0, 100),
    "boundary_layer_height": (0, 10000),
    "geopotential_500hpa": (30000, 70000),
}

for var in EXPECTED_VARIABLES:
    if "wind" in var or "gust" in var:
        RANGE_EXPECTATIONS[var] = (-150, 150)


def expected_hours(year: int, month: int) -> int:
    """Return the number of hourly records expected in a calendar month."""

    return calendar.monthrange(year, month)[1] * 24


def months_range(start_year: int, end_year: int, end_month: int):
    """Yield configured ``(year, month)`` pairs in chronological order."""

    for year in range(start_year, end_year + 1):
        last_month = end_month if year == end_year else 12
        for month in range(1, last_month + 1):
            yield year, month


def add_check(rows, period, check, passed, observed=None, expected=None, severity="error", notes=""):
    """Append one period-aware ERA5 audit result."""

    append_audit_check(
        rows,
        check,
        passed,
        observed,
        expected,
        severity,
        notes,
        period=period,
    )


def standardize_dims(ds: xr.Dataset) -> xr.Dataset:
    """Normalize ERA5 coordinate names and ordering."""

    rename = {}

    if "valid_time" in ds.dims or "valid_time" in ds.coords:
        rename["valid_time"] = "timestamp"
    if "latitude" in ds.dims or "latitude" in ds.coords:
        rename["latitude"] = "lat"
    if "longitude" in ds.dims or "longitude" in ds.coords:
        rename["longitude"] = "lon"

    ds = ds.rename(rename)

    for coord in ["number", "expver"]:
        if coord in ds.coords:
            ds = ds.drop_vars(coord, errors="ignore")

    if "pressure_level" in ds.dims:
        ds = ds.squeeze("pressure_level", drop=True)

    return ds


def open_nc(path: Path) -> xr.Dataset:
    """Open one required NetCDF source file."""

    return standardize_dims(xr.open_dataset(path))


def ensure_single_folder(year: int, month: int) -> Path:
    """Return the extracted single-level folder, extracting its ZIP if needed."""

    year_s = str(year)
    month_s = f"{month:02d}"

    folder = SINGLE_DIR / f"era5_single_levels_alberta_{year_s}_{month_s}"
    zip_file = SINGLE_DIR / f"era5_single_levels_alberta_{year_s}_{month_s}.zip"

    if folder.exists():
        return folder

    if zip_file.exists():
        folder.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_file, "r") as z:
            z.extractall(folder)
        return folder

    raise FileNotFoundError(f"Missing single-level folder or zip for {year_s}-{month_s}")


def pressure_path(year: int, month: int, level: str) -> Path:
    """Return the configured pressure-level file for one month and level."""

    return PRESSURE_DIR / PRESSURE_FILES[level].format(
        year=str(year),
        month=f"{month:02d}",
    )


def raw_files_for_month(year: int, month: int) -> list[Path]:
    """Return every raw file required to build one standardized month."""

    folder = ensure_single_folder(year, month)

    files = [folder / f for f in SINGLE_FILES]
    files += [
        pressure_path(year, month, "850"),
        pressure_path(year, month, "700"),
        pressure_path(year, month, "500"),
    ]

    return files


def audit_raw_files(year: int, month: int) -> pd.DataFrame:
    """Audit presence and readability of one month's raw source files."""

    period = f"{year}-{month:02d}"
    rows = []

    try:
        files = raw_files_for_month(year, month)
    except Exception as e:
        add_check(rows, period, "raw_file_collection", False, repr(e), "all raw files discoverable")
        return pd.DataFrame(rows)

    for path in files:
        add_check(
            rows,
            period,
            f"raw_file_exists__{path.name}",
            path.exists(),
            str(path),
            "file exists",
        )

        if path.exists():
            add_check(
                rows,
                period,
                f"raw_file_non_empty__{path.name}",
                path.stat().st_size > 0,
                f"{path.stat().st_size} bytes",
                "> 0 bytes",
            )

    return pd.DataFrame(rows)


def load_single_levels(year: int, month: int) -> xr.Dataset:
    """Load and merge the single-level ERA5 variables for one month."""

    folder = ensure_single_folder(year, month)

    missing = [f for f in SINGLE_FILES if not (folder / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing single-level files for {year}-{month:02d}: {missing}")

    datasets = []

    for filename in SINGLE_FILES:
        ds = open_nc(folder / filename)
        rename_vars = {k: v for k, v in SINGLE_RENAME_MAP.items() if k in ds.data_vars}
        ds = ds.rename(rename_vars)
        datasets.append(ds)

    return xr.merge(datasets, compat="override")


def load_pressure_level(year: int, month: int, level: str) -> xr.Dataset:
    """Load and standardize one pressure-level ERA5 dataset."""

    path = pressure_path(year, month, level)

    if not path.exists():
        raise FileNotFoundError(path)

    ds = open_nc(path)

    if level in {"850", "700"}:
        rename = {
            "t": f"temperature_{level}hpa",
            "u": f"u_wind_{level}hpa",
            "v": f"v_wind_{level}hpa",
            "r": f"relative_humidity_{level}hpa",
        }
    elif level == "500":
        rename = {
            "z": "geopotential_500hpa",
            "u": "u_wind_500hpa",
            "v": "v_wind_500hpa",
        }
    else:
        raise ValueError(level)

    rename = {k: v for k, v in rename.items() if k in ds.data_vars}
    return ds.rename(rename)


def build_month_dataset(year: int, month: int) -> xr.Dataset:
    """Merge all required ERA5 inputs into one standardized monthly dataset."""

    single = load_single_levels(year, month)
    p850 = load_pressure_level(year, month, "850")
    p700 = load_pressure_level(year, month, "700")
    p500 = load_pressure_level(year, month, "500")

    ds = xr.merge([single, p850, p700, p500], compat="override")

    keep = [v for v in EXPECTED_VARIABLES if v in ds.data_vars]
    ds = ds[keep]

    ds = ds.sortby("timestamp")
    ds = ds.sortby("lat", ascending=True)
    ds = ds.sortby("lon", ascending=True)

    return ds


def audit_grid(ds: xr.Dataset, year: int, month: int) -> pd.DataFrame:
    """Validate one month's latitude-longitude grid contract."""

    period = f"{year}-{month:02d}"
    rows = []

    dims = set(ds.dims)

    add_check(rows, period, "dims_present", {"timestamp", "lat", "lon"}.issubset(dims), sorted(dims), "timestamp, lat, lon")

    if "lat" in ds.coords and "lon" in ds.coords:
        lat = ds["lat"].values
        lon = ds["lon"].values

        add_check(rows, period, "lat_count", len(lat) == EXPECTED_LAT_COUNT, len(lat), EXPECTED_LAT_COUNT)
        add_check(rows, period, "lon_count", len(lon) == EXPECTED_LON_COUNT, len(lon), EXPECTED_LON_COUNT)

        add_check(rows, period, "lat_min", np.isclose(float(np.nanmin(lat)), EXPECTED_LAT_MIN), float(np.nanmin(lat)), EXPECTED_LAT_MIN, severity="warning")
        add_check(rows, period, "lat_max", np.isclose(float(np.nanmax(lat)), EXPECTED_LAT_MAX), float(np.nanmax(lat)), EXPECTED_LAT_MAX, severity="warning")
        add_check(rows, period, "lon_min", np.isclose(float(np.nanmin(lon)), EXPECTED_LON_MIN), float(np.nanmin(lon)), EXPECTED_LON_MIN, severity="warning")
        add_check(rows, period, "lon_max", np.isclose(float(np.nanmax(lon)), EXPECTED_LON_MAX), float(np.nanmax(lon)), EXPECTED_LON_MAX, severity="warning")

        lat_diffs = np.round(np.diff(np.sort(lat)), 6)
        lon_diffs = np.round(np.diff(np.sort(lon)), 6)

        add_check(rows, period, "lat_spacing_025", np.allclose(lat_diffs, 0.25), sorted(set(lat_diffs)), "0.25 degrees", severity="warning")
        add_check(rows, period, "lon_spacing_025", np.allclose(lon_diffs, 0.25), sorted(set(lon_diffs)), "0.25 degrees", severity="warning")

        add_check(rows, period, "grid_cell_count", len(lat) * len(lon) == EXPECTED_GRID_SIZE, len(lat) * len(lon), EXPECTED_GRID_SIZE)

    return pd.DataFrame(rows)


def audit_time(ds: xr.Dataset, year: int, month: int) -> pd.DataFrame:
    """Validate one month's complete, unique hourly UTC timeline."""

    period = f"{year}-{month:02d}"
    rows = []

    ts = pd.DatetimeIndex(pd.to_datetime(ds["timestamp"].values))
    expected_n = expected_hours(year, month)

    expected_start = pd.Timestamp(f"{year}-{month:02d}-01 00:00:00")
    expected_end = expected_start + pd.Timedelta(hours=expected_n - 1)
    expected_index = pd.date_range(expected_start, expected_end, freq="h")

    diffs = pd.Series(ts).diff().dropna()
    missing_hours = expected_index.difference(ts)
    extra_hours = ts.difference(expected_index)

    add_check(rows, period, "expected_hours", len(ts) == expected_n, len(ts), expected_n)
    add_check(rows, period, "timestamp_start", ts.min() == expected_start, str(ts.min()), str(expected_start))
    add_check(rows, period, "timestamp_end", ts.max() == expected_end, str(ts.max()), str(expected_end))
    add_check(rows, period, "timestamps_unique", ts.is_unique, int(ts.duplicated().sum()), 0)
    add_check(rows, period, "timestamps_monotonic", ts.is_monotonic_increasing, ts.is_monotonic_increasing, True)
    add_check(rows, period, "hourly_spacing", len(diffs[diffs != pd.Timedelta(hours=1)]) == 0, int(len(diffs[diffs != pd.Timedelta(hours=1)])), 0)
    add_check(rows, period, "missing_hours", len(missing_hours) == 0, len(missing_hours), 0)
    add_check(rows, period, "extra_hours", len(extra_hours) == 0, len(extra_hours), 0)

    return pd.DataFrame(rows)


def audit_variables(ds: xr.Dataset, year: int, month: int, quick: bool = False) -> tuple[pd.DataFrame, list[dict]]:
    """Validate variable schema and, in full mode, variable-level statistics."""

    period = f"{year}-{month:02d}"
    rows = []
    summaries = []

    observed_vars = set(ds.data_vars)
    missing_vars = sorted(set(EXPECTED_VARIABLES) - observed_vars)
    extra_vars = sorted(observed_vars - set(EXPECTED_VARIABLES))

    add_check(rows, period, "expected_variables_present", len(missing_vars) == 0, "; ".join(missing_vars), "no missing variables")
    add_check(rows, period, "no_unexpected_variables", len(extra_vars) == 0, "; ".join(extra_vars), "no extra variables", severity="warning")

    if quick:
        return pd.DataFrame(rows), summaries

    for var in EXPECTED_VARIABLES:
        if var not in ds.data_vars:
            continue

        arr = ds[var]
        total_cells = int(arr.size)
        missing_count = int(arr.isnull().sum().item())
        missing_pct = missing_count / total_cells * 100

        vmin = float(arr.min(skipna=True).item())
        vmax = float(arr.max(skipna=True).item())
        vmean = float(arr.mean(skipna=True).item())
        vstd = float(arr.std(skipna=True).item())

        try:
            p01 = float(arr.quantile(0.01, skipna=True).item())
            p50 = float(arr.quantile(0.50, skipna=True).item())
            p99 = float(arr.quantile(0.99, skipna=True).item())
        except Exception:
            p01 = np.nan
            p50 = np.nan
            p99 = np.nan

        low, high = RANGE_EXPECTATIONS.get(var, (-np.inf, np.inf))

        add_check(
            rows,
            period,
            f"range_check__{var}",
            (vmin >= low) and (vmax <= high),
            observed=f"min={vmin:.6g}, max={vmax:.6g}",
            expected=f"[{low}, {high}]",
            severity="warning",
        )

        add_check(
            rows,
            period,
            f"missing_check__{var}",
            missing_count == 0,
            observed=missing_count,
            expected=0,
            severity="warning",
        )

        if {"timestamp", "lat", "lon"}.issubset(set(arr.dims)):
            per_hour_non_null = arr.notnull().sum(dim=["lat", "lon"])
            incomplete_hours = int((per_hour_non_null != EXPECTED_GRID_SIZE).sum().item())

            add_check(
                rows,
                period,
                f"complete_grid_each_hour__{var}",
                incomplete_hours == 0,
                observed=incomplete_hours,
                expected=0,
                severity="warning",
                notes="Checks that every hour has full spatial grid coverage.",
            )

        summaries.append({
            "period": period,
            "variable": var,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "min": vmin,
            "p01": p01,
            "mean": vmean,
            "median": p50,
            "p99": p99,
            "max": vmax,
            "std": vstd,
            "units": arr.attrs.get("units", ""),
            "long_name": arr.attrs.get("long_name", ""),
            "dtype": str(arr.dtype),
        })

    return pd.DataFrame(rows), summaries


def audit_meteorology(ds: xr.Dataset, year: int, month: int) -> pd.DataFrame:
    """Validate physical relationships and plausible meteorological ranges."""

    period = f"{year}-{month:02d}"
    rows = []

    if {"dewpoint_2m", "temperature_2m"}.issubset(ds.data_vars):
        violations = int((ds["dewpoint_2m"] > ds["temperature_2m"] + 0.5).sum().item())
        add_check(rows, period, "met_dewpoint_not_above_temperature", violations == 0, violations, 0, severity="warning")

    for cloud_var in ["total_cloud_cover", "high_cloud_cover", "medium_cloud_cover", "low_cloud_cover"]:
        if cloud_var in ds.data_vars:
            violations = int(((ds[cloud_var] < -0.001) | (ds[cloud_var] > 1.001)).sum().item())
            add_check(rows, period, f"met_cloud_cover_0_to_1__{cloud_var}", violations == 0, violations, 0, severity="warning")

    for non_negative_var in [
        "surface_solar_radiation_downwards",
        "surface_solar_radiation_downwards_clear_sky",
        "total_sky_direct_solar_radiation",
        "total_precipitation",
        "snowfall",
        "snow_depth",
        "boundary_layer_height",
        "total_column_water_vapour",
        "max_10m_wind_gust",
        "instantaneous_10m_wind_gust",
    ]:
        if non_negative_var in ds.data_vars:
            violations = int((ds[non_negative_var] < -1e-8).sum().item())
            add_check(rows, period, f"met_non_negative__{non_negative_var}", violations == 0, violations, 0, severity="warning")

    for rh_var in ["relative_humidity_850hpa", "relative_humidity_700hpa"]:
        if rh_var in ds.data_vars:
            violations = int(((ds[rh_var] < -5) | (ds[rh_var] > 150)).sum().item())
            add_check(rows, period, f"met_relative_humidity_reasonable__{rh_var}", violations == 0, violations, 0, severity="warning")

    if {"surface_pressure", "mean_sea_level_pressure"}.issubset(ds.data_vars):
        violations = int((ds["surface_pressure"] > ds["mean_sea_level_pressure"] + 5000).sum().item())
        add_check(
            rows,
            period,
            "met_surface_pressure_not_far_above_mslp",
            violations == 0,
            violations,
            0,
            severity="warning",
            notes="Surface pressure can differ by elevation, but should not be implausibly above MSLP.",
        )

    return pd.DataFrame(rows)

def audit_month(
    ds: xr.Dataset,
    year: int,
    month: int,
    quick: bool = False,
):
    """Combine all enabled audits for one standardized ERA5 month."""

    raw_audit = audit_raw_files(year, month)
    grid_audit = audit_grid(ds, year, month)
    time_audit = audit_time(ds, year, month)

    var_audit, var_summaries = audit_variables(
        ds,
        year,
        month,
        quick=quick,
    )

    met_audit = (
        pd.DataFrame()
        if quick
        else audit_meteorology(ds, year, month)
    )

    audit_frames = [
        raw_audit,
        grid_audit,
        time_audit,
        var_audit,
        met_audit,
    ]

    audit_df = pd.concat(
        [
            frame
            for frame in audit_frames
            if not frame.empty
        ],
        ignore_index=True,
    )

    return audit_df, var_summaries

def process_month(
    year: int,
    month: int,
    overwrite: bool = False,
    quick: bool = False,
):
    """Build or re-audit one standardized ERA5 month."""

    period = f"{year}-{month:02d}"

    MONTHLY_NC_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    nc_out = (
        MONTHLY_NC_DIR
        / f"era5_alberta_standardized_{year}_{month:02d}.nc"
    )

    start = time.perf_counter()

    expected_manifest = build_manifest(
        dataset=f"era5_{period}",
        source_paths=raw_files_for_month(year, month),
        code_paths=preprocessing_code_paths(Path(__file__)),
        configuration={"year": year, "month": month},
    )

    if not overwrite and outputs_are_current([nc_out], expected_manifest):
        ds = xr.open_dataset(nc_out)

        audit_df, var_summaries = audit_month(
            ds=ds,
            year=year,
            month=month,
            quick=quick,
        )

        audit_pass = audit_passes(audit_df)

        result = {
            "period": period,
            "status": "audited_existing",
            "pass": bool(audit_pass),
            "hours": int(ds.sizes["timestamp"]),
            "lat_count": int(ds.sizes["lat"]),
            "lon_count": int(ds.sizes["lon"]),
            "grid_cells": int(
                ds.sizes["lat"] * ds.sizes["lon"]
            ),
            "features": len(ds.data_vars),
            "start": str(
                pd.Timestamp(ds["timestamp"].min().item())
            ),
            "end": str(
                pd.Timestamp(ds["timestamp"].max().item())
            ),
            "nc_file": str(nc_out),
            "processing_seconds": round(
                time.perf_counter() - start,
                3,
            ),
        }

        ds.close()

        return result, audit_df, var_summaries

    ds = build_month_dataset(year, month)

    audit_df, var_summaries = audit_month(
        ds=ds,
        year=year,
        month=month,
        quick=quick,
    )

    audit_pass = audit_passes(audit_df)

    if audit_pass:
        ds.to_netcdf(nc_out)
        write_manifests([nc_out], expected_manifest)

    result = {
        "period": period,
        "status": "saved" if audit_pass else "audit_failed",
        "pass": bool(audit_pass),
        "hours": int(ds.sizes["timestamp"]),
        "lat_count": int(ds.sizes["lat"]),
        "lon_count": int(ds.sizes["lon"]),
        "grid_cells": int(
            ds.sizes["lat"] * ds.sizes["lon"]
        ),
        "features": len(ds.data_vars),
        "start": str(
            pd.Timestamp(ds["timestamp"].min().item())
        ),
        "end": str(
            pd.Timestamp(ds["timestamp"].max().item())
        ),
        "nc_file": str(nc_out),
        "processing_seconds": round(
            time.perf_counter() - start,
            3,
        ),
    }

    ds.close()

    return result, audit_df, var_summaries

def print_report(
    monthly_summary: pd.DataFrame,
    audit_df: pd.DataFrame,
):
    """Print a concise human-readable ERA5 audit report."""

    audit_pass = audit_passes(audit_df)

    failed = audit_df.loc[~audit_df["pass"]]

    total_hours = (
        int(monthly_summary["hours"].sum())
        if "hours" in monthly_summary.columns
        else 0
    )

    print("\n" + "=" * 80)
    print("ERA5 SPATIAL PREPROCESSING AUDIT")
    print("=" * 80)

    print(f"Overall pass   : {audit_pass}")
    print(f"Months         : {len(monthly_summary):,}")
    print(
        "Complete       : "
        f"{int(monthly_summary['pass'].sum()):,}"
        f"/{len(monthly_summary):,}"
    )
    print(f"Monthly hours  : {total_hours:,}")
    print(f"Latitude count : {EXPECTED_LAT_COUNT}")
    print(f"Longitude count: {EXPECTED_LON_COUNT}")
    print(f"Grid cells     : {EXPECTED_GRID_SIZE:,}")
    print(f"Variables      : {len(EXPECTED_VARIABLES)}")

    print("\nFailed checks:")

    if failed.empty:
        print("  None")
    else:
        for _, row in failed.head(80).iterrows():
            print(
                f"  - {row['period']} | "
                f"{row['check']} [{row['severity']}] "
                f"observed={row['observed']} "
                f"expected={row['expected']}"
            )

        if len(failed) > 80:
            print(
                f"  ... {len(failed) - 80} more failed "
                "checks in audit CSV"
            )

    print("=" * 80)

def process_era5(
    overwrite: bool = False,
    quick: bool = False,
) -> dict:
    """Process and audit the configured range of monthly ERA5 files."""

    start = time.perf_counter()

    audit_file = QUICK_AUDIT_FILE if quick else AUDIT_FILE
    monthly_summary_file = (
        QUICK_MONTHLY_SUMMARY_FILE if quick else MONTHLY_SUMMARY_FILE
    )
    feature_summary_file = (
        QUICK_FEATURE_SUMMARY_FILE if quick else FEATURE_SUMMARY_FILE
    )

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PREPROCESSING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MONTHLY_NC_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []
    audits = []
    variable_summaries = []

    for year, month in months_range(
        START_YEAR,
        END_YEAR,
        END_MONTH,
    ):
        print(f"Processing ERA5 {year}-{month:02d}")

        try:
            result, audit_df, var_summaries = process_month(
            year=year,
            month=month,
            overwrite=overwrite,
            quick=quick,
        )

            results.append(result)

            if not audit_df.empty:
                audits.append(audit_df)

            variable_summaries.extend(var_summaries)

        except Exception as exc:
            period = f"{year}-{month:02d}"

            results.append({
                "period": period,
                "status": "error",
                "pass": False,
                "error": repr(exc),
            })

            audits.append(
                pd.DataFrame([
                    {
                        "period": period,
                        "check": "process_month",
                        "pass": False,
                        "severity": "error",
                        "observed": repr(exc),
                        "expected": (
                            "successful spatial month preprocessing"
                        ),
                        "notes": "",
                    }
                ])
            )

    monthly_summary = pd.DataFrame(results)

    audit_mode = "quick" if quick else "full"
    monthly_summary["audit_mode"] = audit_mode

    audit_df = (
        pd.concat(audits, ignore_index=True)
        if audits
        else pd.DataFrame()
    )

    if not audit_df.empty:
        audit_df["audit_mode"] = "quick" if quick else "full"
    feature_summary_df = pd.DataFrame(
        variable_summaries
    )

    all_raw_files = [
        path
        for year, month in months_range(START_YEAR, END_YEAR, END_MONTH)
        for path in raw_files_for_month(year, month)
    ]
    audit_manifest = build_manifest(
        dataset=f"era5_{audit_mode}_audit",
        source_paths=all_raw_files,
        code_paths=preprocessing_code_paths(Path(__file__)),
        configuration={
            "start_year": START_YEAR,
            "end_year": END_YEAR,
            "end_month": END_MONTH,
            "audit_mode": audit_mode,
        },
    )
    write_audit_artifacts(
        {
            monthly_summary_file: monthly_summary,
            audit_file: audit_df,
            feature_summary_file: feature_summary_df,
        },
        audit_manifest,
    )

    print_report(
        monthly_summary=monthly_summary,
        audit_df=audit_df,
    )

    overall_pass = audit_passes(audit_df)

    complete_months = (
        int(monthly_summary["pass"].sum())
        if "pass" in monthly_summary.columns
        else 0
    )

    return {
        "dataset": "era5_spatial",
        "status": "saved",
        "pass": overall_pass,
        "months": len(monthly_summary),
        "complete_months": complete_months,
        "output_directory": str(MONTHLY_NC_DIR),
        "audit_file": str(audit_file),
        "monthly_summary_file": str(
            monthly_summary_file
        ),
        "feature_summary_file": str(
            feature_summary_file
        ),
        "processing_seconds": round(
            time.perf_counter() - start,
            3,
        ),
    }


def main() -> None:
    """Run ERA5 preprocessing from the command line."""

    parser = argparse.ArgumentParser(
        description=(
            "Preprocess raw ERA5 files into standardized "
            "monthly hourly spatial NetCDF datasets."
        )
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild and overwrite existing monthly NetCDF files.",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip expensive variable and meteorological audits.",
    )

    args = parser.parse_args()

    result = process_era5(
        overwrite=args.overwrite,
        quick=args.quick,
    )

    print("\n" + "=" * 80)
    print("ERA5 SPATIAL PREPROCESSING RESULT")
    print("=" * 80)

    for key, value in result.items():
        print(f"{key}: {value}")

    print("=" * 80)


if __name__ == "__main__":
    main()
