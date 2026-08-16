"""Download and validate monthly ERA5 files required by the pipeline.

Downloads use temporary ``.part`` files and are promoted only after the CDS
request completes. Every month is validated before it is skipped, and a batch
run fails with a nonzero exit status when any requested file remains invalid.
"""

from __future__ import annotations

import argparse
import calendar
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import cdsapi
import pandas as pd
import xarray as xr

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    ERA5_ALBERTA_AREA,
    ERA5_PRESSURE_LEVEL_DIR as PRESSURE_DIR,
    ERA5_SINGLE_LEVEL_DIR as SINGLE_DIR,
    PIPELINE_END_MONTH,
    PIPELINE_END_YEAR,
    PIPELINE_START_YEAR,
)


SINGLE_LEVEL_VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "mean_sea_level_pressure",
    "surface_pressure",
    "skin_temperature",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "instantaneous_10m_wind_gust",
    "10m_wind_gust_since_previous_post_processing",
    "surface_solar_radiation_downwards",
    "surface_solar_radiation_downward_clear_sky",
    "total_sky_direct_solar_radiation_at_surface",
    "total_cloud_cover",
    "high_cloud_cover",
    "medium_cloud_cover",
    "low_cloud_cover",
    "total_precipitation",
    "snow_depth",
    "snowfall",
    "total_column_water_vapour",
    "boundary_layer_height",
]

PRESSURE_REQUESTS = [
    {
        "name": "850_temp_wind_rh",
        "pressure_level": ["850"],
        "variable": [
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
            "relative_humidity",
        ],
    },
    {
        "name": "700_temp_wind_rh",
        "pressure_level": ["700"],
        "variable": [
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
            "relative_humidity",
        ],
    },
    {
        "name": "500_geopotential_wind",
        "pressure_level": ["500"],
        "variable": [
            "geopotential",
            "u_component_of_wind",
            "v_component_of_wind",
        ],
    },
]

SINGLE_LEVEL_FILENAMES = {
    "data_stream-oper_stepType-instant.nc",
    "data_stream-oper_stepType-accum.nc",
    "data_stream-oper_stepType-max.nc",
}


def days_for_month(year: int, month: int) -> list[str]:
    """Return every valid day number for a calendar month."""

    last_day = calendar.monthrange(year, month)[1]
    return [f"{day:02d}" for day in range(1, last_day + 1)]


def expected_timestamps(year: int, month: int) -> pd.DatetimeIndex:
    """Return the exact hourly UTC timeline expected for a month."""

    start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    end = start + pd.offsets.MonthBegin(1)
    return pd.date_range(start, end, freq="h", inclusive="left")


def valid_nc(path: Path, year: int, month: int) -> bool:
    """Return whether a NetCDF has the complete timeline for the named month."""

    if not path.exists() or path.stat().st_size == 0:
        return False

    try:
        with xr.open_dataset(path, engine="netcdf4") as dataset:
            time_name = next(
                (name for name in ("valid_time", "time") if name in dataset.coords),
                None,
            )
            if time_name is None:
                return False
            observed = pd.DatetimeIndex(
                pd.to_datetime(dataset[time_name].values, utc=True)
            )
            expected = expected_timestamps(year, month)
            return observed.equals(expected)
    except (OSError, ValueError, TypeError):
        return False


def valid_single_folder(folder: Path, year: int, month: int) -> bool:
    """Return whether all required single-level NetCDF files are valid."""

    return folder.exists() and all(
        valid_nc(folder / filename, year, month)
        for filename in SINGLE_LEVEL_FILENAMES
    )


def valid_single_zip_or_folder(year: int, month: int) -> bool:
    """Validate an extracted single-level month, extracting its ZIP if needed."""

    period = f"{year}_{month:02d}"
    zip_path = SINGLE_DIR / f"era5_single_levels_alberta_{period}.zip"
    folder = SINGLE_DIR / f"era5_single_levels_alberta_{period}"

    if valid_single_folder(folder, year, month):
        return True
    if not zip_path.exists() or zip_path.stat().st_size == 0:
        return False

    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            if not SINGLE_LEVEL_FILENAMES.issubset(archive.namelist()):
                return False
            folder.mkdir(parents=True, exist_ok=True)
            for filename in SINGLE_LEVEL_FILENAMES:
                archive.extract(filename, folder)
        return valid_single_folder(folder, year, month)
    except (OSError, zipfile.BadZipFile):
        return False


def download_to_part(
    client: cdsapi.Client,
    dataset: str,
    request: dict[str, Any],
    output_file: Path,
) -> None:
    """Download one request atomically through a temporary partial file."""

    output_file.parent.mkdir(parents=True, exist_ok=True)
    part_file = output_file.with_suffix(output_file.suffix + ".part")
    if part_file.exists():
        part_file.unlink()

    started = time.perf_counter()
    print(f"[{datetime.now():%H:%M:%S}] Requesting {output_file.name}")
    client.retrieve(dataset, request).download(str(part_file))
    part_file.replace(output_file)

    elapsed_minutes = (time.perf_counter() - started) / 60
    size_mb = output_file.stat().st_size / 1024**2
    print(
        f"[{datetime.now():%H:%M:%S}] Saved {output_file.name} | "
        f"{size_mb:.1f} MB | {elapsed_minutes:.1f} min"
    )


def download_single_levels(client: cdsapi.Client, year: int, month: int) -> None:
    """Download and validate one month of ERA5 single-level variables."""

    period = f"{year}_{month:02d}"
    output_file = SINGLE_DIR / f"era5_single_levels_alberta_{period}.zip"
    if valid_single_zip_or_folder(year, month):
        print(f"Skipping existing single levels: {year}-{month:02d}")
        return
    if output_file.exists():
        print(f"Removing invalid single-level archive: {output_file.name}")
        output_file.unlink()

    request = {
        "product_type": ["reanalysis"],
        "variable": SINGLE_LEVEL_VARIABLES,
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": days_for_month(year, month),
        "time": [f"{hour:02d}:00" for hour in range(24)],
        "area": ERA5_ALBERTA_AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    download_to_part(
        client,
        "reanalysis-era5-single-levels",
        request,
        output_file,
    )
    if not valid_single_zip_or_folder(year, month):
        raise RuntimeError(f"Single-level download failed validation: {output_file}")


def pressure_output_path(year: int, month: int, request_info: dict[str, Any]) -> Path:
    """Return the canonical path for one monthly pressure-level request."""

    return PRESSURE_DIR / (
        f"era5_pressure_{request_info['name']}_alberta_{year}_{month:02d}.nc"
    )


def download_pressure_file(
    client: cdsapi.Client,
    year: int,
    month: int,
    request_info: dict[str, Any],
) -> None:
    """Download and validate one monthly pressure-level file."""

    output_file = pressure_output_path(year, month, request_info)
    if valid_nc(output_file, year, month):
        print(f"Skipping existing pressure file: {output_file.name}")
        return
    if output_file.exists():
        print(f"Removing invalid pressure file: {output_file.name}")
        output_file.unlink()

    request = {
        "product_type": ["reanalysis"],
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": days_for_month(year, month),
        "time": [f"{hour:02d}:00" for hour in range(24)],
        "area": ERA5_ALBERTA_AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
        "pressure_level": request_info["pressure_level"],
        "variable": request_info["variable"],
    }
    download_to_part(
        client,
        "reanalysis-era5-pressure-levels",
        request,
        output_file,
    )
    if not valid_nc(output_file, year, month):
        raise RuntimeError(f"Pressure-level download failed validation: {output_file}")


def download_month(client: cdsapi.Client, year: int, month: int) -> list[str]:
    """Attempt every file for one month and return readable failure messages."""

    print(f"\n{'=' * 70}\nProcessing {year}-{month:02d}\n{'=' * 70}")
    failures: list[str] = []
    try:
        download_single_levels(client, year, month)
    except Exception as exc:
        failures.append(f"{year}-{month:02d} single levels: {exc}")

    for request_info in PRESSURE_REQUESTS:
        try:
            download_pressure_file(client, year, month, request_info)
        except Exception as exc:
            failures.append(
                f"{year}-{month:02d} pressure {request_info['name']}: {exc}"
            )

    for failure in failures:
        print(f"FAILED {failure}")
    return failures


def download_range(
    start_year: int = PIPELINE_START_YEAR,
    end_year: int = PIPELINE_END_YEAR,
    end_month: int = PIPELINE_END_MONTH,
) -> list[str]:
    """Download a year range and raise when any requested file fails."""

    SINGLE_DIR.mkdir(parents=True, exist_ok=True)
    PRESSURE_DIR.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()
    failures: list[str] = []

    for year in range(start_year, end_year + 1):
        last_month = end_month if year == end_year else 12
        for month in range(1, last_month + 1):
            failures.extend(download_month(client, year, month))

    if failures:
        details = "\n  - ".join(failures)
        raise RuntimeError(
            f"ERA5 download batch completed with {len(failures)} failures:\n  - "
            f"{details}"
        )
    print("\nAll requested ERA5 downloads validated successfully.")
    return failures


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for ERA5 acquisition."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=PIPELINE_START_YEAR)
    parser.add_argument("--end-year", type=int, default=PIPELINE_END_YEAR)
    parser.add_argument("--end-month", type=int, default=PIPELINE_END_MONTH)
    return parser


def main() -> None:
    """Run the configured ERA5 download batch from the command line."""

    args = build_argument_parser().parse_args()
    download_range(args.start_year, args.end_year, args.end_month)


if __name__ == "__main__":
    main()
