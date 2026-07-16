from pathlib import Path
from datetime import datetime
import calendar
import time
import zipfile

import cdsapi
import xarray as xr


PROJECT_ROOT = Path("/Users/brodiehasein/alberta_power_markets_project")

ERA5_DIR = PROJECT_ROOT / "data/raw/weather/era5"
SINGLE_DIR = ERA5_DIR / "single_levels"
PRESSURE_DIR = ERA5_DIR / "pressure_levels"

SINGLE_DIR.mkdir(parents=True, exist_ok=True)
PRESSURE_DIR.mkdir(parents=True, exist_ok=True)

ALBERTA_AREA = [60, -120.5, 48.5, -109]


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


def days_for_month(year: int, month: int) -> list[str]:
    return [f"{d:02d}" for d in range(1, calendar.monthrange(year, month)[1] + 1)]


def expected_hours(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1] * 24


def valid_nc(path: Path, year: int, month: int) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False

    try:
        with xr.open_dataset(path, engine="netcdf4") as ds:
            return ds.sizes.get("valid_time") == expected_hours(year, month)
    except Exception:
        return False


def valid_single_folder(folder: Path, year: int, month: int) -> bool:
    required_files = [
        "data_stream-oper_stepType-instant.nc",
        "data_stream-oper_stepType-accum.nc",
        "data_stream-oper_stepType-max.nc",
    ]

    if not folder.exists():
        return False

    for file in required_files:
        path = folder / file
        if not valid_nc(path, year, month):
            return False

    return True


def valid_single_zip_or_folder(year: int, month: int) -> bool:
    year_s = str(year)
    month_s = f"{month:02d}"

    zip_path = SINGLE_DIR / f"era5_single_levels_alberta_{year_s}_{month_s}.zip"
    folder = SINGLE_DIR / f"era5_single_levels_alberta_{year_s}_{month_s}"

    if valid_single_folder(folder, year, month):
        return True

    if not zip_path.exists() or zip_path.stat().st_size == 0:
        return False

    required = {
        "data_stream-oper_stepType-instant.nc",
        "data_stream-oper_stepType-accum.nc",
        "data_stream-oper_stepType-max.nc",
    }

    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            names = set(z.namelist())
            if not required.issubset(names):
                return False

            folder.mkdir(exist_ok=True)

            for name in required:
                out = folder / name
                if not out.exists() or out.stat().st_size == 0:
                    z.extract(name, folder)

        return valid_single_folder(folder, year, month)

    except Exception:
        return False


def download_to_part(client: cdsapi.Client, dataset: str, request: dict, output_file: Path) -> None:
    part_file = output_file.with_suffix(output_file.suffix + ".part")

    if part_file.exists():
        part_file.unlink()

    start = time.time()
    print(f"[{datetime.now():%H:%M:%S}] Requesting {output_file.name}")

    client.retrieve(dataset, request).download(str(part_file))

    part_file.rename(output_file)

    elapsed = (time.time() - start) / 60
    size_mb = output_file.stat().st_size / 1024**2

    print(f"[{datetime.now():%H:%M:%S}] Saved {output_file.name} | {size_mb:.1f} MB | {elapsed:.1f} min")


def download_single_levels(client: cdsapi.Client, year: int, month: int) -> None:
    year_s = str(year)
    month_s = f"{month:02d}"

    output_file = SINGLE_DIR / f"era5_single_levels_alberta_{year_s}_{month_s}.zip"

    if valid_single_zip_or_folder(year, month):
        print(f"Skipping existing single levels: {year_s}-{month_s}")
        return

    if output_file.exists():
        print("Removing invalid/incomplete single zip:", output_file.name)
        output_file.unlink()

    request = {
        "product_type": ["reanalysis"],
        "variable": SINGLE_LEVEL_VARIABLES,
        "year": [year_s],
        "month": [month_s],
        "day": days_for_month(year, month),
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": ALBERTA_AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    download_to_part(client, "reanalysis-era5-single-levels", request, output_file)

    if not valid_single_zip_or_folder(year, month):
        raise RuntimeError(f"Downloaded single-level file failed validation: {output_file}")


def download_pressure_file(client: cdsapi.Client, year: int, month: int, request_info: dict) -> None:
    year_s = str(year)
    month_s = f"{month:02d}"

    output_file = PRESSURE_DIR / f"era5_pressure_{request_info['name']}_alberta_{year_s}_{month_s}.nc"

    if valid_nc(output_file, year, month):
        print("Skipping existing pressure file:", output_file.name)
        return

    if output_file.exists():
        print("Removing invalid/incomplete pressure file:", output_file.name)
        output_file.unlink()

    request = {
        "product_type": ["reanalysis"],
        "year": [year_s],
        "month": [month_s],
        "day": days_for_month(year, month),
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": ALBERTA_AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
        "pressure_level": request_info["pressure_level"],
        "variable": request_info["variable"],
    }

    download_to_part(client, "reanalysis-era5-pressure-levels", request, output_file)

    if not valid_nc(output_file, year, month):
        raise RuntimeError(f"Downloaded pressure file failed validation: {output_file}")


def download_pressure_levels(client: cdsapi.Client, year: int, month: int) -> None:
    for request_info in PRESSURE_REQUESTS:
        download_pressure_file(client, year, month, request_info)
        time.sleep(5)


def download_month(client: cdsapi.Client, year: int, month: int) -> None:
    print(f"\n{'=' * 70}")
    print(f"Processing {year}-{month:02d}")
    print(f"{'=' * 70}")

    try:
        download_single_levels(client, year, month)
    except Exception as e:
        print(f"FAILED single levels {year}-{month:02d}: {e}")

    time.sleep(5)

    try:
        download_pressure_levels(client, year, month)
    except Exception as e:
        print(f"FAILED pressure levels {year}-{month:02d}: {e}")

    time.sleep(10)


def download_range(start_year: int = 2015, end_year: int = 2026, end_month_by_year: dict | None = None) -> None:
    if end_month_by_year is None:
        end_month_by_year = {2026: 6}

    client = cdsapi.Client()

    for year in range(start_year, end_year + 1):
        last_month = end_month_by_year.get(year, 12)

        for month in range(1, last_month + 1):
            download_month(client, year, month)

    print("\nAll downloads attempted.")


if __name__ == "__main__":
    download_range(
        start_year=2015,
        end_year=2026,
        end_month_by_year={2026: 6},
    )