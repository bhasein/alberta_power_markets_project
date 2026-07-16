from pathlib import Path
import argparse
import os
import time
from datetime import datetime
import pandas as pd


DEFAULT_PROJECT_ROOT = Path("/Users/brodiehasein/alberta_power_markets_project")

START_YEAR = 2015
END_YEAR = 2026
END_MONTH = 6

EXPECTED_SINGLE_NC_COUNT = 3


def clear_terminal() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def audit_era5_downloads(
    project_root: Path = DEFAULT_PROJECT_ROOT,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    end_month: int = END_MONTH,
    expected_single_nc_count: int = EXPECTED_SINGLE_NC_COUNT,
) -> pd.DataFrame:
    era5_dir = project_root / "data" / "raw" / "weather" / "era5"
    single_dir = era5_dir / "single_levels"
    pressure_dir = era5_dir / "pressure_levels"

    records = []

    for year in range(start_year, end_year + 1):
        last_month = end_month if year == end_year else 12

        for month in range(1, last_month + 1):
            year_s = str(year)
            month_s = f"{month:02d}"
            period = f"{year_s}-{month_s}"

            single_folder = single_dir / f"era5_single_levels_alberta_{year_s}_{month_s}"
            single_nc_files = sorted(single_folder.glob("*.nc")) if single_folder.exists() else []

            single_nc_count = len(single_nc_files)
            single_complete = single_nc_count == expected_single_nc_count

            p850 = pressure_dir / f"era5_pressure_850_temp_wind_rh_alberta_{year_s}_{month_s}.nc"
            p700 = pressure_dir / f"era5_pressure_700_temp_wind_rh_alberta_{year_s}_{month_s}.nc"
            p500 = pressure_dir / f"era5_pressure_500_geopotential_wind_alberta_{year_s}_{month_s}.nc"

            p850_exists = p850.exists()
            p700_exists = p700.exists()
            p500_exists = p500.exists()

            pressure_downloaded = sum([p850_exists, p700_exists, p500_exists])
            pressure_complete = pressure_downloaded == 3

            total_files_downloaded = single_nc_count + pressure_downloaded
            total_files_needed = expected_single_nc_count + 3

            records.append(
                {
                    "period": period,
                    "year": year,
                    "month": month,
                    "single_folder_exists": single_folder.exists(),
                    "single_nc_count": single_nc_count,
                    "single_nc_expected": expected_single_nc_count,
                    "single_progress": f"{single_nc_count}/{expected_single_nc_count}",
                    "single_complete": single_complete,
                    "single_nc_files": "; ".join([f.name for f in single_nc_files]),
                    "pressure_downloaded": pressure_downloaded,
                    "pressure_total_needed": 3,
                    "pressure_progress": f"{pressure_downloaded}/3",
                    "pressure_complete": pressure_complete,
                    "p850_exists": p850_exists,
                    "p700_exists": p700_exists,
                    "p500_exists": p500_exists,
                    "total_files_downloaded": total_files_downloaded,
                    "total_files_needed": total_files_needed,
                    "total_progress": f"{total_files_downloaded}/{total_files_needed}",
                    "month_complete": single_complete and pressure_complete,
                }
            )

    return pd.DataFrame(records)


def save_audit_csv(audit: pd.DataFrame, project_root: Path) -> Path:
    output_path = (
        project_root
        / "data"
        / "raw"
        / "weather"
        / "era5"
        / "era5_download_audit.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_path, index=False)
    return output_path


def print_summary(download_audit: pd.DataFrame) -> None:
    single_done = download_audit["single_nc_count"].sum()
    single_needed = download_audit["single_nc_expected"].sum()

    pressure_done = download_audit["pressure_downloaded"].sum()
    pressure_needed = download_audit["pressure_total_needed"].sum()

    total_done = download_audit["total_files_downloaded"].sum()
    total_needed = download_audit["total_files_needed"].sum()

    complete_months = download_audit["month_complete"].sum()
    total_months = len(download_audit)

    print("=" * 80)
    print("ERA5 RAW DOWNLOAD DISK AUDIT")
    print("=" * 80)
    print(f"Last refreshed:                {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Single-level .nc files present: {single_done}/{single_needed}")
    print(f"Pressure-level files present:   {pressure_done}/{pressure_needed}")
    print(f"Total usable raw files present: {total_done}/{total_needed}")
    print(f"Complete months:                {complete_months}/{total_months}")


def run_once(args: argparse.Namespace) -> None:
    audit = audit_era5_downloads(
        project_root=args.project_root,
        start_year=args.start_year,
        end_year=args.end_year,
        end_month=args.end_month,
        expected_single_nc_count=args.expected_single_nc_count,
    )

    print_summary(audit)

    if args.save_csv:
        output_path = save_audit_csv(audit, args.project_root)
        print(f"\nSaved audit CSV to: {output_path}")


def run_watch(args: argparse.Namespace) -> None:
    try:
        while True:
            clear_terminal()
            run_once(args)
            print(f"\nWatching for changes. Refreshing every {args.refresh_seconds} seconds.")
            print("Press Ctrl+C to stop.")
            time.sleep(args.refresh_seconds)

    except KeyboardInterrupt:
        print("\nStopped live ERA5 download monitor.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit ERA5 usable raw NetCDF files on disk."
    )

    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument("--end-year", type=int, default=END_YEAR)
    parser.add_argument("--end-month", type=int, default=END_MONTH)
    parser.add_argument("--expected-single-nc-count", type=int, default=EXPECTED_SINGLE_NC_COUNT)
    parser.add_argument("--save-csv", action="store_true")
    parser.add_argument("--watch", action="store_true", help="Refresh the audit repeatedly.")
    parser.add_argument("--refresh-seconds", type=int, default=10)

    args = parser.parse_args()

    if args.watch:
        run_watch(args)
    else:
        run_once(args)


if __name__ == "__main__":
    main()