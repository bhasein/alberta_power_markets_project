"""Audit or monitor the completeness of raw ERA5 downloads on disk."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    ERA5_PRESSURE_LEVEL_DIR,
    ERA5_RAW_DIR,
    ERA5_SINGLE_LEVEL_DIR,
    PIPELINE_END_MONTH,
    PIPELINE_END_YEAR,
    PIPELINE_START_YEAR,
    PROJECT_ROOT,
)
from era5.era5_downloader import (
    PRESSURE_REQUESTS,
    SINGLE_LEVEL_FILENAMES,
    pressure_output_path,
    valid_nc,
    valid_single_folder,
)


def clear_terminal() -> None:
    """Clear an interactive terminal without launching a shell command."""

    print("\033[2J\033[H", end="")


def audit_era5_downloads(
    project_root: Path = PROJECT_ROOT,
    start_year: int = PIPELINE_START_YEAR,
    end_year: int = PIPELINE_END_YEAR,
    end_month: int = PIPELINE_END_MONTH,
) -> pd.DataFrame:
    """Return monthly file-presence and content-validation results."""

    if project_root == PROJECT_ROOT:
        single_dir = ERA5_SINGLE_LEVEL_DIR
        pressure_dir = ERA5_PRESSURE_LEVEL_DIR
    else:
        era5_dir = project_root / "data" / "raw" / "weather" / "era5"
        single_dir = era5_dir / "single_levels"
        pressure_dir = era5_dir / "pressure_levels"

    records: list[dict[str, object]] = []
    for year in range(start_year, end_year + 1):
        last_month = end_month if year == end_year else 12
        for month in range(1, last_month + 1):
            period = f"{year}-{month:02d}"
            folder = single_dir / f"era5_single_levels_alberta_{year}_{month:02d}"
            single_paths = [folder / name for name in SINGLE_LEVEL_FILENAMES]
            single_present = sum(path.exists() for path in single_paths)
            single_complete = valid_single_folder(folder, year, month)

            pressure_paths = [
                pressure_dir / pressure_output_path(year, month, request).name
                for request in PRESSURE_REQUESTS
            ]
            pressure_present = sum(path.exists() for path in pressure_paths)
            pressure_valid = [
                valid_nc(path, year, month)
                for path in pressure_paths
            ]
            pressure_complete = all(pressure_valid)

            records.append(
                {
                    "period": period,
                    "year": year,
                    "month": month,
                    "single_files_present": single_present,
                    "single_files_expected": len(SINGLE_LEVEL_FILENAMES),
                    "single_complete": single_complete,
                    "pressure_files_present": pressure_present,
                    "pressure_files_expected": len(PRESSURE_REQUESTS),
                    "pressure_files_valid": sum(pressure_valid),
                    "pressure_complete": pressure_complete,
                    "month_complete": single_complete and pressure_complete,
                }
            )
    return pd.DataFrame(records)


def save_audit_csv(audit: pd.DataFrame, project_root: Path) -> Path:
    """Save the raw-download audit beside the ERA5 source archive."""

    output_path = (
        ERA5_RAW_DIR
        if project_root == PROJECT_ROOT
        else project_root / "data" / "raw" / "weather" / "era5"
    ) / "era5_download_audit.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_path, index=False)
    return output_path


def print_summary(download_audit: pd.DataFrame) -> None:
    """Print a concise summary of valid ERA5 months and files."""

    complete_months = int(download_audit["month_complete"].sum())
    total_months = len(download_audit)
    valid_pressure = int(download_audit["pressure_files_valid"].sum())
    expected_pressure = int(download_audit["pressure_files_expected"].sum())
    valid_single_months = int(download_audit["single_complete"].sum())

    print("=" * 80)
    print("ERA5 RAW DOWNLOAD VALIDATION")
    print("=" * 80)
    print(f"Last refreshed:               {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Valid single-level months:    {valid_single_months}/{total_months}")
    print(f"Valid pressure-level files:   {valid_pressure}/{expected_pressure}")
    print(f"Complete validated months:    {complete_months}/{total_months}")


def run_once(args: argparse.Namespace) -> bool:
    """Run one validation pass and return whether every month is complete."""

    audit = audit_era5_downloads(
        project_root=args.project_root,
        start_year=args.start_year,
        end_year=args.end_year,
        end_month=args.end_month,
    )
    print_summary(audit)
    if args.save_csv:
        output_path = save_audit_csv(audit, args.project_root)
        print(f"\nSaved audit CSV to: {output_path}")
    return bool(audit["month_complete"].all()) if not audit.empty else False


def run_watch(args: argparse.Namespace) -> None:
    """Repeat validation until interrupted by the user."""

    try:
        while True:
            clear_terminal()
            run_once(args)
            print(f"\nRefreshing every {args.refresh_seconds} seconds. Ctrl+C stops.")
            time.sleep(args.refresh_seconds)
    except KeyboardInterrupt:
        print("\nStopped live ERA5 download monitor.")


def build_argument_parser() -> argparse.ArgumentParser:
    """Build command-line arguments for one-shot and watch modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--start-year", type=int, default=PIPELINE_START_YEAR)
    parser.add_argument("--end-year", type=int, default=PIPELINE_END_YEAR)
    parser.add_argument("--end-month", type=int, default=PIPELINE_END_MONTH)
    parser.add_argument("--save-csv", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--refresh-seconds", type=int, default=10)
    return parser


def main() -> None:
    """Run the ERA5 disk validator and signal incomplete one-shot audits."""

    parser = build_argument_parser()
    args = parser.parse_args()
    if args.refresh_seconds <= 0:
        parser.error("--refresh-seconds must be positive")
    if args.watch:
        run_watch(args)
    elif not run_once(args):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
