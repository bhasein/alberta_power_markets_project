# src/run_pipeline.py

"""
================================================================================
PURPOSE:
    Run the entire Alberta power markets data pipeline end-to-end, in the
    correct dependency order, with a single command.

WHY THIS FILE IS USEFUL:
    Adding or changing a feature in any feature_engineering script means the
    downstream master_hourly.parquet is stale until every dependent stage is
    rebuilt. Rather than manually re-running each script in the right order,
    this file wires the whole pipeline together and reruns it top to bottom.

PIPELINE ORDER:
    Stage 1 - Preprocessing (raw -> data/processed/preprocessing)
        era5_preprocessing        (slow; produces monthly NetCDFs used below)
        pa_preprocessing
        outages_preprocessing
        interties_hour_ahead_preprocessing
        intertie_capability_preprocessing
        generation_preprocessing
        area_load_preprocessing

    Stage 2 - Feature engineering
        (data/processed/preprocessing -> data/processed/feature_engineering)
        calendar_features          (no preprocessing dependency)
        market_features            (needs pa, interties_hour_ahead, outages)
        generation_features        (needs generation_by_fuel, pa)
        load_weather_features      (needs area_load, era5 monthly)
        renewable_weather_features (needs wind/solar project files, era5 monthly)

    Stage 3 - Master merge
        master_preprocessing       (needs everything produced above)

USAGE:
    Run the full pipeline, reusing any existing outputs:

        python src/run_pipeline.py

    Force every stage to rebuild from scratch:

        python src/run_pipeline.py --overwrite

    Also emit CSV copies alongside the canonical Parquet files:

        python src/run_pipeline.py --overwrite --write-csv

    Skip one or more stages by name (comma-separated):

        python src/run_pipeline.py --skip era5,renewable_weather_features

    Run a stage and all of its prerequisites. Existing current prerequisites
    will skip quickly, while stale ones rebuild before the requested stage:

        python src/run_pipeline.py --only master

    Continue past a failed stage instead of stopping immediately:

        python src/run_pipeline.py --continue-on-failure
================================================================================
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Callable

# ============================================================================
# Make sibling packages (preprocessing, feature_engineering) importable
# ============================================================================

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing import era5_preprocessing
from preprocessing import pa_preprocessing
from preprocessing import outages_preprocessing
from preprocessing import interties_hour_ahead_preprocessing
from preprocessing import intertie_capability_preprocessing
from preprocessing import generation_preprocessing
from preprocessing import area_load_preprocessing
from preprocessing import master_preprocessing

from feature_engineering import calendar_features
from feature_engineering import market_features
from feature_engineering import generation_features
from feature_engineering import load_weather_features
from feature_engineering import renewable_weather_features


# ============================================================================
# Stage registry
# ============================================================================

# Each stage is (name, callable). The callable must accept overwrite=bool
# and return a result dict with at least a "pass" key, matching the pattern
# used by every process_*/build_* function in this project.
#
# write_csv is only passed to stages whose function signature accepts it;
# see run_stage() below.
STAGES: list[tuple[str, Callable[..., dict[str, Any]]]] = [
    # ---- Stage 1: preprocessing --------------------------------------------
    ("era5", era5_preprocessing.process_era5),
    ("pa", pa_preprocessing.process_pa),
    ("outages", outages_preprocessing.process_outages),
    (
        "interties_hour_ahead",
        interties_hour_ahead_preprocessing.process_interties,
    ),
    (
        "intertie_capability",
        intertie_capability_preprocessing.process_intertie_capability,
    ),
    ("generation", generation_preprocessing.process_generation),
    ("area_load", area_load_preprocessing.process_area_load),

    # ---- Stage 2: feature engineering ---------------------------------------
    ("calendar_features", calendar_features.process_calendar_features),
    ("market_features", market_features.build_market_features),
    ("generation_features", generation_features.build_generation_features),
    ("load_weather_features", load_weather_features.build_load_weather_features),
    ("renewable_weather_features", renewable_weather_features.build_weather_features),

    # ---- Stage 3: master merge ----------------------------------------------
    ("master", master_preprocessing.process_master),
]

STAGE_DEPENDENCIES: dict[str, set[str]] = {
    "era5": set(),
    "pa": set(),
    "outages": set(),
    "interties_hour_ahead": set(),
    "intertie_capability": set(),
    "generation": set(),
    "area_load": set(),
    "calendar_features": set(),
    "market_features": {"pa", "outages", "interties_hour_ahead"},
    "generation_features": {"generation", "pa"},
    "load_weather_features": {"era5", "area_load"},
    "renewable_weather_features": {"era5"},
    "master": {
        "calendar_features",
        "market_features",
        "intertie_capability",
        "generation_features",
        "load_weather_features",
        "renewable_weather_features",
    },
}

# Functions that accept write_csv as a kwarg. Everything else (era5, and the
# raw preprocessing scripts that only ever emit CSV+Parquet together) does
# not take this argument, so we don't pass it there.
STAGES_WITH_WRITE_CSV = {
    "calendar_features",
    "market_features",
    "generation_features",
    "load_weather_features",
    "renewable_weather_features",
    "master",
}

# era5_preprocessing.process_era5 takes `quick` instead of `write_csv`.
STAGES_WITH_QUICK = {
    "era5",
}


# ============================================================================
# Runner
# ============================================================================

def expand_dependencies(stage_names: set[str]) -> set[str]:
    """Return requested stages plus every transitive prerequisite."""

    expanded: set[str] = set()

    def include(name: str) -> None:
        """Add one stage and recursively include its prerequisites."""

        if name in expanded:
            return
        expanded.add(name)
        for dependency in STAGE_DEPENDENCIES[name]:
            include(dependency)

    for stage_name in stage_names:
        include(stage_name)
    return expanded

def run_stage(
    name: str,
    func: Callable[..., dict[str, Any]],
    overwrite: bool,
    write_csv: bool,
    quick: bool,
) -> dict[str, Any]:
    """Run one pipeline stage and return its result dictionary."""

    kwargs: dict[str, Any] = {"overwrite": overwrite}

    if name in STAGES_WITH_WRITE_CSV:
        kwargs["write_csv"] = write_csv

    if name in STAGES_WITH_QUICK:
        kwargs["quick"] = quick

    print("\n" + "#" * 80)
    print(f"# STAGE: {name}")
    print("#" * 80)

    started = time.perf_counter()

    try:
        result = func(**kwargs)
    except Exception as exc:
        elapsed = round(time.perf_counter() - started, 3)
        print(f"[{name}] raised an unexpected exception: {exc!r}")
        return {
            "stage": name,
            "status": "exception",
            "pass": False,
            "error": repr(exc),
            "processing_seconds": elapsed,
        }

    result = dict(result)
    result.setdefault("stage", name)

    return result


def print_pipeline_summary(results: list[dict[str, Any]]) -> None:
    """Print a compact table summarizing every stage that ran."""

    print("\n" + "=" * 80)
    print("PIPELINE SUMMARY")
    print("=" * 80)

    total_seconds = 0.0

    for result in results:
        stage = result.get("stage", "unknown")
        status = result.get("status", "unknown")
        passed = result.get("pass", False)
        seconds = result.get("processing_seconds", 0.0) or 0.0
        total_seconds += float(seconds)

        icon = "✓" if passed else "✗"
        print(f"  {icon} {stage:<28} status={status:<20} time={seconds}s")

    overall_pass = all(result.get("pass", False) for result in results)

    print("-" * 80)
    print(f"Total stages run : {len(results)}")
    print(f"Total time       : {round(total_seconds, 3)}s")
    print(f"Overall pass     : {overall_pass}")
    print("=" * 80)


def run_pipeline(
    overwrite: bool = False,
    write_csv: bool = False,
    quick_era5: bool = False,
    skip: set[str] | None = None,
    only: set[str] | None = None,
    continue_on_failure: bool = False,
) -> list[dict[str, Any]]:
    """
    Run the full pipeline in dependency order.

    If `only` is provided, run those stages and all transitive prerequisites
    in pipeline order. Otherwise run everything except whatever is in `skip`.
    """

    skip = skip or set()
    valid_names = set(STAGE_DEPENDENCIES)
    unknown = (skip | (only or set())) - valid_names
    if unknown:
        raise ValueError(f"Unknown pipeline stages: {sorted(unknown)}")
    if only is not None and not only:
        raise ValueError("The requested stage set is empty.")

    selected = expand_dependencies(only) if only is not None else None
    runnable = selected if selected is not None else valid_names - skip
    if not runnable:
        raise ValueError("No pipeline stages were selected.")
    results: list[dict[str, Any]] = []

    for name, func in STAGES:
        if selected is not None:
            if name not in selected:
                continue
        elif name in skip:
            print(f"\n[skip] {name}")
            continue

        result = run_stage(
            name=name,
            func=func,
            overwrite=overwrite,
            write_csv=write_csv,
            quick=quick_era5,
        )

        results.append(result)

        if name == "area_load" and result.get("pass", False):
            print(
                "  Area load: 2024 observed data extended through "
                "the end of 2025 using a frozen 2024 load distribution."
            )

        if not result.get("pass", False) and not continue_on_failure:
            print(
                f"\nStopping pipeline: stage '{name}' did not pass "
                f"(status={result.get('status')}). "
                "Pass --continue-on-failure to run remaining stages anyway."
            )
            break

    return results


# ============================================================================
# CLI
# ============================================================================

def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for the pipeline runner."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the full Alberta power markets pipeline: preprocessing, "
            "feature engineering, and the master merge, in dependency order."
        )
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force every stage to rebuild, even if outputs already exist.",
    )

    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Also write CSV copies alongside canonical Parquet outputs.",
    )

    parser.add_argument(
        "--quick-era5",
        action="store_true",
        help="Skip expensive ERA5 variable/meteorology audits (still writes outputs).",
    )

    parser.add_argument(
        "--skip",
        default="",
        help=(
            "Comma-separated stage names to skip "
            "(see the module docstring for names)."
        ),
    )

    parser.add_argument(
        "--only",
        default="",
        help=(
            "Comma-separated stages to run with all prerequisites "
            "(ignores --skip)."
        ),
    )

    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Keep running remaining stages even if an earlier stage fails.",
    )

    return parser


def parse_stage_list(raw: str) -> set[str]:
    """Parse a comma-separated CLI stage list into unique names."""

    return {item.strip() for item in raw.split(",") if item.strip()}


def main() -> None:
    """Run the requested stages and return a failing process status on error."""

    parser = build_argument_parser()
    args = parser.parse_args()

    skip = parse_stage_list(args.skip)
    only = parse_stage_list(args.only) or None

    valid_names = {name for name, _ in STAGES}
    for name in skip | (only or set()):
        if name not in valid_names:
            raise SystemExit(
                f"Unknown stage name: '{name}'. Valid stages: {sorted(valid_names)}"
            )

    started = time.perf_counter()

    try:
        results = run_pipeline(
            overwrite=args.overwrite,
            write_csv=args.write_csv,
            quick_era5=args.quick_era5,
            skip=skip,
            only=only,
            continue_on_failure=args.continue_on_failure,
        )
    except ValueError as exc:
        parser.error(str(exc))

    print_pipeline_summary(results)

    print(f"\nTotal wall-clock time: {round(time.perf_counter() - started, 3)}s")

    if not all(result.get("pass", False) for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
