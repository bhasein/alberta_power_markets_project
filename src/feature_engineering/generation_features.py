# src/feature_engineering/generation_features.py

"""
Build one canonical hourly Alberta generation-feature dataset.

Inputs
------
The script searches for these preprocessed datasets:

1. Generation-by-fuel data
   - generation_by_fuel_preprocessed.parquet

2. P&A / price-load-gas data (AIL only)
   - pa_hourly_preprocessed.parquet
   - pa_preprocessed.parquet
   - price_ail_gas_preprocessed.parquet
   - p_and_a_preprocessed.parquet

Outputs
-------
Canonical feature output:

    data/processed/feature_engineering/generation/generation_features_hourly.parquet

Optional full CSV output:

    data/processed/feature_engineering/generation/generation_features_hourly.csv

Audit outputs:

    data/audits/feature_engineering/generation_features_audit_checks.csv
    data/audits/feature_engineering/generation_features_feature_summary.csv
    data/audits/feature_engineering/generation_features_source_summary.csv

Run
---
Standard run:

    python src/feature_engineering/generation_features.py

Overwrite existing outputs:

    python src/feature_engineering/generation_features.py \
        --overwrite

Write CSV as well:

    python src/feature_engineering/generation_features.py \
        --overwrite \
        --write-csv

Verbose logging:

    python src/feature_engineering/generation_features.py \
        --verbose

Design
------
The generation-by-fuel table is the hourly backbone. AIL is left-joined onto
it, used only to derive net load and load-relative generation shares. Hours
outside a source's historical coverage remain missing and are identified
with explicit source-availability flags.

This file intentionally does not depend on market_features_hourly.parquet.
It loads AIL directly from the preprocessed P&A table so that
generation_features.py and market_features.py remain independent of each
other and can be run in either order.

The file keeps feature logic modular. Add or remove functions from
FEATURE_BUILDERS without changing the input/output pipeline.

Information timing
------------------
Current-hour generation, load shares, net load, and change features are
contemporaneous/ex-post. Lag and ``prior`` rolling columns are historical.
The audit feature summary records this timing classification explicitly.
"""

# ============================================================================
# Imports
# ============================================================================

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

try:
    from .shared import (
        add_changes_through_current_hour as add_changes,
        add_check,
        add_lags,
        add_prior_rolling_statistics as add_rolling_statistics,
        apply_feature_builders as run_feature_builders,
        audit_passed,
        build_manifest,
        classify_feature_timing,
        configure_logging,
        ensure_directories,
        feature_code_paths,
        ensure_src_on_path,
        existing_outputs_satisfy_request as outputs_satisfy_request,
        find_first_column,
        load_parquet_table,
        merge_hourly_sources,
        numeric_feature_summary,
        numericize_except_timestamp,
        output_is_current,
        read_existing_parquet,
        require_columns,
        resolve_file,
        safe_divide,
        save_feature_outputs as write_feature_outputs,
        save_tables,
        source_summary,
        write_manifest,
    )
except ImportError:  # Support direct execution of this file.
    from shared import (
        add_changes_through_current_hour as add_changes,
        add_check,
        add_lags,
        add_prior_rolling_statistics as add_rolling_statistics,
        apply_feature_builders as run_feature_builders,
        audit_passed,
        build_manifest,
        classify_feature_timing,
        configure_logging,
        ensure_directories,
        feature_code_paths,
        ensure_src_on_path,
        existing_outputs_satisfy_request as outputs_satisfy_request,
        find_first_column,
        load_parquet_table,
        merge_hourly_sources,
        numeric_feature_summary,
        numericize_except_timestamp,
        output_is_current,
        read_existing_parquet,
        require_columns,
        resolve_file,
        safe_divide,
        save_feature_outputs as write_feature_outputs,
        save_tables,
        source_summary,
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
    PREPROCESSING_DIR,
    FEATURES_DIR,
    FEATURE_ENGINEERING_AUDITS_DIR,
    PROJECT_ROOT,
)


OUTPUT_DIR = (
    FEATURES_DIR
    / "generation"
)

OUTPUT_PARQUET = (
    OUTPUT_DIR
    / "generation_features_hourly.parquet"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "generation_features_hourly.csv"
)

OUTPUT_FILE = OUTPUT_PARQUET


AUDIT_DIR = FEATURE_ENGINEERING_AUDITS_DIR


AUDIT_FILE = (
    AUDIT_DIR
    / "generation_features_audit_checks.csv"
)

FEATURE_SUMMARY_FILE = (
    AUDIT_DIR
    / "generation_features_feature_summary.csv"
)

SOURCE_SUMMARY_FILE = (
    AUDIT_DIR
    / "generation_features_source_summary.csv"
)


GENERATION_FILE_CANDIDATES = [
    PREPROCESSING_DIR / "generation_by_fuel.parquet",
]


PA_FILE_CANDIDATES = [
    PREPROCESSING_DIR / "pa_hourly_preprocessed.parquet",
]


# ============================================================================
# Configuration
# ============================================================================

LOCAL_TIMEZONE = "America/Edmonton"
DATASET_NAME = "generation_features"
FEATURE_INFORMATION_POLICY = "mixed_ex_post_and_historical"

FUEL_TYPES = [
    "coal",
    "cogeneration",
    "combined_cycle",
    "dual_fuel",
    "gas_fired_steam",
    "hydro",
    "other",
    "simple_cycle",
    "solar",
    "storage",
    "wind",
]

# Storage generation can legitimately be negative while charging, so it is
# excluded from the non-negative generation audit check further below.
NON_NEGATIVE_EXEMPT_FUELS = {
    "storage",
}

RENEWABLE_FUELS = [
    "wind",
    "solar",
    "hydro",
]

GENERATION_COLUMNS = [
    f"{fuel}_system_generation"
    for fuel in FUEL_TYPES
]

RENEWABLE_GENERATION_COLUMNS = [
    f"{fuel}_system_generation"
    for fuel in RENEWABLE_FUELS
]

REQUIRED_BACKBONE_COLUMNS = [
    "timestamp_utc",
    "total_system_generation",
    *GENERATION_COLUMNS,
    "generation_available",
]

NET_LOAD_LAGS = [1, 3, 6, 12, 24, 48, 168]
NET_LOAD_CHANGE_LAGS = [1, 3, 6, 12, 24]
NET_LOAD_ROLLING_WINDOWS = [3, 6, 24, 72, 168]

# Tolerance for reconciling summed per-fuel generation against the reported
# total. Historical AESO fuel-category changes (dual fuel introduced 2018,
# gas-fired steam split out in 2021, coal retired 2024, etc.) mean this will
# not reconcile for every hour, which is why it is a warning-level check.
GENERATION_RECONCILIATION_TOLERANCE_MW = 1.0


# ============================================================================
# Source loading and standardization
# ============================================================================

def load_generation_data(
    path: Path,
) -> pd.DataFrame:
    """Load the generation-by-fuel hourly backbone."""
    frame = numericize_except_timestamp(
        load_parquet_table(
            path,
            "generation",
        )
    )

    require_columns(
        frame,
        [
            "timestamp_utc",
            "total_system_generation",
            *GENERATION_COLUMNS,
        ],
        "Generation",
    )

    frame["generation_available"] = 1

    return frame


def load_ail_data(
    path: Path,
) -> pd.DataFrame:
    """Load AIL alone from the independent P&A source."""
    frame = numericize_except_timestamp(
        load_parquet_table(
            path,
            "P&A (AIL)",
        )
    )

    aliases = {
        "ail_mw": [
            "ail",
            "actual_ail",
            "ail_mw",
        ],
    }

    rename = {}

    for canonical, candidates in aliases.items():
        observed = find_first_column(
            frame.columns,
            candidates,
        )

        if observed is not None:
            rename[observed] = canonical

    frame = frame.rename(
        columns=rename
    )

    require_columns(
        frame,
        [
            "timestamp_utc",
            "ail_mw",
        ],
        "P&A (AIL)",
    )

    frame = frame[
        [
            "timestamp_utc",
            "ail_mw",
        ]
    ].copy()

    frame["ail_available"] = 1

    return frame


# ============================================================================
# Generation and net-load features
# ============================================================================

def add_renewable_generation_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Add renewable generation totals and generation shares."""

    output = frame.copy()

    # Wind, solar, and hydro form the project-wide renewable definition.
    if set(RENEWABLE_GENERATION_COLUMNS).issubset(output.columns):
        output["renewable_generation_mw"] = output[
            RENEWABLE_GENERATION_COLUMNS
        ].sum(
            axis=1,
            min_count=1,
        )

    # This share uses reported total generation as its denominator; the
    # load-relative renewable share is constructed separately below.
    if {
        "renewable_generation_mw",
        "total_system_generation",
    }.issubset(output.columns):
        output["renewable_share"] = safe_divide(
            output["renewable_generation_mw"].to_numpy(dtype=float),
            output["total_system_generation"].to_numpy(dtype=float),
        )

    return output


def add_net_load_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add net load and its lag, change, and rolling-statistic features.

    Net load is defined as AIL minus wind and solar generation, i.e. the
    demand remaining after variable renewable output. Both wind and solar
    are subtracted so the definition matches the province-wide renewable
    fleet rather than wind alone.
    """

    output = frame.copy()

    # Net load is contemporaneous and therefore ex-post at timestamp t.
    if {
        "ail_mw",
        "wind_system_generation",
        "solar_system_generation",
    }.issubset(output.columns):
        output["net_load_mw"] = (
            output["ail_mw"]
            - output["wind_system_generation"]
            - output["solar_system_generation"]
        )

        # Lags and prior rolling statistics are historical. Change fields use
        # the current observation and are explicitly classified as ex-post.
        output = add_lags(
            output,
            "net_load_mw",
            NET_LOAD_LAGS,
        )

        output = add_changes(
            output,
            "net_load_mw",
            NET_LOAD_CHANGE_LAGS,
        )

        output = add_rolling_statistics(
            output,
            "net_load_mw",
            NET_LOAD_ROLLING_WINDOWS,
        )

    return output


def add_share_of_load_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Add wind, solar, and renewable output as shares of AIL."""

    output = frame.copy()

    # Each ratio uses the same AIL denominator so the component shares remain
    # directly comparable. Zero or missing AIL produces a missing ratio.
    if {
        "wind_system_generation",
        "ail_mw",
    }.issubset(output.columns):
        output["wind_share_of_load"] = safe_divide(
            output["wind_system_generation"].to_numpy(dtype=float),
            output["ail_mw"].to_numpy(dtype=float),
        )

    if {
        "solar_system_generation",
        "ail_mw",
    }.issubset(output.columns):
        output["solar_share_of_load"] = safe_divide(
            output["solar_system_generation"].to_numpy(dtype=float),
            output["ail_mw"].to_numpy(dtype=float),
        )

    if {
        "renewable_generation_mw",
        "ail_mw",
    }.issubset(output.columns):
        output["renewable_share_of_load"] = safe_divide(
            output["renewable_generation_mw"].to_numpy(dtype=float),
            output["ail_mw"].to_numpy(dtype=float),
        )

    return output


FEATURE_BUILDERS: list[
    Callable[
        [pd.DataFrame],
        pd.DataFrame,
    ]
] = [
    add_renewable_generation_features,
    add_net_load_features,
    add_share_of_load_features,
]


# ============================================================================
# Merge and audit
# ============================================================================

def merge_sources(
    generation: pd.DataFrame,
    ail: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join AIL to the complete generation backbone."""
    return merge_hourly_sources(
        generation,
        [ail],
        ["generation_available", "ail_available"],
    )


def audit_generation_features(
    frame: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    bool,
]:
    """Audit generation features and build their numeric catalog."""
    rows: list[dict] = []

    timestamps = pd.DatetimeIndex(
        pd.to_datetime(
            frame["timestamp_utc"],
            utc=True,
        )
    )

    expected = pd.date_range(
        timestamps.min(),
        timestamps.max(),
        freq="h",
        tz="UTC",
    )

    add_check(
        rows,
        "row_count_positive",
        len(frame) > 0,
        len(frame),
        "> 0",
    )

    add_check(
        rows,
        "timestamps_unique",
        not timestamps.has_duplicates,
        int(timestamps.duplicated().sum()),
        0,
    )

    add_check(
        rows,
        "timestamps_monotonic",
        timestamps.is_monotonic_increasing,
        timestamps.is_monotonic_increasing,
        True,
    )

    add_check(
        rows,
        "complete_generation_backbone",
        len(expected.difference(timestamps)) == 0,
        len(expected.difference(timestamps)),
        0,
    )

    add_check(
        rows,
        "total_system_generation_present",
        frame["total_system_generation"].notna().all(),
        int(frame["total_system_generation"].isna().sum()),
        0,
        severity="warning",
        notes=(
            "Structural gaps are possible before a fuel category was "
            "separately reported by AESO."
        ),
    )

    if "ail_mw" in frame.columns:
        add_check(
            rows,
            "ail_mw_present",
            frame["ail_mw"].notna().all(),
            int(frame["ail_mw"].isna().sum()),
            0,
        )

    for flag in [
        "generation_available",
        "ail_available",
    ]:
        invalid = int(
            (~frame[flag].isin([0, 1])).sum()
        )

        add_check(
            rows,
            f"{flag}_binary",
            invalid == 0,
            invalid,
            0,
        )

        add_check(
            rows,
            f"{flag}_hours",
            True,
            int(frame[flag].sum()),
            "recorded",
            severity="info",
        )

    # Generation should not be negative, with the exception of storage,
    # which can report negative generation while charging.
    for fuel in FUEL_TYPES:
        if fuel in NON_NEGATIVE_EXEMPT_FUELS:
            continue

        column = f"{fuel}_system_generation"

        if column not in frame.columns:
            continue

        negative = int(
            frame[column]
            .lt(0)
            .sum()
        )

        add_check(
            rows,
            f"{column}_non_negative",
            negative == 0,
            negative,
            0,
        )

    # Reconcile summed per-fuel generation against the reported total.
    #
    # This is warning-level, not error-level, because AESO fuel-category
    # coverage changed over the study period (dual fuel introduced 2018,
    # gas-fired steam split out in 2021, coal retired 2024, etc.), so exact
    # reconciliation is not expected for every hour.
    summed_generation = frame[GENERATION_COLUMNS].sum(
        axis=1,
        min_count=1,
    )

    reconciliation_gap = (
        summed_generation
        - frame["total_system_generation"]
    ).abs()

    unreconciled_hours = int(
        (
            reconciliation_gap
            > GENERATION_RECONCILIATION_TOLERANCE_MW
        )
        .sum()
    )

    add_check(
        rows,
        "generation_components_reconcile_to_total",
        unreconciled_hours == 0,
        unreconciled_hours,
        0,
        severity="warning",
        notes=(
            "Expected to diverge during periods when a fuel category was "
            "not yet separately reported by AESO."
        ),
    )

    if "renewable_share" in frame.columns:
        out_of_range = int(
            (
                frame["renewable_share"].lt(-0.01)
                | frame["renewable_share"].gt(1.05)
            )
            .sum()
        )

        add_check(
            rows,
            "renewable_share_within_expected_range",
            out_of_range == 0,
            out_of_range,
            0,
            severity="warning",
            notes="Expected range is approximately [0, 1].",
        )

    for column in [
        "wind_share_of_load",
        "solar_share_of_load",
        "renewable_share_of_load",
    ]:
        if column not in frame.columns:
            continue

        negative = int(
            frame[column]
            .lt(0)
            .sum()
        )

        add_check(
            rows,
            f"{column}_non_negative",
            negative == 0,
            negative,
            0,
            severity="warning",
        )

    if "net_load_mw" in frame.columns:
        negative_hours = int(
            frame["net_load_mw"]
            .lt(0)
            .sum()
        )

        add_check(
            rows,
            "net_load_negative_hours",
            True,
            negative_hours,
            "recorded",
            severity="info",
            notes=(
                "Negative net load is physically valid during high-"
                "renewable, low-demand hours and is not itself an error."
            ),
        )

    summary = numeric_feature_summary(
        frame,
        timing_classifier=lambda column: classify_feature_timing(
            column,
            target_derived_prefixes={
                "net_load_mw_change_",
                "wind_share_of_load",
                "solar_share_of_load",
                "renewable_share_of_load",
            },
        ),
    )

    audit = pd.DataFrame(rows)
    return audit, summary, audit_passed(audit)


def print_report(
    frame: pd.DataFrame,
    audit: pd.DataFrame,
    passed: bool,
) -> None:
    """Print a compact human-readable generation audit report."""
    timestamps = pd.DatetimeIndex(
        frame["timestamp_utc"]
    )

    failed = audit.loc[
        ~audit["pass"]
    ]

    print("\n" + "=" * 80)
    print("GENERATION FEATURES AUDIT")
    print("=" * 80)
    print(f"Overall pass       : {passed}")
    print(f"Rows               : {len(frame):,}")
    print(f"Columns            : {len(frame.columns):,}")
    print(f"Start UTC          : {timestamps.min()}")
    print(f"End UTC            : {timestamps.max()}")
    print(f"Generation coverage: {int(frame['generation_available'].sum()):,}")
    print(f"AIL coverage       : {int(frame['ail_available'].sum()):,}")

    print("\nFailed checks:")

    if failed.empty:
        print("  None")
    else:
        for _, row in failed.iterrows():
            print(
                f"  - {row['check']} [{row['severity']}] "
                f"observed={row['observed']} expected={row['expected']}"
            )

    print("=" * 80)


# ============================================================================
# Pipeline
# ============================================================================

def build_generation_features(
    overwrite: bool = False,
    write_csv: bool = False,
) -> dict[str, Any]:
    """Load, merge, engineer, audit, and save generation features."""

    started = time.perf_counter()

    LOGGER.info("Starting generation-feature pipeline.")
    LOGGER.debug("Project root: %s", PROJECT_ROOT)
    LOGGER.debug("Preprocessing directory: %s", PREPROCESSING_DIR)
    LOGGER.debug("Feature output directory: %s", OUTPUT_DIR)
    LOGGER.debug("Audit output directory: %s", AUDIT_DIR)

    ensure_directories(OUTPUT_DIR, AUDIT_DIR)

    generation_path = resolve_file(GENERATION_FILE_CANDIDATES, "generation")
    ail_path = resolve_file(PA_FILE_CANDIDATES, "P&A (AIL)")
    expected_manifest = build_manifest(
        dataset=DATASET_NAME,
        source_paths=[generation_path, ail_path],
        code_paths=feature_code_paths(Path(__file__)),
        configuration={
            "feature_information_policy": FEATURE_INFORMATION_POLICY,
            "net_load_lags": NET_LOAD_LAGS,
            "net_load_rolling_windows": NET_LOAD_ROLLING_WINDOWS,
        },
    )

    # Skip only when all explicitly requested feature outputs already exist.
    if (
        not overwrite
        and outputs_satisfy_request(
            OUTPUT_PARQUET,
            OUTPUT_CSV,
            write_csv=write_csv,
            expected_manifest=expected_manifest,
            required_artifacts=[
                AUDIT_FILE,
                FEATURE_SUMMARY_FILE,
                SOURCE_SUMMARY_FILE,
            ],
        )
    ):
        LOGGER.info(
            "Requested generation-feature outputs already exist. "
            "Use --overwrite to rebuild them."
        )

        return {
            "dataset": DATASET_NAME,
            "status": "skipped_existing",
            "pass": True,
            "rows": None,
            "columns": None,
            "parquet_file": str(OUTPUT_PARQUET),
            "csv_file": (
                str(OUTPUT_CSV)
                if write_csv
                else "not requested"
            ),
            "audit_file": (
                str(AUDIT_FILE)
                if AUDIT_FILE.exists()
                else "not available"
            ),
            "feature_summary_file": (
                str(FEATURE_SUMMARY_FILE)
                if FEATURE_SUMMARY_FILE.exists()
                else "not available"
            ),
            "source_summary_file": (
                str(SOURCE_SUMMARY_FILE)
                if SOURCE_SUMMARY_FILE.exists()
                else "not available"
            ),
            "processing_seconds": round(
                time.perf_counter()
                - started,
                3,
            ),
        }

    # If the canonical Parquet already exists and CSV is the only missing
    # requested representation, create it directly without rebuilding.
    if (
        not overwrite
        and output_is_current(OUTPUT_PARQUET, expected_manifest)
        and write_csv
        and not OUTPUT_CSV.exists()
    ):
        frame = read_existing_parquet(OUTPUT_PARQUET)

        LOGGER.info(
            "Creating missing CSV from existing canonical Parquet."
        )

        frame.to_csv(
            OUTPUT_CSV,
            index=False,
        )
        write_manifest(OUTPUT_CSV, expected_manifest)

        return {
            "dataset": DATASET_NAME,
            "status": "csv_created_from_existing_parquet",
            "pass": True,
            "rows": len(frame),
            "columns": len(frame.columns),
            "start_utc": str(
                frame["timestamp_utc"].min()
            ),
            "end_utc": str(
                frame["timestamp_utc"].max()
            ),
            "parquet_file": str(OUTPUT_PARQUET),
            "csv_file": str(OUTPUT_CSV),
            "audit_file": (
                str(AUDIT_FILE)
                if AUDIT_FILE.exists()
                else "not available"
            ),
            "feature_summary_file": (
                str(FEATURE_SUMMARY_FILE)
                if FEATURE_SUMMARY_FILE.exists()
                else "not available"
            ),
            "source_summary_file": (
                str(SOURCE_SUMMARY_FILE)
                if SOURCE_SUMMARY_FILE.exists()
                else "not available"
            ),
            "processing_seconds": round(
                time.perf_counter()
                - started,
                3,
            ),
        }

    # ------------------------------------------------------------------------
    # Source resolution and loading
    # ------------------------------------------------------------------------

    generation = load_generation_data(
        generation_path
    )

    ail = load_ail_data(
        ail_path
    )

    source_summary_frame = pd.DataFrame(
        [
            source_summary(
                "generation",
                generation,
                generation_path,
            ),
            source_summary(
                "ail",
                ail,
                ail_path,
            ),
        ]
    )

    LOGGER.info(
        "Loaded sources: generation=%s rows, ail=%s rows.",
        f"{len(generation):,}",
        f"{len(ail):,}",
    )

    # ------------------------------------------------------------------------
    # Merge and feature construction
    # ------------------------------------------------------------------------

    LOGGER.info(
        "Merging AIL onto the generation-by-fuel hourly backbone."
    )

    master = merge_sources(
        generation,
        ail,
    )

    LOGGER.info(
        "Applying %s generation feature builders.",
        len(FEATURE_BUILDERS),
    )

    master = run_feature_builders(master, FEATURE_BUILDERS)

    LOGGER.info(
        "Generation feature table constructed with %s rows and %s columns.",
        f"{len(master):,}",
        f"{len(master.columns):,}",
    )

    # ------------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------------

    audit, feature_summary, passed = audit_generation_features(
        master
    )

    print_report(
        master,
        audit,
        passed,
    )

    # Audit files are retained even when the feature audit fails.
    save_tables(
        {
            AUDIT_FILE: audit,
            FEATURE_SUMMARY_FILE: feature_summary,
            SOURCE_SUMMARY_FILE: source_summary_frame,
        },
        {
            AUDIT_FILE: "generation-feature audit checks",
            FEATURE_SUMMARY_FILE: "generation-feature numeric summary",
            SOURCE_SUMMARY_FILE: "generation-feature source summary",
        },
    )

    if not passed:
        LOGGER.error(
            "Generation-feature audit failed. "
            "Canonical feature outputs were not written."
        )

        return {
            "dataset": DATASET_NAME,
            "status": "audit_failed",
            "pass": False,
            "rows": len(master),
            "columns": len(master.columns),
            "start_utc": str(
                master["timestamp_utc"].min()
            ),
            "end_utc": str(
                master["timestamp_utc"].max()
            ),
            "generation_file": str(generation_path),
            "ail_file": str(ail_path),
            "parquet_file": "not written",
            "csv_file": "not written",
            "audit_file": str(AUDIT_FILE),
            "feature_summary_file": str(FEATURE_SUMMARY_FILE),
            "source_summary_file": str(SOURCE_SUMMARY_FILE),
            "processing_seconds": round(
                time.perf_counter()
                - started,
                3,
            ),
        }

    # ------------------------------------------------------------------------
    # Canonical feature outputs
    # ------------------------------------------------------------------------

    write_feature_outputs(
        master,
        OUTPUT_PARQUET,
        OUTPUT_CSV,
        write_csv,
        "generation-feature",
        manifest=expected_manifest,
    )
    for artifact in [AUDIT_FILE, FEATURE_SUMMARY_FILE, SOURCE_SUMMARY_FILE]:
        write_manifest(artifact, expected_manifest)
    provenance_file = write_manifest(OUTPUT_PARQUET, expected_manifest)

    processing_seconds = round(
        time.perf_counter()
        - started,
        3,
    )

    LOGGER.info(
        "Generation-feature pipeline completed successfully in %.3f seconds.",
        processing_seconds,
    )

    return {
        "dataset": DATASET_NAME,
        "status": "saved",
        "pass": True,
        "rows": len(master),
        "columns": len(master.columns),
        "start_utc": str(
            master["timestamp_utc"].min()
        ),
        "end_utc": str(
            master["timestamp_utc"].max()
        ),
        "generation_hours": int(
            master["generation_available"].sum()
        ),
        "ail_hours": int(
            master["ail_available"].sum()
        ),
        "generation_file": str(generation_path),
        "ail_file": str(ail_path),
        "parquet_file": str(OUTPUT_PARQUET),
        "csv_file": (
            str(OUTPUT_CSV)
            if write_csv
            else "not requested"
        ),
        "audit_file": str(AUDIT_FILE),
        "feature_summary_file": str(FEATURE_SUMMARY_FILE),
        "source_summary_file": str(SOURCE_SUMMARY_FILE),
        "manifest_file": str(provenance_file),
        "processing_seconds": processing_seconds,
    }


# ============================================================================
# CLI
# ============================================================================

def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Build canonical hourly Alberta generation features "
            "from generation-by-fuel and AIL data."
        )
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Rebuild and overwrite existing generation-feature outputs."
        ),
    )

    parser.add_argument(
        "--write-csv",
        action="store_true",
        help=(
            "Also write the full generation feature table to CSV."
        ),
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Enable verbose DEBUG-level logging."
        ),
    )

    return parser


def print_pipeline_result(
    result: dict[str, Any],
) -> None:
    """Print the final pipeline result dictionary."""

    print(
        "\n"
        + "=" * 80
    )

    print(
        "GENERATION FEATURE RESULT"
    )

    print(
        "=" * 80
    )

    for key, value in result.items():
        print(
            f"{key}: {value}"
        )

    print(
        "=" * 80
    )


def main() -> None:
    """Run the generation-feature pipeline from the command line."""

    parser = build_argument_parser()

    args = parser.parse_args()

    configure_logging(
        verbose=args.verbose
    )

    try:
        result = build_generation_features(
            overwrite=args.overwrite,
            write_csv=args.write_csv,
        )

    except Exception:
        LOGGER.exception(
            "Generation-feature pipeline terminated with an unexpected error."
        )
        raise

    print_pipeline_result(
        result
    )

    if not result.get(
        "pass",
        False,
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
