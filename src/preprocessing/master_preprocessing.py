"""
================================================================================
PURPOSE:
    Build and audit the canonical hourly master dataset.

WHY THIS FILE IS USEFUL:
    The project produces several clean source-level and feature-level datasets:
    calendar features, market features, intertie capability, generation,
    load-weather features, and renewable-weather features. This file combines
    those products into one UTC-indexed, one-row-per-hour master parquet while
    explicitly handling overlapping columns. Calendar-derived fields are kept
    from the canonical calendar table and removed from incoming datasets only
    after their values are confirmed to match.

PIPELINE OVERVIEW:
    preprocessed and feature parquet files
        --> load_source_datasets()       reads every required input
        --> merge_master_sources()       validates keys and reconciles overlaps
        --> audit_master_dataset()       validates timeline, schema, and merge logic
        --> build_feature_summary()      summarizes master feature missingness
        --> process_master()             writes parquet and audit CSV outputs
        --> main()                       exposes the pipeline as a CLI script
================================================================================
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys
import time

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    CALENDAR_FEATURES,
    DATA_DIR,
    GENERATION_FEATURES,
    INTERTIE_CAPABILITY_PARQUET,
    LOAD_WEATHER_FEATURES,
    MARKET_FEATURES,
    MASTER_CSV,
    MASTER_PARQUET,
    RENEWABLE_WEATHER_FEATURES,
)
from preprocessing.shared import (
    add_check,
    audit_passes,
    build_manifest,
    outputs_are_current,
    preprocessing_code_paths,
    write_audit_artifacts,
    write_tabular_outputs,
)

AUDIT_DIR = DATA_DIR / "audits" / "master"

OUTPUT_PARQUET = MASTER_PARQUET
OUTPUT_CSV = MASTER_CSV

AUDIT_FILE = AUDIT_DIR / "master_audit_checks.csv"
FEATURE_SUMMARY_FILE = AUDIT_DIR / "master_feature_summary.csv"
SOURCE_SUMMARY_FILE = AUDIT_DIR / "master_source_summary.csv"
RECONCILED_COLUMNS_FILE = AUDIT_DIR / "master_reconciled_columns.csv"


SOURCE_FILES = {
    "calendar": CALENDAR_FEATURES,
    "market": MARKET_FEATURES,
    "intertie_capability": INTERTIE_CAPABILITY_PARQUET,
    "generation": GENERATION_FEATURES,
    "load_weather": LOAD_WEATHER_FEATURES,
    "renewable_weather": RENEWABLE_WEATHER_FEATURES,
}

MERGE_KEY = "timestamp_utc"


def values_match(
    left: pd.Series,
    right: pd.Series,
) -> pd.Series:
    """
    Compare two overlapping columns while treating paired missing values as equal.
    """
    return (
        left.eq(right)
        | (
            left.isna()
            & right.isna()
        )
    )


def load_source_datasets(
    source_files: dict[str, Path] = SOURCE_FILES,
) -> dict[str, pd.DataFrame]:
    """
    Read each required source parquet and normalize timestamp values to UTC.
    """
    datasets = {}

    for dataset_name, path in source_files.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required source file for {dataset_name}: {path}"
            )

        frame = pd.read_parquet(path)

        if MERGE_KEY not in frame.columns:
            raise KeyError(
                f"{MERGE_KEY!r} is missing from source {dataset_name!r}."
            )

        frame = frame.copy()
        frame[MERGE_KEY] = pd.to_datetime(
            frame[MERGE_KEY],
            utc=True,
        )

        datasets[dataset_name] = frame

    return datasets


def build_source_summary(
    datasets: dict[str, pd.DataFrame],
    source_files: dict[str, Path] = SOURCE_FILES,
) -> pd.DataFrame:
    """
    Build one row per source dataset for audit and review.
    """
    rows = []

    for dataset_name, frame in datasets.items():
        path = source_files[dataset_name]
        rows.append({
            "dataset": dataset_name,
            "path": str(path),
            "rows": len(frame),
            "columns": len(frame.columns),
            "start": frame[MERGE_KEY].min(),
            "end": frame[MERGE_KEY].max(),
            "duplicate_timestamps": int(frame[MERGE_KEY].duplicated().sum()),
            "missing_values": int(frame.isna().sum().sum()),
            "file_size_mb": round(path.stat().st_size / 1024**2, 3),
        })

    return pd.DataFrame(rows)


def validate_sources(
    datasets: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Validate that every source has a clean hourly timestamp key.
    """
    rows = []

    for dataset_name, frame in datasets.items():
        has_key = MERGE_KEY in frame.columns

        add_check(
            rows,
            f"{dataset_name}__timestamp_column_exists",
            has_key,
            has_key,
            True,
        )

        if not has_key:
            continue

        duplicate_timestamps = int(
            frame[MERGE_KEY]
            .duplicated()
            .sum()
        )

        add_check(
            rows,
            f"{dataset_name}__unique_timestamps",
            duplicate_timestamps == 0,
            duplicate_timestamps,
            0,
        )

        add_check(
            rows,
            f"{dataset_name}__nonempty",
            len(frame) > 0,
            len(frame),
            ">0",
        )

        timestamp_index = pd.DatetimeIndex(
            frame[MERGE_KEY]
            .sort_values()
        )

        if len(timestamp_index) > 1:
            bad_spacing = int(
                pd.Series(timestamp_index)
                .diff()
                .dropna()
                .ne(pd.Timedelta(hours=1))
                .sum()
            )
        else:
            bad_spacing = 0

        add_check(
            rows,
            f"{dataset_name}__hourly_spacing",
            bad_spacing == 0,
            bad_spacing,
            0,
            severity="warning",
            notes=(
                "Warns when the source itself is not a continuous hourly "
                "table over its own coverage window."
            ),
        )

    return pd.DataFrame(rows)


def merge_master_sources(
    datasets: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Merge all sources into one hourly master table and drop duplicate columns
    only after confirming overlapping values are identical.
    """
    if "calendar" not in datasets:
        raise KeyError("The canonical 'calendar' dataset is required.")

    source_audit = validate_sources(datasets)

    failed_source_checks = source_audit.loc[
        source_audit["severity"].eq("error")
        & ~source_audit["pass"]
    ]

    if not failed_source_checks.empty:
        raise ValueError("One or more source datasets failed validation.")

    master = (
        datasets["calendar"]
        .sort_values(MERGE_KEY)
        .copy()
    )

    reconciled_rows = []

    for dataset_name, frame in datasets.items():
        if dataset_name == "calendar":
            continue

        incoming = (
            frame
            .sort_values(MERGE_KEY)
            .copy()
        )

        overlapping_columns = sorted(
            (
                set(master.columns)
                & set(incoming.columns)
            )
            - {MERGE_KEY}
        )

        columns_to_drop = []

        for column in overlapping_columns:
            comparison = master[[MERGE_KEY, column]].merge(
                incoming[[MERGE_KEY, column]],
                on=MERGE_KEY,
                how="inner",
                suffixes=("_master", "_incoming"),
                validate="one_to_one",
            )

            master_column = f"{column}_master"
            incoming_column = f"{column}_incoming"

            matched = values_match(
                comparison[master_column],
                comparison[incoming_column],
            )

            mismatch_count = int(
                (~matched).sum()
            )

            if mismatch_count > 0:
                raise ValueError(
                    f"Overlapping column {column!r} conflicts between "
                    f"master and {dataset_name!r} for "
                    f"{mismatch_count:,} timestamps."
                )

            columns_to_drop.append(column)

            reconciled_rows.append({
                "incoming_dataset": dataset_name,
                "column": column,
                "overlap_rows_checked": len(comparison),
                "conflicting_rows": mismatch_count,
                "resolution": "dropped_incoming_kept_master",
            })

        incoming = incoming.drop(
            columns=columns_to_drop
        )

        master = master.merge(
            incoming,
            on=MERGE_KEY,
            how="inner",
            validate="one_to_one",
        )

    master = (
        master
        .sort_values(MERGE_KEY)
        .reset_index(drop=True)
    )

    reconciled = pd.DataFrame(
        reconciled_rows,
        columns=[
            "incoming_dataset",
            "column",
            "overlap_rows_checked",
            "conflicting_rows",
            "resolution",
        ],
    )

    return master, reconciled, source_audit


def audit_master_dataset(
    master: pd.DataFrame,
    datasets: dict[str, pd.DataFrame],
    reconciled_columns: pd.DataFrame,
    source_audit: pd.DataFrame,
) -> tuple[pd.DataFrame, bool]:
    """
    Validate the completed master dataset.
    """
    rows = source_audit.to_dict("records")

    add_check(
        rows,
        "master__timestamp_column_exists",
        MERGE_KEY in master.columns,
        MERGE_KEY in master.columns,
        True,
    )

    if MERGE_KEY not in master.columns:
        audit = pd.DataFrame(rows)
        return audit, False

    duplicate_timestamps = int(
        master[MERGE_KEY]
        .duplicated()
        .sum()
    )

    add_check(
        rows,
        "master__unique_timestamps",
        duplicate_timestamps == 0,
        duplicate_timestamps,
        0,
    )

    duplicate_columns = int(
        pd.Index(master.columns)
        .duplicated()
        .sum()
    )

    add_check(
        rows,
        "master__unique_columns",
        duplicate_columns == 0,
        duplicate_columns,
        0,
    )

    suffix_columns = [
        column
        for column in master.columns
        if column.endswith(("_x", "_y"))
    ]

    add_check(
        rows,
        "master__no_merge_suffix_columns",
        len(suffix_columns) == 0,
        len(suffix_columns),
        0,
        notes="No unresolved pandas merge suffixes should remain.",
    )

    timestamp_index = pd.DatetimeIndex(
        master[MERGE_KEY]
        .sort_values()
    )

    expected_hours = len(
        pd.date_range(
            timestamp_index.min(),
            timestamp_index.max(),
            freq="h",
            tz="UTC",
        )
    )

    add_check(
        rows,
        "master__complete_hourly_range",
        len(master) == expected_hours,
        len(master),
        expected_hours,
    )

    add_check(
        rows,
        "master__one_row_per_hour",
        (
            duplicate_timestamps == 0
            and len(master) == expected_hours
        ),
        (
            f"rows={len(master)}, duplicates={duplicate_timestamps}, "
            f"expected_hours={expected_hours}"
        ),
        "unique timestamp for each expected UTC hour",
    )

    calendar_columns = [
        column
        for column in datasets["calendar"].columns
        if column != MERGE_KEY
    ]

    missing_calendar_columns = [
        column
        for column in calendar_columns
        if column not in master.columns
    ]

    add_check(
        rows,
        "master__calendar_columns_retained_once",
        len(missing_calendar_columns) == 0,
        (
            f"{len(calendar_columns) - len(missing_calendar_columns)}"
            f"/{len(calendar_columns)}"
        ),
        f"{len(calendar_columns)}/{len(calendar_columns)}",
        notes=(
            "Calendar is the canonical source for timestamp-derived fields."
        ),
    )

    add_check(
        rows,
        "master__overlapping_columns_reconciled",
        (
            reconciled_columns.empty
            or reconciled_columns["conflicting_rows"].eq(0).all()
        ),
        (
            0
            if reconciled_columns.empty
            else int(reconciled_columns["conflicting_rows"].sum())
        ),
        0,
    )

    source_starts = {
        dataset_name: frame[MERGE_KEY].min()
        for dataset_name, frame in datasets.items()
    }

    source_ends = {
        dataset_name: frame[MERGE_KEY].max()
        for dataset_name, frame in datasets.items()
    }

    expected_start = max(source_starts.values())
    expected_end = min(source_ends.values())

    add_check(
        rows,
        "master__start_matches_common_source_window",
        master[MERGE_KEY].min() == expected_start,
        master[MERGE_KEY].min(),
        expected_start,
    )

    add_check(
        rows,
        "master__end_matches_common_source_window",
        master[MERGE_KEY].max() == expected_end,
        master[MERGE_KEY].max(),
        expected_end,
    )

    add_check(
        rows,
        "master__missing_values_recorded",
        True,
        int(master.isna().sum().sum()),
        "recorded",
        severity="info",
        notes=(
            "Missing values are expected for structural gaps, lag windows, "
            "and weather variables that are undefined in some conditions."
        ),
    )

    audit = pd.DataFrame(rows)

    audit_pass = audit_passes(audit)

    return audit, audit_pass


def build_feature_summary(
    master: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize every master column for missingness and basic distribution.
    """
    rows = []

    for column in master.columns:
        series = master[column]
        row = {
            "feature": column,
            "dtype": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "missing_pct": float(series.isna().mean() * 100),
            "non_missing_count": int(series.notna().sum()),
        }

        if pd.api.types.is_numeric_dtype(series):
            description = series.describe(
                percentiles=[
                    0.01,
                    0.05,
                    0.50,
                    0.95,
                    0.99,
                ]
            )

            row.update({
                "mean": description.get("mean"),
                "std": description.get("std"),
                "min": description.get("min"),
                "p01": description.get("1%"),
                "p05": description.get("5%"),
                "median": description.get("50%"),
                "p95": description.get("95%"),
                "p99": description.get("99%"),
                "max": description.get("max"),
            })

        else:
            row.update({
                "mean": None,
                "std": None,
                "min": None,
                "p01": None,
                "p05": None,
                "median": None,
                "p95": None,
                "p99": None,
                "max": None,
            })

        rows.append(row)

    return pd.DataFrame(rows)


def print_audit_report(
    audit_df: pd.DataFrame,
    feature_summary: pd.DataFrame,
    source_summary: pd.DataFrame,
    reconciled_columns: pd.DataFrame,
    master: pd.DataFrame,
) -> None:
    """
    Print a concise terminal report for the master preprocessing run.
    """
    audit_pass = audit_passes(audit_df)

    failed = audit_df.loc[
        ~audit_df["pass"]
    ]

    print("\n" + "=" * 80)
    print("MASTER DATASET AUDIT")
    print("=" * 80)
    print(f"Overall pass       : {bool(audit_pass)}")
    print(f"Rows               : {len(master):,}")
    print(f"Columns            : {master.shape[1]:,}")
    print(f"Start UTC          : {master[MERGE_KEY].min()}")
    print(f"End UTC            : {master[MERGE_KEY].max()}")
    print(f"Missing values     : {int(master.isna().sum().sum()):,}")
    print(f"Sources merged     : {len(source_summary):,}")
    print(f"Columns reconciled : {len(reconciled_columns):,}")

    print("\nFailed checks:")
    if failed.empty:
        print("  None")
    else:
        for _, row in failed.iterrows():
            print(
                f"  - {row['check']} [{row['severity']}] "
                f"observed={row['observed']} expected={row['expected']}"
            )

    print("\nSource coverage:")
    compact_sources = source_summary[
        [
            "dataset",
            "rows",
            "columns",
            "start",
            "end",
            "duplicate_timestamps",
        ]
    ].copy()
    print(compact_sources.to_string(index=False))

    print("\nMost missing features:")
    compact_features = (
        feature_summary
        .sort_values("missing_count", ascending=False)
        .head(10)
        [
            [
                "feature",
                "missing_count",
                "missing_pct",
                "dtype",
            ]
        ]
        .copy()
    )
    compact_features["missing_pct"] = (
        compact_features["missing_pct"]
        .round(3)
    )
    print(compact_features.to_string(index=False))

    print("=" * 80)


def process_master(
    overwrite: bool = False,
    write_csv: bool = False,
) -> dict:
    """
    Run the full master-dataset preprocessing pipeline.
    """
    start_time = time.perf_counter()

    expected_manifest = build_manifest(
        dataset="master",
        source_paths=list(SOURCE_FILES.values()),
        code_paths=preprocessing_code_paths(Path(__file__)),
        configuration={"merge_key": MERGE_KEY},
    )

    audit_artifacts = [
        AUDIT_FILE,
        FEATURE_SUMMARY_FILE,
        SOURCE_SUMMARY_FILE,
        RECONCILED_COLUMNS_FILE,
    ]
    requested_outputs = [OUTPUT_PARQUET, *audit_artifacts]
    if write_csv:
        requested_outputs.append(OUTPUT_CSV)

    if not overwrite and outputs_are_current(
        requested_outputs,
        expected_manifest,
    ):
        return {
            "dataset": "master",
            "status": "skipped_existing",
            "pass": True,
            "parquet_file": str(OUTPUT_PARQUET),
        }

    try:
        datasets = load_source_datasets()

        (
            master,
            reconciled_columns,
            source_audit,
        ) = merge_master_sources(datasets)

        source_summary = build_source_summary(datasets)
        feature_summary = build_feature_summary(master)

        (
            audit_df,
            audit_pass,
        ) = audit_master_dataset(
            master,
            datasets,
            reconciled_columns,
            source_audit,
        )

        print_audit_report(
            audit_df,
            feature_summary,
            source_summary,
            reconciled_columns,
            master,
        )

        write_audit_artifacts(
            {
                AUDIT_FILE: audit_df,
                FEATURE_SUMMARY_FILE: feature_summary,
                SOURCE_SUMMARY_FILE: source_summary,
                RECONCILED_COLUMNS_FILE: reconciled_columns,
            }
        )

        if not audit_pass:
            return {
                "dataset": "master",
                "status": "audit_failed",
                "pass": False,
                "audit_file": str(AUDIT_FILE),
                "feature_summary_file": str(FEATURE_SUMMARY_FILE),
                "source_summary_file": str(SOURCE_SUMMARY_FILE),
                "reconciled_columns_file": str(RECONCILED_COLUMNS_FILE),
                "processing_seconds": round(
                    time.perf_counter()
                    - start_time,
                    3,
                ),
            }

        csv_file = "not written"
        if write_csv:
            csv_file = str(OUTPUT_CSV)

        write_tabular_outputs(
            master,
            parquet_path=OUTPUT_PARQUET,
            csv_path=OUTPUT_CSV if write_csv else None,
            manifest=expected_manifest,
            provenance_artifacts=audit_artifacts,
        )

        return {
            "dataset": "master",
            "status": "saved",
            "pass": True,
            "rows": len(master),
            "columns": master.shape[1],
            "start": str(master[MERGE_KEY].min()),
            "end": str(master[MERGE_KEY].max()),
            "missing_values": int(master.isna().sum().sum()),
            "reconciled_columns": len(reconciled_columns),
            "parquet_file": str(OUTPUT_PARQUET),
            "csv_file": csv_file,
            "audit_file": str(AUDIT_FILE),
            "feature_summary_file": str(FEATURE_SUMMARY_FILE),
            "source_summary_file": str(SOURCE_SUMMARY_FILE),
            "reconciled_columns_file": str(RECONCILED_COLUMNS_FILE),
            "processing_seconds": round(
                time.perf_counter()
                - start_time,
                3,
            ),
        }

    except Exception as exc:
        return {
            "dataset": "master",
            "status": "error",
            "pass": False,
            "error": repr(exc),
            "processing_seconds": round(
                time.perf_counter()
                - start_time,
                3,
            ),
        }


def main() -> None:
    """
    Provide a command-line entry point for the master preprocessing pipeline.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Build one canonical hourly master parquet from preprocessed "
            "and feature-engineered Alberta market datasets."
        )
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Also write a CSV copy of the master table.",
    )

    args = parser.parse_args()

    result = process_master(
        overwrite=args.overwrite,
        write_csv=args.write_csv,
    )

    print("\n" + "=" * 80)
    print("MASTER PREPROCESSING RESULT")
    print("=" * 80)

    for key, value in result.items():
        print(f"{key}: {value}")

    print("=" * 80)


if __name__ == "__main__":
    main()
