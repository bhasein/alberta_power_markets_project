"""
================================================================================
PURPOSE:
    Preprocess and audit AESO hourly generation-outage data.

WHY THIS FILE IS USEFUL:
    AESO publishes hourly unavailable generation capacity by fuel type in a
    raw table using fixed-MST timestamps. This file standardizes those records
    into one chronological UTC row per hour, with one outage-capacity feature
    for each fuel category and a system-wide total outage measure. It also
    audits the resulting dataset for timeline continuity, schema consistency,
    missing values, plausible ranges, non-negative outage values, and agreement
    between fuel-specific outages and the calculated total.

PIPELINE OVERVIEW:
    raw AESO generation-outage file
        --> load_raw_outages()   reads the source table
        --> clean_outages()      standardizes timestamps and outage columns
        --> audit_outages()      validates data quality and creates statistics
        --> print_audit_report() displays the audit results
        --> process_outages()    writes audit and approved data products
        --> main()               exposes the pipeline through the CLI
================================================================================
"""

from pathlib import Path
import argparse
from functools import partial
import sys
import time

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    OUTAGES_CSV,
    OUTAGES_PARQUET,
    PROJECT_ROOT,
    PREPROCESSING_AUDITS_DIR as AUDIT_DIR,
)
from preprocessing.shared import (
    DuplicateConflictError,
    add_check,
    add_duplicate_checks,
    audit_passes,
    build_manifest,
    deduplicate_or_raise,
    duplicate_failure_audit,
    outputs_are_current,
    preprocessing_code_paths,
    set_duplicate_stats,
    write_audit_artifacts,
    write_tabular_outputs,
)

RAW_OUTAGE_FILE = PROJECT_ROOT / "data/raw/Outage Chart_Full Data_data.csv"

OUTPUT_CSV = OUTAGES_CSV
OUTPUT_PARQUET = OUTAGES_PARQUET

AUDIT_FILE = AUDIT_DIR / "outages_audit_checks.csv"
SUMMARY_FILE = AUDIT_DIR / "outages_feature_summary.csv"

FUEL_MAP = {
    "Coal": "coal",
    "Cogeneration": "cogeneration",
    "Combined Cycle": "combined_cycle",
    "Dual Fuel": "dual_fuel",
    "Gas Fired Steam": "gas_fired_steam",
    "Hydro": "hydro",
    "Other": "other",
    "Simple Cycle": "simple_cycle",
    "Solar": "solar",
    "Storage": "storage",
    "Wind": "wind",
}

EXPECTED_COLUMNS = {
    "timestamp_utc",
    "coal_outage",
    "cogeneration_outage",
    "combined_cycle_outage",
    "dual_fuel_outage",
    "gas_fired_steam_outage",
    "hydro_outage",
    "other_outage",
    "simple_cycle_outage",
    "solar_outage",
    "storage_outage",
    "wind_outage",
    "total_outage",
}

OUTAGE_COLUMNS = [
    "coal_outage",
    "cogeneration_outage",
    "combined_cycle_outage",
    "dual_fuel_outage",
    "gas_fired_steam_outage",
    "hydro_outage",
    "other_outage",
    "simple_cycle_outage",
    "solar_outage",
    "storage_outage",
    "wind_outage",
]

RANGE_EXPECTATIONS = {
    "coal_outage": (0, 10000),
    "cogeneration_outage": (0, 10000),
    "combined_cycle_outage": (0, 10000),
    "dual_fuel_outage": (0, 5000),
    "gas_fired_steam_outage": (0, 10000),
    "hydro_outage": (0, 5000),
    "other_outage": (0, 5000),
    "simple_cycle_outage": (0, 10000),
    "solar_outage": (0, 5000),
    "storage_outage": (0, 1000),
    "wind_outage": (0, 10000),
    "total_outage": (0, 30000),
}


def load_raw_outages(
    raw_file: Path = RAW_OUTAGE_FILE,
) -> pd.DataFrame:
    """
    General purpose:
        Read the raw AESO generation-outage table from disk using the
        encoding and delimiter used by the source file.

    Role in the pipeline:
        This is the ingestion stage. It verifies that the configured source
        file exists and returns the raw DataFrame for validation and
        transformation in clean_outages().
    """
    if not raw_file.exists():
        raise FileNotFoundError(f"Missing raw outage file: {raw_file}")

    return pd.read_csv(raw_file, encoding="utf-16", sep="\t", low_memory=False)


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str:
    """
    General purpose:
        Search a DataFrame for the first matching column name from an ordered
        list of candidates while ignoring capitalization and surrounding
        whitespace.

    Role in the pipeline:
        This utility supports source files whose timestamp or feature headers
        may use slightly different labels. It returns the original DataFrame
        column name so the caller can select that column directly.
    """
    lower_map = {
        c.lower().strip(): c
        for c in df.columns
    }

    for candidate in candidates:
        key = candidate.lower().strip()
        if key in lower_map:
            return lower_map[key]

    raise ValueError(f"Could not find any of these columns: {candidates}")


def clean_outages(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    General purpose:
        Standardize the raw outage table by normalizing column headers,
        validating required fields, converting fixed-MST timestamps to UTC,
        converting fuel values to numeric outage features, and calculating
        total hourly outage capacity.

    Role in the pipeline:
        This is the transformation stage. It converts the raw AESO source
        into one chronological UTC record per hour using the exact schema
        expected by audit_outages() and downstream market models.
    """

    raw = raw.copy()

    raw.columns = (
        raw.columns
        .str.strip()
        .str.replace("\ufeff", "", regex=False)
    )

    timestamp_col = "Date - MST"

    required_cols = {timestamp_col} | set(FUEL_MAP.keys())
    missing = required_cols - set(raw.columns)

    if missing:
        raise ValueError(f"Missing raw outage columns: {missing}")

    df = raw[[timestamp_col] + list(FUEL_MAP.keys())].copy()

    local_ts = pd.to_datetime(
        df[timestamp_col],
        format="%m/%d/%Y %I:%M:%S %p",
        errors="coerce",
    )

    out = pd.DataFrame()

    out["timestamp_utc"] = (
        local_ts
        .dt.tz_localize("Etc/GMT+7")
        .dt.tz_convert("UTC")
    )

    for raw_col, fuel_clean in FUEL_MAP.items():
        out[f"{fuel_clean}_outage"] = pd.to_numeric(df[raw_col], errors="coerce")

    out["total_outage"] = out[OUTAGE_COLUMNS].sum(axis=1, min_count=1)

    out = out.dropna(subset=["timestamp_utc"])
    out, exact_duplicate_rows = deduplicate_or_raise(
        out,
        ["timestamp_utc"],
        dataset_name="outages",
    )
    out = out.sort_values("timestamp_utc").reset_index(drop=True)

    final_columns = ["timestamp_utc"] + OUTAGE_COLUMNS + ["total_outage"]

    return set_duplicate_stats(
        out[final_columns],
        exact_duplicate_rows=exact_duplicate_rows,
    )


def audit_outages(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    bool,
]:
    """
    General purpose:
        Run a complete validation suite against the cleaned outage dataset
        and produce descriptive statistics for every outage feature.

    Role in the pipeline:
        This is the quality-control gate between cleaning and saving. Its
        boolean pass/fail result determines whether process_outages() may
        write the cleaned dataset as an approved preprocessing product.
    """

    rows = []
    add = partial(add_check, rows)
    add_duplicate_checks(rows, df)

    add("timestamp_column_exists", "timestamp_utc" in df.columns, "timestamp_utc" in df.columns, True)

    if "timestamp_utc" not in df.columns:
        audit_df = pd.DataFrame(rows)
        return audit_df, pd.DataFrame(), False

    ts = pd.DatetimeIndex(pd.to_datetime(df["timestamp_utc"], utc=True))
    feature_cols = [c for c in df.columns if c != "timestamp_utc"]

    observed_start = ts.min()
    observed_end = ts.max()
    expected_index = pd.date_range(observed_start, observed_end, freq="h", tz="UTC")
    expected_hours = len(expected_index)

    add("row_count_positive", len(df) > 0, len(df), "> 0")
    add("observed_period_start", True, str(observed_start), "recorded")
    add("observed_period_end", True, str(observed_end), "recorded")
    add("observed_hours", True, len(df), "recorded")

    add("expected_hours_from_start_to_end", len(df) == expected_hours, len(df), expected_hours)

    add("timestamps_monotonic", ts.is_monotonic_increasing, ts.is_monotonic_increasing, True)
    add("duplicate_timestamps", not ts.has_duplicates, int(ts.duplicated().sum()), 0)

    if len(ts) > 1:
        diffs = pd.Series(ts).diff().dropna()
        bad_diffs = diffs[diffs != pd.Timedelta(hours=1)]
        missing_hours = expected_index.difference(ts)
        extra_hours = ts.difference(expected_index)

        add("hourly_spacing", len(bad_diffs) == 0, len(bad_diffs), 0)
        add("missing_hours", len(missing_hours) == 0, len(missing_hours), 0)
        add("extra_hours", len(extra_hours) == 0, len(extra_hours), 0)

    actual_cols = set(df.columns)
    missing_cols = EXPECTED_COLUMNS - actual_cols
    extra_cols = actual_cols - EXPECTED_COLUMNS

    add(
        "expected_columns_present",
        len(missing_cols) == 0,
        "; ".join(sorted(missing_cols)),
        "no missing expected columns",
    )

    add(
        "no_unexpected_extra_columns",
        len(extra_cols) == 0,
        "; ".join(sorted(extra_cols)),
        "no extra columns",
        severity="warning",
    )

    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    add("all_features_numeric", len(non_numeric) == 0, "; ".join(non_numeric), "all numeric")

    all_null = [c for c in feature_cols if df[c].isna().all()]
    any_null = [c for c in feature_cols if df[c].isna().any()]

    add("no_all_null_feature_columns", len(all_null) == 0, "; ".join(all_null), "")
    add(
        "no_partial_null_feature_columns",
        len(any_null) == 0,
        "; ".join(any_null),
        "",
        severity="warning",
        notes="Some missing outage categories may be expected before a technology enters service.",
    )

    for col, (low, high) in RANGE_EXPECTATIONS.items():
        if col in df.columns:
            col_min = df[col].min(skipna=True)
            col_max = df[col].max(skipna=True)

            add(
                f"range_check__{col}",
                (col_min >= low) and (col_max <= high),
                observed=f"min={col_min:.6g}, max={col_max:.6g}",
                expected=f"[{low}, {high}]",
                severity="warning",
            )

    for col in feature_cols:
        if col in df.columns:
            negative_count = int((df[col] < 0).sum())
            zero_count = int((df[col] == 0).sum())
            missing_count = int(df[col].isna().sum())

            add(
                f"domain_non_negative__{col}",
                negative_count == 0,
                observed=negative_count,
                expected=0,
                severity="warning",
            )

            add(
                f"domain_zero_hours__{col}",
                True,
                observed=zero_count,
                expected="recorded",
                severity="info",
            )

            add(
                f"domain_missing_count__{col}",
                True,
                observed=missing_count,
                expected="recorded",
                severity="info",
            )

            add(
                f"domain_peak_recorded__{col}",
                True,
                observed=f"max={df[col].max(skipna=True):.6g}",
                expected="recorded",
                severity="info",
            )

    if set(OUTAGE_COLUMNS).issubset(df.columns):
        implied_total = df[OUTAGE_COLUMNS].sum(axis=1, min_count=1)
        diff = (df["total_outage"] - implied_total).abs()

        add(
            "domain_total_outage_matches_sum",
            diff.max(skipna=True) < 1e-6,
            observed=f"max_abs_diff={diff.max(skipna=True):.6g}",
            expected="0",
            severity="error",
        )

    summary = (
        df[feature_cols]
        .describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99])
        .T
        .reset_index()
    )

    summary = summary.rename(columns={
        "index": "feature",
        "1%": "p01",
        "50%": "median",
        "99%": "p99",
    })

    summary["missing_count"] = df[feature_cols].isna().sum().values
    summary["missing_pct"] = (summary["missing_count"] / len(df)) * 100
    summary["dtype"] = [str(df[c].dtype) for c in feature_cols]

    audit_df = pd.DataFrame(rows)
    audit_pass = audit_passes(audit_df)

    return audit_df, summary, bool(audit_pass)


def print_audit_report(
    audit_df: pd.DataFrame,
    feature_summary: pd.DataFrame,
    clean: pd.DataFrame,
) -> None:
    """
    General purpose:
        Print a formatted, human-readable summary of the outage audit
        results and feature statistics to the terminal.

    Role in the pipeline:
        This is the presentation stage. It does not modify the cleaned data
        or the pass/fail result; it exposes the findings from audit_outages()
        for manual review.
    """
    audit_pass = audit_passes(audit_df)
    failed = audit_df.loc[~audit_df["pass"]]

    ts = pd.DatetimeIndex(pd.to_datetime(clean["timestamp_utc"], utc=True))
    expected_hours = len(pd.date_range(ts.min(), ts.max(), freq="h", tz="UTC"))

    print("\n" + "=" * 80)
    print("OUTAGES AUDIT")
    print("=" * 80)
    print(f"Overall pass  : {audit_pass}")
    print(f"Rows          : {len(clean):,}")
    print(f"Features      : {len(clean.columns) - 1}")
    print(f"Start UTC     : {ts.min()}")
    print(f"End UTC       : {ts.max()}")
    print(f"Observed hrs  : {len(clean):,}")
    print(f"Expected hrs  : {expected_hours:,}")
    print(f"Coverage      : {len(clean) / expected_hours:.2%}")

    print("\nFailed checks:")
    if failed.empty:
        print("  None")
    else:
        for _, row in failed.iterrows():
            print(
                f"  - {row['check']} [{row['severity']}] "
                f"observed={row['observed']} expected={row['expected']}"
            )

    print("\nAudit checklist:")
    for _, row in audit_df.iterrows():
        icon = "✓" if row["pass"] else "✗"
        sev = row["severity"]
        check = row["check"]
        print(f"  {icon} {check} ({sev})")

    print("\nFeature statistics:")
    compact = feature_summary[
        ["feature", "missing_count", "mean", "median", "p01", "p99", "min", "max"]
    ].copy()

    compact = compact.rename(columns={"missing_count": "missing"})

    for col in ["mean", "median", "p01", "p99", "min", "max"]:
        compact[col] = compact[col].round(2)

    print(compact.to_string(index=False))

    print("=" * 80)


def process_outages(
    overwrite: bool = False,
) -> dict:
    """
    General purpose:
        Run the complete outage preprocessing pipeline: load the raw table,
        clean it, audit it, print the audit report, and conditionally save
        the approved dataset and audit evidence.

    Role in the pipeline:
        This is the top-level orchestrator called by main(). It reuses
        existing outputs unless overwrite is requested and only saves the
        cleaned products after every error-level audit check passes.
    """

    start_time = time.perf_counter()

    expected_manifest = build_manifest(
        dataset="outages",
        source_paths=[RAW_OUTAGE_FILE],
        code_paths=preprocessing_code_paths(Path(__file__)),
    )

    if not overwrite and outputs_are_current(
        [OUTPUT_CSV, OUTPUT_PARQUET, AUDIT_FILE, SUMMARY_FILE],
        expected_manifest,
    ):
        return {
            "dataset": "outages",
            "status": "skipped_existing",
            "pass": True,
            "csv_file": str(OUTPUT_CSV),
            "parquet_file": str(OUTPUT_PARQUET),
        }

    try:
        raw = load_raw_outages()
        clean = clean_outages(raw)
        audit_df, summary_df, audit_pass = audit_outages(clean)
        print_audit_report(audit_df, summary_df, clean)

        write_audit_artifacts(
            {AUDIT_FILE: audit_df, SUMMARY_FILE: summary_df}
        )

        if not audit_pass:
            return {
                "dataset": "outages",
                "status": "audit_failed",
                "pass": False,
                "audit_file": str(AUDIT_FILE),
                "summary_file": str(SUMMARY_FILE),
                "processing_seconds": round(time.perf_counter() - start_time, 3),
            }

        write_tabular_outputs(
            clean,
            parquet_path=OUTPUT_PARQUET,
            csv_path=OUTPUT_CSV,
            manifest=expected_manifest,
            provenance_artifacts=[AUDIT_FILE, SUMMARY_FILE],
        )

        return {
            "dataset": "outages",
            "status": "saved",
            "pass": True,
            "rows": len(clean),
            "features": len(clean.columns) - 1,
            "start": str(clean["timestamp_utc"].min()),
            "end": str(clean["timestamp_utc"].max()),
            "raw_file": str(RAW_OUTAGE_FILE),
            "raw_file_size_mb": round(RAW_OUTAGE_FILE.stat().st_size / 1024**2, 3),
            "csv_file": str(OUTPUT_CSV),
            "parquet_file": str(OUTPUT_PARQUET),
            "audit_file": str(AUDIT_FILE),
            "summary_file": str(SUMMARY_FILE),
            "processing_seconds": round(time.perf_counter() - start_time, 3),
        }

    except DuplicateConflictError as exc:
        write_audit_artifacts({AUDIT_FILE: duplicate_failure_audit(exc)})
        return {
            "dataset": "outages",
            "status": "audit_failed",
            "pass": False,
            "error": str(exc),
            "audit_file": str(AUDIT_FILE),
            "processing_seconds": round(time.perf_counter() - start_time, 3),
        }

    except Exception as e:
        return {
            "dataset": "outages",
            "status": "error",
            "pass": False,
            "error": repr(e),
            "processing_seconds": round(time.perf_counter() - start_time, 3),
        }


def main() -> None:
    """
    General purpose:
        Provide a command-line entry point for running the outage
        preprocessing pipeline directly from a terminal.

    Role in the pipeline:
        This is the outermost wrapper. It parses the --overwrite option,
        calls process_outages(), and prints the returned status dictionary
        so the user can see what the pipeline did.
    """
    parser = argparse.ArgumentParser(description="Preprocess AESO outage data.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    result = process_outages(overwrite=args.overwrite)

    print("\n" + "=" * 80)
    print("OUTAGES PREPROCESSING RESULT")
    print("=" * 80)

    for key, value in result.items():
        print(f"{key}: {value}")

    print("=" * 80)


if __name__ == "__main__":
    main()
