"""
================================================================================
PURPOSE:
    Preprocess and audit AESO hourly generation-by-fuel data.

WHY THIS FILE IS USEFUL:
    AESO publishes generation data in a raw, long-format table that mixes
    multiple fuel types and metrics into repeated rows. This file reshapes
    that raw table into one chronological row per UTC hour, with a clean,
    consistent column for every fuel/metric combination. It also runs an
    automated audit to confirm the resulting dataset is trustworthy enough
    to feed into downstream forecasting or trading-logic models.

PIPELINE OVERVIEW:
    RAW AESO generation file (long format, MST timestamps)
        --> load_raw_generation()   reads and normalizes column headers
        --> clean_generation()      pivots into wide, hourly, UTC-indexed features
        --> audit_generation()      validates coverage, schema, ranges, and logic
        --> print_audit_report()    prints a human-readable summary of the audit
        --> process_generation()    orchestrates the steps and writes output files
        --> main()                  exposes the pipeline as a command-line script
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
    GENERATION_CSV,
    GENERATION_PARQUET,
    PROJECT_ROOT,
    PREPROCESSING_AUDITS_DIR,
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

RAW_GENERATION_FILE = PROJECT_ROOT / "data" / "raw" / "Gen Table_Full Data_data.csv"
OUTPUT_CSV = GENERATION_CSV
OUTPUT_PARQUET = GENERATION_PARQUET
AUDIT_DIR = PREPROCESSING_AUDITS_DIR
AUDIT_FILE = AUDIT_DIR / "generation_audit_checks.csv"
SUMMARY_FILE = AUDIT_DIR / "generation_feature_summary.csv"

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

METRIC_MAP = {
    "System Generation": "system_generation",
    "Total Generation": "total_generation",
    "System Available": "system_available",
    "System Capacity": "system_capacity",
    "Maximum Capacity": "maximum_capacity",
}

FUELS = list(FUEL_MAP.values())
METRICS = list(METRIC_MAP.values())

FEATURE_COLUMNS = [
    f"{fuel}_{metric}"
    for metric in METRICS
    for fuel in FUELS
]

TOTAL_COLUMNS = [
    f"total_{metric}"
    for metric in METRICS
]

EXPECTED_COLUMNS = {"timestamp_utc", *FEATURE_COLUMNS, *TOTAL_COLUMNS}

RANGE_EXPECTATIONS = {
    **{f"{fuel}_system_generation": (0, 10000) for fuel in FUELS},
    **{f"{fuel}_total_generation": (0, 10000) for fuel in FUELS},
    **{f"{fuel}_system_available": (0, 10000) for fuel in FUELS},
    **{f"{fuel}_system_capacity": (0, 10000) for fuel in FUELS},
    **{f"{fuel}_maximum_capacity": (0, 10000) for fuel in FUELS},
    "total_system_generation": (0, 30000),
    "total_total_generation": (0, 30000),
    "total_system_available": (0, 30000),
    "total_system_capacity": (0, 30000),
    "total_maximum_capacity": (0, 30000),
}


def load_raw_generation(raw_file: Path = RAW_GENERATION_FILE) -> pd.DataFrame:
    """
    General purpose:
        Read the raw AESO generation CSV from disk and normalize its column
        headers so downstream functions can rely on consistent naming.

    Role in the pipeline:
        This is the ingestion stage — the very first step. It hands off a
        clean, readable raw DataFrame to clean_generation(), which depends
        on stripped/normalized headers to find the columns it needs.
    """
    if not raw_file.exists():
        raise FileNotFoundError(f"Missing raw generation file: {raw_file}")

    df = pd.read_csv(raw_file, encoding="utf-16", sep="\t", low_memory=False)

    df.columns = (
        df.columns
        .str.strip()
        .str.replace("\ufeff", "", regex=False)
    )

    return df


def clean_generation(raw: pd.DataFrame) -> pd.DataFrame:
    """
    General purpose:
        Transform the raw, long-format fuel records into a wide table of
        standardized hourly generation features indexed by UTC timestamp.

    Role in the pipeline:
        This is the core transformation stage. It converts the messy raw
        format into the exact schema (FEATURE_COLUMNS + TOTAL_COLUMNS) that
        audit_generation() expects to validate and that downstream models
        will eventually consume.
    """
    required_raw_columns = {
        "Date - MST",
        "Fuel Type",
        *METRIC_MAP.keys(),
    }

    missing = required_raw_columns - set(raw.columns)
    if missing:
        raise ValueError(f"Missing raw generation columns: {missing}")

    df = raw[["Date - MST", "Fuel Type", *METRIC_MAP.keys()]].copy()

    df = df.rename(columns={
        "Date - MST": "timestamp",
        "Fuel Type": "fuel_type",
    })

    local_ts = pd.to_datetime(
        df["timestamp"],
        format="%m/%d/%Y %I:%M:%S %p",
        errors="coerce",
    )

    df["timestamp_utc"] = (
        local_ts
        .dt.tz_localize("Etc/GMT+7")
        .dt.tz_convert("UTC")
    )

    df["fuel_type"] = df["fuel_type"].astype(str).str.strip()
    df = df[df["fuel_type"].isin(FUEL_MAP)].copy()

    for col in METRIC_MAP:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    long = df.melt(
        id_vars=["timestamp_utc", "fuel_type"],
        value_vars=list(METRIC_MAP.keys()),
        var_name="metric",
        value_name="value",
    )

    long["fuel_clean"] = long["fuel_type"].map(FUEL_MAP)
    long["metric_clean"] = long["metric"].map(METRIC_MAP)
    long["feature"] = long["fuel_clean"] + "_" + long["metric_clean"]

    long, exact_duplicate_rows = deduplicate_or_raise(
        long,
        ["timestamp_utc", "feature"],
        ignore_columns=["fuel_type", "metric", "fuel_clean", "metric_clean"],
        dataset_name="generation",
    )

    wide = (
        long
        .pivot_table(
            index="timestamp_utc",
            columns="feature",
            values="value",
            aggfunc="first",
        )
        .sort_index()
    )

    for col in FEATURE_COLUMNS:
        if col not in wide.columns:
            wide[col] = pd.NA

    wide = wide[FEATURE_COLUMNS]

    for metric in METRICS:
        metric_cols = [f"{fuel}_{metric}" for fuel in FUELS]
        wide[f"total_{metric}"] = wide[metric_cols].sum(axis=1, min_count=1)

    out = wide.reset_index()

    final_columns = ["timestamp_utc"] + FEATURE_COLUMNS + TOTAL_COLUMNS

    out = (
        out[final_columns]
        .dropna(subset=["timestamp_utc"])
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )

    return set_duplicate_stats(
        out,
        exact_duplicate_rows=exact_duplicate_rows,
    )


def audit_generation(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """
    General purpose:
        Run a full suite of validation checks against the cleaned generation
        data and produce descriptive statistics for every feature column.

    Role in the pipeline:
        This is the quality-control gate between cleaning and saving. Its
        boolean pass/fail result determines whether process_generation()
        is allowed to write the cleaned data out as an approved product.
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
        notes="Some missing fuel categories are expected before/after technology entry or retirement.",
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

    for metric in METRICS:
        metric_cols = [f"{fuel}_{metric}" for fuel in FUELS]
        total_col = f"total_{metric}"

        if set(metric_cols).issubset(df.columns) and total_col in df.columns:
            implied_total = df[metric_cols].sum(axis=1, min_count=1)
            diff = (df[total_col] - implied_total).abs()

            add(
                f"domain_{total_col}_matches_sum",
                diff.max(skipna=True) < 1e-6,
                observed=f"max_abs_diff={diff.max(skipna=True):.6g}",
                expected="0",
                severity="error",
            )

    if {"total_system_generation", "total_system_capacity"}.issubset(df.columns):
        over_capacity_count = int((df["total_system_generation"] > df["total_system_capacity"] * 1.10).sum())

        add(
            "domain_total_generation_not_materially_above_capacity",
            over_capacity_count == 0,
            observed=over_capacity_count,
            expected=0,
            severity="warning",
        )

    if {"total_system_available", "total_maximum_capacity"}.issubset(df.columns):
        over_max_count = int((df["total_system_available"] > df["total_maximum_capacity"] * 1.10).sum())

        add(
            "domain_total_available_not_materially_above_maximum_capacity",
            over_max_count == 0,
            observed=over_max_count,
            expected=0,
            severity="warning",
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
        Print a formatted, human-readable summary of the audit results and
        feature statistics to the terminal.

    Role in the pipeline:
        This is the presentation stage. It does not alter any data or the
        pass/fail decision — it only surfaces what audit_generation() found
        so a person can quickly review the health of the cleaned dataset.
    """
    audit_pass = audit_passes(audit_df)
    failed = audit_df.loc[~audit_df["pass"]]

    ts = pd.DatetimeIndex(pd.to_datetime(clean["timestamp_utc"], utc=True))
    expected_hours = len(pd.date_range(ts.min(), ts.max(), freq="h", tz="UTC"))

    print("\n" + "=" * 80)
    print("GENERATION BY FUEL AUDIT")
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


def process_generation(overwrite: bool = False) -> dict:
    """
    General purpose:
        Run the full generation pipeline end-to-end: load, clean, audit,
        report, and conditionally save the cleaned data and audit evidence.

    Role in the pipeline:
        This is the top-level orchestrator that main() calls. It decides
        whether to skip work (if outputs already exist), and only writes
        the final cleaned files if the audit's error-level checks pass.
    """
    start_time = time.perf_counter()

    expected_manifest = build_manifest(
        dataset="generation_by_fuel",
        source_paths=[RAW_GENERATION_FILE],
        code_paths=preprocessing_code_paths(Path(__file__)),
    )

    if not overwrite and outputs_are_current(
        [OUTPUT_CSV, OUTPUT_PARQUET, AUDIT_FILE, SUMMARY_FILE],
        expected_manifest,
    ):
        return {
            "dataset": "generation_by_fuel",
            "status": "skipped_existing",
            "pass": True,
            "csv_file": str(OUTPUT_CSV),
            "parquet_file": str(OUTPUT_PARQUET),
        }

    try:
        raw = load_raw_generation()
        clean = clean_generation(raw)
        audit_df, summary_df, audit_pass = audit_generation(clean)
        print_audit_report(audit_df, summary_df, clean)

        write_audit_artifacts(
            {AUDIT_FILE: audit_df, SUMMARY_FILE: summary_df}
        )

        if not audit_pass:
            return {
                "dataset": "generation_by_fuel",
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
            "dataset": "generation_by_fuel",
            "status": "saved",
            "pass": True,
            "rows": len(clean),
            "features": len(clean.columns) - 1,
            "start": str(clean["timestamp_utc"].min()),
            "end": str(clean["timestamp_utc"].max()),
            "raw_file": str(RAW_GENERATION_FILE),
            "raw_file_size_mb": round(RAW_GENERATION_FILE.stat().st_size / 1024**2, 3),
            "csv_file": str(OUTPUT_CSV),
            "parquet_file": str(OUTPUT_PARQUET),
            "audit_file": str(AUDIT_FILE),
            "summary_file": str(SUMMARY_FILE),
            "processing_seconds": round(time.perf_counter() - start_time, 3),
        }

    except DuplicateConflictError as exc:
        write_audit_artifacts({AUDIT_FILE: duplicate_failure_audit(exc)})
        return {
            "dataset": "generation_by_fuel",
            "status": "audit_failed",
            "pass": False,
            "error": str(exc),
            "audit_file": str(AUDIT_FILE),
            "processing_seconds": round(time.perf_counter() - start_time, 3),
        }

    except Exception as e:
        return {
            "dataset": "generation_by_fuel",
            "status": "error",
            "pass": False,
            "error": repr(e),
            "processing_seconds": round(time.perf_counter() - start_time, 3),
        }


def main() -> None:
    """
    General purpose:
        Provide a command-line entry point for running this file's
        preprocessing pipeline directly from a terminal.

    Role in the pipeline:
        This is the outermost wrapper. It parses the --overwrite flag,
        calls process_generation(), and prints the resulting status
        dictionary so a user running this script sees what happened.
    """
    parser = argparse.ArgumentParser(description="Preprocess AESO generation by fuel data.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    result = process_generation(overwrite=args.overwrite)

    print("\n" + "=" * 80)
    print("GENERATION BY FUEL PREPROCESSING RESULT")
    print("=" * 80)

    for key, value in result.items():
        print(f"{key}: {value}")

    print("=" * 80)


if __name__ == "__main__":
    main()
