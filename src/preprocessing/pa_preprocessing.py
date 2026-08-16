"""
================================================================================
PURPOSE:
    Preprocess and audit AESO hourly price-and-demand data.

WHY THIS FILE IS USEFUL:
    AESO publishes its historical Price and AIL table with hourly demand,
    pool price, natural-gas price, and spark-spread measurements using
    fixed-MST timestamps. This file standardizes those records into a stable
    UTC-indexed dataset for downstream market analysis. It also audits the
    cleaned data for timeline continuity, schema consistency, missing values,
    plausible ranges, important price events, and broad consistency between
    the reported spark spread, pool price, and gas price.

PIPELINE OVERVIEW:
    raw AESO P&A file
        --> load_raw_pa()       reads the source table
        --> clean_pa()          standardizes timestamps and numeric features
        --> audit_pa()          validates coverage, schema, and plausible values
        --> print_audit_report() presents the checks for review
        --> process_pa()        writes audit and approved data products
        --> main()              exposes the pipeline through the CLI
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
    PA_TABLE_CSV,
    PA_TABLE_PARQUET,
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

RAW_PA_FILE = PROJECT_ROOT / "data" / "raw" / "P&A Table_Full Data_data.csv"

OUTPUT_CSV = PA_TABLE_CSV
OUTPUT_PARQUET = PA_TABLE_PARQUET

AUDIT_FILE = AUDIT_DIR / "pa_audit_checks.csv"
SUMMARY_FILE = AUDIT_DIR / "pa_feature_summary.csv"


EXPECTED_COLUMNS = {
    "timestamp_utc",
    "ail_mw",
    "gas_price_cad_gj",
    "pool_price_cad_mwh",
    "spark_spread",
}


RANGE_EXPECTATIONS = {
    "ail_mw": (0, 20000),
    "gas_price_cad_gj": (-20, 100),
    "pool_price_cad_mwh": (-100, 2000),
    "spark_spread": (-500, 2000),
}


def load_raw_pa(
    raw_file: Path = RAW_PA_FILE,
) -> pd.DataFrame:
    """
    General purpose:
        Read the raw AESO price-and-demand table from disk using the
        encoding and delimiter used by the source file.

    Role in the pipeline:
        This is the ingestion stage. It verifies that the configured source
        file exists and returns the raw DataFrame for validation and
        transformation in clean_pa().
    """
    if not raw_file.exists():
        raise FileNotFoundError(f"Missing raw P&A file: {raw_file}")

    return pd.read_csv(raw_file, encoding="utf-16", sep="\t")


def clean_pa(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    General purpose:
        Standardize the raw P&A table by validating required source fields,
        converting fixed-MST timestamps to UTC, and converting demand,
        prices, and spark spread to numeric features.

    Role in the pipeline:
        This is the transformation stage. It converts the AESO source table
        into one chronological UTC record per hour using the exact schema
        expected by audit_pa() and downstream market models.
    """

    required_raw_columns = {
        "Date (MST)",
        "AIL",
        "Gas Price",
        "Price",
        "Spark Spread",
    }

    missing = required_raw_columns - set(raw.columns)
    if missing:
        raise ValueError(f"Missing raw P&A columns: {missing}")

    df = raw.copy()

    out = pd.DataFrame()

    local_ts = pd.to_datetime(df["Date (MST)"], errors="coerce")

    out["timestamp_utc"] = (
        local_ts
        .dt.tz_localize("Etc/GMT+7")
        .dt.tz_convert("UTC")
    )

    out["ail_mw"] = pd.to_numeric(df["AIL"], errors="coerce")
    out["gas_price_cad_gj"] = pd.to_numeric(df["Gas Price"], errors="coerce")
    out["pool_price_cad_mwh"] = pd.to_numeric(df["Price"], errors="coerce")
    out["spark_spread"] = pd.to_numeric(df["Spark Spread"], errors="coerce")

    out = out.dropna(subset=["timestamp_utc"])
    out, exact_duplicate_rows = deduplicate_or_raise(
        out,
        ["timestamp_utc"],
        dataset_name="P&A",
    )
    out = out.sort_values("timestamp_utc").reset_index(drop=True)
    out = out.loc[:, ~out.columns.duplicated()]

    return set_duplicate_stats(
        out,
        exact_duplicate_rows=exact_duplicate_rows,
    )


def audit_pa(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    bool,
]:
    """
    General purpose:
        Run a complete validation suite against the cleaned P&A dataset and
        produce descriptive statistics for every market feature.

    Role in the pipeline:
        This is the quality-control gate between cleaning and saving. Its
        boolean pass/fail result determines whether process_pa() may write
        the cleaned dataset as an approved preprocessing product.
    """

    rows = []
    add = partial(add_check, rows)
    add_duplicate_checks(rows, df)

    add(
        "timestamp_column_exists",
        "timestamp_utc" in df.columns,
        "timestamp_utc" in df.columns,
        True,
    )

    if "timestamp_utc" not in df.columns:
        audit_df = pd.DataFrame(rows)
        return audit_df, pd.DataFrame(), False

    ts = pd.DatetimeIndex(
        pd.to_datetime(
            df["timestamp_utc"],
            utc=True,
        )
    )

    feature_cols = [
        c
        for c in df.columns
        if c != "timestamp_utc"
    ]

    observed_start = ts.min()
    observed_end = ts.max()

    expected_index = pd.date_range(
        observed_start,
        observed_end,
        freq="h",
        tz="UTC",
    )

    expected_hours = len(expected_index)

    add(
        "row_count_positive",
        len(df) > 0,
        len(df),
        "> 0",
    )
    add(
        "observed_period_start",
        True,
        str(observed_start),
        "recorded",
    )
    add(
        "observed_period_end",
        True,
        str(observed_end),
        "recorded",
    )
    add(
        "observed_hours",
        True,
        len(df),
        "recorded",
    )
    add(
        "expected_hours_from_start_to_end",
        len(df) == expected_hours,
        len(df),
        expected_hours,
    )

    add(
        "timestamps_monotonic",
        ts.is_monotonic_increasing,
        ts.is_monotonic_increasing,
        True,
    )

    add(
        "duplicate_timestamps",
        not ts.has_duplicates,
        int(ts.duplicated().sum()),
        0,
    )

    if len(ts) > 1:
        diffs = pd.Series(ts).diff().dropna()

        bad_diffs = diffs[
            diffs != pd.Timedelta(hours=1)
        ]

        missing_hours = expected_index.difference(ts)

        extra_hours = ts.difference(expected_index)

        add(
            "hourly_spacing",
            len(bad_diffs) == 0,
            len(bad_diffs),
            0,
        )
        add(
            "missing_hours",
            len(missing_hours) == 0,
            len(missing_hours),
            0,
        )
        add(
            "extra_hours",
            len(extra_hours) == 0,
            len(extra_hours),
            0,
        )

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

    non_numeric = [
        c
        for c in feature_cols
        if not pd.api.types.is_numeric_dtype(df[c])
    ]

    add(
        "all_features_numeric",
        len(non_numeric) == 0,
        "; ".join(non_numeric),
        "all numeric",
    )

    all_null = [
        c
        for c in feature_cols
        if df[c].isna().all()
    ]

    any_null = [
        c
        for c in feature_cols
        if df[c].isna().any()
    ]

    add(
        "no_all_null_feature_columns",
        len(all_null) == 0,
        "; ".join(all_null),
        "",
    )

    add(
        "no_partial_null_feature_columns",
        len(any_null) == 0,
        "; ".join(any_null),
        "",
        severity="warning",
    )

    for col, (low, high) in RANGE_EXPECTATIONS.items():
        if col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()

            add(
                f"range_check__{col}",
                (col_min >= low) and (col_max <= high),
                observed=f"min={col_min:.6g}, max={col_max:.6g}",
                expected=f"[{low}, {high}]",
                severity="warning",
            )

    if "pool_price_cad_mwh" in df.columns:
        spike_250 = int(
            (df["pool_price_cad_mwh"] >= 250).sum()
        )
        spike_500 = int(
            (df["pool_price_cad_mwh"] >= 500).sum()
        )
        spike_900 = int(
            (df["pool_price_cad_mwh"] >= 900).sum()
        )

        add(
            "domain_pool_price_spikes_250_plus",
            True,
            spike_250,
            "recorded",
            severity="info",
        )
        add(
            "domain_pool_price_spikes_500_plus",
            True,
            spike_500,
            "recorded",
            severity="info",
        )
        add(
            "domain_pool_price_spikes_900_plus",
            True,
            spike_900,
            "recorded",
            severity="info",
        )

    if "pool_price_cad_mwh" in df.columns:
        zero_price_hours = int(
            (df["pool_price_cad_mwh"] == 0).sum()
        )

        add(
            "domain_zero_price_hours",
            True,
            zero_price_hours,
            "recorded",
            severity="info",
        )

    if "gas_price_cad_gj" in df.columns:
        negative_gas_hours = int(
            (df["gas_price_cad_gj"] < 0).sum()
        )

        add(
            "domain_negative_gas_price_hours",
            True,
            negative_gas_hours,
            "recorded",
            severity="info",
        )

    if "ail_mw" in df.columns:
        ail_min = df["ail_mw"].min()
        ail_max = df["ail_mw"].max()

        add(
            "domain_ail_positive",
            ail_min > 0,
            f"min={ail_min:.6g}",
            "> 0",
        )

        add(
            "domain_ail_peak_recorded",
            True,
            f"max={ail_max:.6g}",
            "recorded",
            severity="info",
        )

    if {
        "pool_price_cad_mwh",
        "gas_price_cad_gj",
        "spark_spread",
    }.issubset(df.columns):
        implied_heat_rate = (
            df["pool_price_cad_mwh"]
            - df["spark_spread"]
        ) / df["gas_price_cad_gj"].replace(0, pd.NA)

        implied_heat_rate = implied_heat_rate.dropna()

        median_hr = implied_heat_rate.median()

        add(
            "domain_implied_heat_rate_median",
            5 <= median_hr <= 15,
            observed=f"{median_hr:.3f}",
            expected="roughly 5-15",
            severity="warning",
            notes=(
                "Checks whether spark_spread is broadly consistent with "
                "pool_price - heat_rate * gas_price."
            ),
        )

    summary = (
        df[feature_cols]
        .describe(
            percentiles=[
                0.01,
                0.25,
                0.5,
                0.75,
                0.99,
            ]
        )
        .T
        .reset_index()
    )

    summary = summary.rename(columns={
        "index": "feature",
        "1%": "p01",
        "50%": "median",
        "99%": "p99",
    })

    summary["missing_count"] = (
        df[feature_cols]
        .isna()
        .sum()
        .values
    )

    summary["missing_pct"] = (
        summary["missing_count"] / len(df)
    ) * 100

    summary["dtype"] = [
        str(df[c].dtype)
        for c in feature_cols
    ]

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
        Print a formatted, human-readable summary of the P&A audit results
        and feature statistics to the terminal.

    Role in the pipeline:
        This is the presentation stage. It does not modify the cleaned data
        or the pass/fail result; it exposes the findings from audit_pa() for
        manual review.
    """

    audit_pass = audit_passes(audit_df)
    failed = audit_df.loc[~audit_df["pass"]]

    ts = pd.DatetimeIndex(pd.to_datetime(clean["timestamp_utc"], utc=True))
    expected_hours = len(pd.date_range(ts.min(), ts.max(), freq="h", tz="UTC"))

    print("\n" + "=" * 80)
    print("P&A AUDIT")
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
            print(f"  - {row['check']} [{row['severity']}] observed={row['observed']} expected={row['expected']}")

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

    rename_map = {
        "missing_count": "missing",
        "mean": "mean",
        "median": "median",
        "p01": "p01",
        "p99": "p99",
        "min": "min",
        "max": "max",
    }

    compact = compact.rename(columns=rename_map)

    for col in ["mean", "median", "p01", "p99", "min", "max"]:
        compact[col] = compact[col].round(2)

    print(compact.to_string(index=False))

    print("=" * 80)


def process_pa(
    overwrite: bool = False,
) -> dict:
    """
    General purpose:
        Run the complete P&A preprocessing pipeline: load the raw table,
        clean it, audit it, print the audit report, and conditionally save
        the approved dataset and audit evidence.

    Role in the pipeline:
        This is the top-level orchestrator called by main(). It reuses
        existing outputs unless overwrite is requested and only saves the
        cleaned products after every error-level audit check passes.
    """
    start_time = time.perf_counter()

    expected_manifest = build_manifest(
        dataset="pa",
        source_paths=[RAW_PA_FILE],
        code_paths=preprocessing_code_paths(Path(__file__)),
    )

    if not overwrite and outputs_are_current(
        [OUTPUT_CSV, OUTPUT_PARQUET, AUDIT_FILE, SUMMARY_FILE],
        expected_manifest,
    ):
        return {
            "dataset": "pa",
            "status": "skipped_existing",
            "pass": True,
            "csv_file": str(OUTPUT_CSV),
            "parquet_file": str(OUTPUT_PARQUET),
        }

    try:
        raw = load_raw_pa()
        clean = clean_pa(raw)
        audit_df, summary_df, audit_pass = audit_pa(clean)
        print_audit_report(audit_df, summary_df, clean)

        write_audit_artifacts(
            {AUDIT_FILE: audit_df, SUMMARY_FILE: summary_df}
        )

        if not audit_pass:
            return {
                "dataset": "pa",
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
            "dataset": "pa",
            "status": "saved",
            "pass": True,
            "rows": len(clean),
            "features": len(clean.columns) - 1,
            "start": str(clean["timestamp_utc"].min()),
            "end": str(clean["timestamp_utc"].max()),
            "raw_file": str(RAW_PA_FILE),
            "raw_file_size_mb": round(RAW_PA_FILE.stat().st_size / 1024**2, 3),
            "csv_file": str(OUTPUT_CSV),
            "parquet_file": str(OUTPUT_PARQUET),
            "audit_file": str(AUDIT_FILE),
            "summary_file": str(SUMMARY_FILE),
            "processing_seconds": round(time.perf_counter() - start_time, 3),
        }

    except DuplicateConflictError as exc:
        write_audit_artifacts({AUDIT_FILE: duplicate_failure_audit(exc)})
        return {
            "dataset": "pa",
            "status": "audit_failed",
            "pass": False,
            "error": str(exc),
            "audit_file": str(AUDIT_FILE),
            "processing_seconds": round(time.perf_counter() - start_time, 3),
        }

    except Exception as e:
        return {
            "dataset": "pa",
            "status": "error",
            "pass": False,
            "error": repr(e),
            "processing_seconds": round(time.perf_counter() - start_time, 3),
        }


def main() -> None:
    """
    General purpose:
        Provide a command-line entry point for running the P&A preprocessing
        pipeline directly from a terminal.

    Role in the pipeline:
        This is the outermost wrapper. It parses the --overwrite option,
        calls process_pa(), and prints the returned status dictionary so the
        user can see what the pipeline did.
    """
    parser = argparse.ArgumentParser(description="Preprocess P&A hourly market data.")

    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    result = process_pa(overwrite=args.overwrite)

    print("\n" + "=" * 80)
    print("P&A PREPROCESSING RESULT")
    print("=" * 80)

    for key, value in result.items():
        print(f"{key}: {value}")

    print("=" * 80)


if __name__ == "__main__":
    main()
