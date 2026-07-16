"""
================================================================================
PURPOSE:
    Preprocess and audit AESO hourly intertie-flow and hour-ahead forecast data.

WHY THIS FILE IS USEFUL:
    AESO publishes historical intertie flows and hour-ahead pool-price
    forecasts across multiple source files covering different time periods.
    This file standardizes those files into one continuous UTC-indexed hourly
    dataset containing imports, exports, and the hour-ahead price forecast.
    It also audits the combined series for timeline continuity, schema
    consistency, plausible ranges, missing values, and basic market-domain
    rules before the data is used in forecasting or trading analysis.

PIPELINE OVERVIEW:
    raw AESO intertie files
        --> load_raw_intertie_file() reads each source table
        --> clean_one_intertie_file() standardizes one file
        --> clean_interties()          combines and deduplicates all files
        --> audit_interties()          validates coverage, schema, and values
        --> print_audit_report()       presents the checks for review
        --> process_interties()        writes audit and approved data products
        --> main()                     exposes the pipeline through the CLI
================================================================================
"""

from pathlib import Path
import argparse
import pandas as pd
import time


# Path objects represent filesystem locations, and the / operator appends
# folders or filenames without manually constructing platform-specific strings.
PROJECT_ROOT = Path("/Users/brodiehasein/alberta_power_markets_project")

RAW_DIR = PROJECT_ROOT / "data" / "raw"

RAW_INTERTIE_FILES = [
    RAW_DIR / "Hourly_Metered_Volumes_and_Pool_Price_and_AIL_2010-2019.csv",
    RAW_DIR / "Hourly_Metered_Volumes_and_Pool_Price_and_AIL_2020-Jul2025.csv",
]

PREPROCESSING_DIR = PROJECT_ROOT / "data" / "preprocessing"
AUDIT_DIR = PROJECT_ROOT / "data" / "audits"

OUTPUT_CSV = PREPROCESSING_DIR / "interties_hour_ahead.csv"
OUTPUT_PARQUET = PREPROCESSING_DIR / "interties_hour_ahead.parquet"

AUDIT_FILE = AUDIT_DIR / "interties_audit_checks.csv"
SUMMARY_FILE = AUDIT_DIR / "interties_feature_summary.csv"

EXPECTED_COLUMNS = {
    "timestamp_utc",
    "hour_ahead_price_forecast",
    "export_bc",
    "export_mt",
    "export_sk",
    "import_bc",
    "import_mt",
    "import_sk",
}

RAW_COLUMN_MAP = {
    "HOUR_AHEAD_POOL_PRICE_FORECAST": "hour_ahead_price_forecast",
    "EXPORT_BC": "export_bc",
    "EXPORT_MT": "export_mt",
    "EXPORT_SK": "export_sk",
    "IMPORT_BC": "import_bc",
    "IMPORT_MT": "import_mt",
    "IMPORT_SK": "import_sk",
}

RANGE_EXPECTATIONS = {
    "hour_ahead_price_forecast": (-100, 2000),
    "export_bc": (0, 2000),
    "export_mt": (0, 1000),
    "export_sk": (0, 1500),
    "import_bc": (0, 2000),
    "import_mt": (0, 1000),
    "import_sk": (0, 1500),
}


def load_raw_intertie_file(
    raw_file: Path,
) -> pd.DataFrame:
    """
    General purpose:
        Read one raw AESO intertie and hour-ahead forecast source file from
        disk and return it as a pandas DataFrame.

    Role in the pipeline:
        This is the per-file ingestion stage. It gives clean_interties() a
        single validated way to load each historical source table before
        the files are standardized and combined.
    """
    if not raw_file.exists():
        raise FileNotFoundError(f"Missing raw intertie file: {raw_file}")

    return pd.read_csv(raw_file)


def clean_one_intertie_file(
    raw: pd.DataFrame,
    source_file: Path,
) -> pd.DataFrame:
    """
    General purpose:
        Standardize the timestamp and selected intertie measures from one
        raw source file, convert them to numeric values, and record which
        source file produced each row.

    Role in the pipeline:
        This is the per-file cleaning stage. It converts each historical
        source table into the same schema so clean_interties() can safely
        concatenate files from different time periods.
    """
    raw = raw.copy()

    raw.columns = (
        raw.columns
        .str.strip()
        .str.replace("\ufeff", "", regex=False)
    )

    required_raw_columns = {"Date_Begin_GMT"} | set(RAW_COLUMN_MAP.keys())

    missing = required_raw_columns - set(raw.columns)
    if missing:
        raise ValueError(f"Missing raw columns in {source_file.name}: {missing}")

    out = pd.DataFrame()

    out["timestamp_utc"] = pd.to_datetime(
        raw["Date_Begin_GMT"],
        errors="coerce",
        utc=True,
    )

    for raw_col, clean_col in RAW_COLUMN_MAP.items():
        out[clean_col] = pd.to_numeric(raw[raw_col], errors="coerce")

    out["source_file"] = source_file.name

    return out


def clean_interties(
    raw_files: list[Path] = RAW_INTERTIE_FILES,
) -> pd.DataFrame:
    """
    General purpose:
        Load, clean, combine, and deduplicate all configured intertie and
        hour-ahead forecast source files into one chronological UTC dataset.

    Role in the pipeline:
        This is the multi-file combination stage. It produces the unified
        hourly table consumed by audit_interties(), keeping the last record
        when source files overlap on the same timestamp.
    """
    frames = []

    for raw_file in raw_files:
        raw = load_raw_intertie_file(raw_file)
        clean = clean_one_intertie_file(raw, raw_file)
        frames.append(clean)

    out = pd.concat(frames, ignore_index=True)

    out = out.dropna(subset=["timestamp_utc"])
    out = out.sort_values("timestamp_utc").reset_index(drop=True)

    duplicate_count = int(out["timestamp_utc"].duplicated().sum())

    if duplicate_count > 0:
        out = (
            out
            .drop_duplicates(subset=["timestamp_utc"], keep="last")
            .sort_values("timestamp_utc")
            .reset_index(drop=True)
        )

    out = out.drop(columns=["source_file"])

    return out.loc[:, ~out.columns.duplicated()]


def audit_interties(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    bool,
]:
    """
    General purpose:
        Run a complete validation suite against the combined intertie-flow
        and hour-ahead forecast dataset and summarize every feature.

    Role in the pipeline:
        This is the quality-control gate between cleaning and saving. Its
        boolean pass/fail result determines whether process_interties() may
        write the cleaned dataset as an approved preprocessing product.
    """
    rows = []

    
    def add(
        check,
        passed,
        observed=None,
        expected=None,
        severity="error",
        notes="",
    ):
        """
        General purpose:
            Append one standardized audit result to the shared `rows` list,
            recording the check name, result, severity, observed value,
            expected value, and explanatory notes.

        Role in the pipeline:
            This local helper keeps every validation result in the same
            structure so the enclosing function can assemble one consistent
            audit DataFrame at the end.
        """
        rows.append({
            "check": check,
            "pass": bool(passed),
            "severity": severity,
            "observed": observed,
            "expected": expected,
            "notes": notes,
        })

    add(
        "timestamp_column_exists",
        "timestamp_utc" in df.columns,
        "timestamp_utc" in df.columns,
        True,
    )

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
        notes="Some missing values may be expected, especially MT intertie before its start date or rare missing forecasts.",
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

    
    flow_cols = [
        "export_bc",
        "export_mt",
        "export_sk",
        "import_bc",
        "import_mt",
        "import_sk",
    ]

    for col in flow_cols:
        if col in df.columns:
            negative_count = int((df[col] < 0).sum())
            add(
                f"domain_non_negative__{col}",
                negative_count == 0,
                observed=negative_count,
                expected=0,
                severity="warning",
            )

            add(
                f"domain_max_recorded__{col}",
                True,
                observed=f"max={df[col].max(skipna=True):.6g}",
                expected="recorded",
                severity="info",
            )

            missing_count = int(df[col].isna().sum())
            add(
                f"domain_missing_count__{col}",
                True,
                observed=missing_count,
                expected="recorded",
                severity="info",
            )

    if {"import_bc", "import_mt", "import_sk"}.issubset(df.columns):
        total_imports = df[["import_bc", "import_mt", "import_sk"]].sum(axis=1, min_count=1)
        add(
            "domain_total_import_peak_recorded",
            True,
            observed=f"max={total_imports.max(skipna=True):.6g}",
            expected="recorded",
            severity="info",
        )

    if {"export_bc", "export_mt", "export_sk"}.issubset(df.columns):
        total_exports = df[["export_bc", "export_mt", "export_sk"]].sum(axis=1, min_count=1)
        add(
            "domain_total_export_peak_recorded",
            True,
            observed=f"max={total_exports.max(skipna=True):.6g}",
            expected="recorded",
            severity="info",
        )

    if "hour_ahead_price_forecast" in df.columns:
        missing_forecast = int(df["hour_ahead_price_forecast"].isna().sum())
        add(
            "domain_missing_hour_ahead_forecast",
            missing_forecast == 0,
            observed=missing_forecast,
            expected=0,
            severity="warning",
            notes="Missing hour-ahead forecasts are retained but should be known before modelling.",
        )

        spike_250 = int((df["hour_ahead_price_forecast"] >= 250).sum())
        spike_500 = int((df["hour_ahead_price_forecast"] >= 500).sum())
        spike_900 = int((df["hour_ahead_price_forecast"] >= 900).sum())

        add("domain_hour_ahead_forecast_250_plus", True, spike_250, "recorded", severity="info")
        add("domain_hour_ahead_forecast_500_plus", True, spike_500, "recorded", severity="info")
        add("domain_hour_ahead_forecast_900_plus", True, spike_900, "recorded", severity="info")

    
    summary = df[feature_cols].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T.reset_index()

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
    audit_pass = audit_df.loc[audit_df["severity"].eq("error"), "pass"].all()

    return audit_df, summary, bool(audit_pass)


def print_audit_report(
    audit_df: pd.DataFrame,
    feature_summary: pd.DataFrame,
    clean: pd.DataFrame,
) -> None:
    """
    General purpose:
        Print a formatted, human-readable summary of the intertie audit
        results and feature statistics to the terminal.

    Role in the pipeline:
        This is the presentation stage. It does not modify the cleaned data
        or the pass/fail result; it exposes the findings from
        audit_interties() for manual review.
    """
    audit_pass = audit_df.loc[audit_df["severity"] == "error", "pass"].all()
    failed = audit_df.loc[~audit_df["pass"]]

    ts = pd.DatetimeIndex(pd.to_datetime(clean["timestamp_utc"], utc=True))
    expected_hours = len(pd.date_range(ts.min(), ts.max(), freq="h", tz="UTC"))

    print("\n" + "=" * 80)
    print("INTERTIES / HOUR-AHEAD AUDIT")
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


def process_interties(
    overwrite: bool = False,
) -> dict:
    """
    General purpose:
        Run the complete intertie-flow and hour-ahead forecast pipeline:
        load all source files, clean and combine them, audit the result,
        print the report, and conditionally save approved outputs.

    Role in the pipeline:
        This is the top-level orchestrator called by main(). It reuses
        existing outputs unless overwrite is requested and only saves the
        cleaned products after every error-level audit check passes.
    """
    start_time = time.perf_counter()

    if OUTPUT_CSV.exists() and OUTPUT_PARQUET.exists() and not overwrite:
        return {
            "dataset": "interties_hour_ahead",
            "status": "skipped_existing",
            "pass": True,
            "csv_file": str(OUTPUT_CSV),
            "parquet_file": str(OUTPUT_PARQUET),
        }

    try:
        clean = clean_interties()
        audit_df, summary_df, audit_pass = audit_interties(clean)
        print_audit_report(audit_df, summary_df, clean)

        AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        audit_df.to_csv(AUDIT_FILE, index=False)
        summary_df.to_csv(SUMMARY_FILE, index=False)

        if not audit_pass:
            return {
                "dataset": "interties_hour_ahead",
                "status": "audit_failed",
                "pass": False,
                "audit_file": str(AUDIT_FILE),
                "summary_file": str(SUMMARY_FILE),
                "processing_seconds": round(time.perf_counter() - start_time, 3),
            }

        PREPROCESSING_DIR.mkdir(parents=True, exist_ok=True)

        clean.to_csv(OUTPUT_CSV, index=False)
        clean.to_parquet(OUTPUT_PARQUET, index=False)

        return {
            "dataset": "interties_hour_ahead",
            "status": "saved",
            "pass": True,
            "rows": len(clean),
            "features": len(clean.columns) - 1,
            "start": str(clean["timestamp_utc"].min()),
            "end": str(clean["timestamp_utc"].max()),
            "raw_files": "; ".join(str(p) for p in RAW_INTERTIE_FILES),
            "raw_file_size_mb": round(sum(p.stat().st_size for p in RAW_INTERTIE_FILES if p.exists()) / 1024**2, 3),
            "csv_file": str(OUTPUT_CSV),
            "parquet_file": str(OUTPUT_PARQUET),
            "audit_file": str(AUDIT_FILE),
            "summary_file": str(SUMMARY_FILE),
            "processing_seconds": round(time.perf_counter() - start_time, 3),
        }

    except Exception as e:
        return {
            "dataset": "interties_hour_ahead",
            "status": "error",
            "pass": False,
            "error": repr(e),
            "processing_seconds": round(time.perf_counter() - start_time, 3),
        }


def main() -> None:
    """
    General purpose:
        Provide a command-line entry point for running the intertie-flow and
        hour-ahead forecast preprocessing pipeline directly from a terminal.

    Role in the pipeline:
        This is the outermost wrapper. It parses the --overwrite option,
        calls process_interties(), and prints the returned status dictionary
        so the user can see what the pipeline did.
    """
    parser = argparse.ArgumentParser(description="Preprocess AESO intertie and hour-ahead data.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    result = process_interties(overwrite=args.overwrite)

    print("\n" + "=" * 80)
    print("INTERTIES / HOUR-AHEAD PREPROCESSING RESULT")
    print("=" * 80)

    for key, value in result.items():
        print(f"{key}: {value}")

    print("=" * 80)


# Python assigns __name__ the value "__main__" when this file is executed
# directly. When the file is imported, __name__ contains the module name,
# which prevents main() from running automatically during the import.
if __name__ == "__main__":
    main()