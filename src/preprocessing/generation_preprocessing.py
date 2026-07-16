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
import pandas as pd
import time


# Path objects represent filesystem locations, and the / operator appends
# folders or filenames without manually constructing platform-specific strings.
PROJECT_ROOT = Path("/Users/brodiehasein/alberta_power_markets_project")
RAW_GENERATION_FILE = PROJECT_ROOT / "data" / "raw" / "Gen Table_Full Data_data.csv"
PREPROCESSING_DIR = PROJECT_ROOT / "data" / "preprocessing"
AUDIT_DIR = PROJECT_ROOT / "data" / "audits"
OUTPUT_CSV = PREPROCESSING_DIR / "generation_by_fuel.csv"
OUTPUT_PARQUET = PREPROCESSING_DIR / "generation_by_fuel.parquet"
AUDIT_FILE = AUDIT_DIR / "generation_audit_checks.csv"
SUMMARY_FILE = AUDIT_DIR / "generation_feature_summary.csv"

# This dictionary translates the fuel labels used in the raw AESO file into
# consistent snake_case names suitable for cleaned DataFrame columns.
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

# This mapping standardizes the raw generation metric labels so each metric
# can be incorporated into predictable fuel-and-metric feature names.
METRIC_MAP = {
    "System Generation": "system_generation",
    "Total Generation": "total_generation",
    "System Available": "system_available",
    "System Capacity": "system_capacity",
    "Maximum Capacity": "maximum_capacity",
}

# Extract the standardized mapping values into reusable ordered lists.
# These lists are later used to generate every expected feature column.
FUELS = list(FUEL_MAP.values())
METRICS = list(METRIC_MAP.values())

# The nested comprehension creates every possible fuel-and-metric feature.
# The metric loop is outermost, so columns are grouped by metric and then fuel.
FEATURE_COLUMNS = [
    f"{fuel}_{metric}"
    for metric in METRICS
    for fuel in FUELS
]

# Build one aggregate column name for each generation metric, such as
# total_system_generation and total_system_capacity.
TOTAL_COLUMNS = [
    f"total_{metric}"
    for metric in METRICS
]

# The starred expressions unpack all generated feature and total names into
# one set alongside the timestamp column. A set makes schema comparisons easy.
EXPECTED_COLUMNS = {"timestamp_utc", *FEATURE_COLUMNS, *TOTAL_COLUMNS}

# Each dictionary comprehension assigns the same reasonable range to every
# fuel column belonging to one metric. The ** operators merge those generated
# dictionaries with the separate, wider ranges used for system-wide totals.
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
    # raw_file is expected to be a Path and defaults to the configured AESO
    # source file. The return annotation documents that the function produces
    # a pandas DataFrame containing the raw generation records.
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
    # This set defines the minimum raw schema required for the transformation.
    # Unpacking METRIC_MAP.keys() includes every expected AESO metric column.
    required_raw_columns = {
        "Date - MST",
        "Fuel Type",
        *METRIC_MAP.keys(),
    }

    # Set subtraction returns the required names that are absent from the
    # actual DataFrame columns, providing a direct missing-schema check.
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

    # melt converts the separate metric columns into rows, producing one
    # measurement per timestamp, fuel type, and metric. This normalized long
    # form makes it easy to build standardized feature names before pivoting.
    long = df.melt(
        id_vars=["timestamp_utc", "fuel_type"],
        value_vars=list(METRIC_MAP.keys()),
        var_name="metric",
        value_name="value",
    )

    long["fuel_clean"] = long["fuel_type"].map(FUEL_MAP)
    long["metric_clean"] = long["metric"].map(METRIC_MAP)
    long["feature"] = long["fuel_clean"] + "_" + long["metric_clean"]

    # pivot_table converts the normalized rows into one row per timestamp.
    # Each generated feature name becomes a column, and repeated observations
    # for the same timestamp and feature are combined using sum.
    wide = (
        long
        .pivot_table(
            index="timestamp_utc",
            columns="feature",
            values="value",
            aggfunc="sum",
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
        .drop_duplicates(subset=["timestamp_utc"], keep="last")
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )

    return out


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
    # The return annotation documents that this function produces three
    # results in order: the audit table, the feature summary, and a pass flag.
    rows = []

    def add(check, passed, observed=None, expected=None, severity="error", notes=""):
        """
        General purpose:
            Append one standardized audit result (as a dict) to the shared
            `rows` list, so every check is recorded in the same shape.

        Role in the pipeline:
            This is a local helper used only within audit_generation(). It
            keeps every individual check consistent so they can all be
            converted into a single audit_df table at the end.
        """
        # These optional parameters provide defaults for ordinary error-level
        # checks while allowing callers to attach expectations, notes, or a
        # different severity when a check is informational or non-fatal.
        rows.append({
            "check": check,
            "pass": bool(passed),
            "severity": severity,
            "observed": observed,
            "expected": expected,
            "notes": notes,
        })

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
    audit_pass = audit_df.loc[audit_df["severity"].eq("error"), "pass"].all()

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
    # The -> None annotation documents that this function does not return a
    # value. Its work consists entirely of printing the audit to the terminal.
    audit_pass = audit_df.loc[audit_df["severity"] == "error", "pass"].all()
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
    # overwrite defaults to False so existing outputs are normally preserved.
    # The function always returns a dictionary describing whether processing
    # was skipped, failed its audit, completed successfully, or raised an error.
    start_time = time.perf_counter()

    if OUTPUT_CSV.exists() and OUTPUT_PARQUET.exists() and not overwrite:
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

        AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        audit_df.to_csv(AUDIT_FILE, index=False)
        summary_df.to_csv(SUMMARY_FILE, index=False)

        if not audit_pass:
            return {
                "dataset": "generation_by_fuel",
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


# Python assigns __name__ the value "__main__" when this file is executed
# directly. When the file is imported, __name__ contains the module name,
# which prevents main() from running automatically during the import.
if __name__ == "__main__":
    main()