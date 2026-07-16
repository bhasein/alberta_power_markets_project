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

# Imports
from pathlib import Path
import argparse
import pandas as pd
import time


# Path objects represent filesystem locations, and the / operator appends
# folders or filenames without manually constructing path strings.
PROJECT_ROOT = Path("/Users/brodiehasein/alberta_power_markets_project")

RAW_OUTAGE_FILE = PROJECT_ROOT / "data/raw/Outage Chart_Full Data_data.csv"

PREPROCESSING_DIR = PROJECT_ROOT / "data/preprocessing"
AUDIT_DIR = PROJECT_ROOT / "data/audits"

OUTPUT_CSV = PREPROCESSING_DIR / "outages_preprocessed.csv"
OUTPUT_PARQUET = PREPROCESSING_DIR / "outages_preprocessed.parquet"

AUDIT_FILE = AUDIT_DIR / "outages_audit_checks.csv"
SUMMARY_FILE = AUDIT_DIR / "outages_feature_summary.csv"

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

# This set defines the exact schema expected after cleaning. Sets do not
# preserve a meaningful order, which makes them useful for detecting missing
# or unexpected columns through set subtraction.
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

# This ordered list identifies the fuel-specific outage columns. Its order is
# reused when selecting the final output schema and calculating total outages.
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

# Each mapping value is a reasonable lower and upper bound used to flag
# implausible outage-capacity observations during the audit.
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
        # Throw an error if the raw file is missing. 
        # The result of the preprocessing can not complete otherwise
        raise FileNotFoundError(f"Missing raw outage file: {raw_file}")

    # Read and return the raw csv file. 
    # Handle utf-16 encoding, and tabular-separated. 
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
    # Loop through columns in a dataframe. 
    # Store formated - lowercase stripped column names - as keys, 
    # and the original name as the corresponding value.
    lower_map = {
        c.lower().strip(): c 
        for c in df.columns
    }

    # Loop through each candidate in order, 
    # if the normalized candidate matches any of the keys in the lower_map dictionary, 
    # return the value associated with that key. 
    for candidate in candidates:
        key = candidate.lower().strip()
        if key in lower_map:
            return lower_map[key]

    # Raise an error if the candidate values don't match any of the column names. 
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

    # Protect the original dataframe by creating a copy for use in this function.
    raw = raw.copy()

    # Clean up the source's column names by removing whitespace, 
    # and deleting a possible utf-8 byte-order mark from the header text.
    raw.columns = (
        raw.columns
        .str.strip()
        # regex=False means "treat \ufeff as literal text, not a regular-expression pattern".
        .str.replace("\ufeff", "", regex=False)
    )

    # Identify the raw column containing local fixed-MST timestamps.
    timestamp_col = "Date - MST"

    # Perform a set union between the timestamp column and the raw fuel columns.
    # These together define the required source schema.
    required_cols = {timestamp_col} | set(FUEL_MAP.keys())
    # Find required columns that are absent from the raw DataFrame.
    missing = required_cols - set(raw.columns)

    # If there are missing columns, throw an error printing them out. 
    if missing:
        raise ValueError(f"Missing raw outage columns: {missing}")

    # Keep only the source columns required for the clean outage dataset. 
    df = raw[[timestamp_col] + list(FUEL_MAP.keys())].copy()

    # Parse fixed-format local timestamp strings; invalid values become NaT. 
    local_ts = pd.to_datetime(
        df[timestamp_col],
        format="%m/%d/%Y %I:%M:%S %p",
        errors="coerce",
    )

    # Create new 'out' dataframe. 
    out = pd.DataFrame()

    # Localize the source timestamps to fixed MST and convert them to UTC. 
    out["timestamp_utc"] = (
        local_ts
        .dt.tz_localize("Etc/GMT+7")
        .dt.tz_convert("UTC")
    )

    # Loop through the fuel_map dictionary, storing keys in the raw_col, 
    # and values in the fuel_clean. 
    for raw_col, fuel_clean in FUEL_MAP.items():
        # Create column names in the out dataframe for each normalized fuel type
        # Convert each raw fuel column to numeric values and store it
        # under its standardized outage column name.
        out[f"{fuel_clean}_outage"] = pd.to_numeric(df[raw_col], errors="coerce")

    # Sum outages across columns along the same row to create a new total outage column. 
    out["total_outage"] = out[OUTAGE_COLUMNS].sum(axis=1, min_count=1)

    # Cleanup chain
    out = (
        out
        # Remove rows where timestamp parsing failed. 
        .dropna(subset=["timestamp_utc"])
        # Remove duplicate timestamps, keeping the last occurrence.
        .drop_duplicates(subset=["timestamp_utc"], keep="last")
        # Sort values by timestamp. 
        .sort_values("timestamp_utc")
        # Rebuild a sequential row index without retaining the old index as a column.
        .reset_index(drop=True)
    )

    # Create a list of final columns: timestamp, individual outage columns, and a total outage column. 
    final_columns = ["timestamp_utc"] + OUTAGE_COLUMNS + ["total_outage"]

    # Return a final dataset containing each of the final_columns 
    return out[final_columns]


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

    # Create empty row list to append audit dictionaries to. 
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
        Add one audit result to the shared rows list. 
        """

        # Append a dictionary to the rows list. 
        # Standardize the structure of every audit result. 
        rows.append({
            "check": check,
            "pass": bool(passed),
            "severity": severity,
            "observed": observed,
            "expected": expected,
            "notes": notes,
        })

    # Confirms that the required utc timestamp column exists. 
    add("timestamp_column_exists", "timestamp_utc" in df.columns, "timestamp_utc" in df.columns, True)

    # Stop the function early if the timestamp is not found. 
    # Later audits require time series to be present. 
    if "timestamp_utc" not in df.columns:
        # Return the audit result recorded so far, an empty summary dataframe, and False.
        # Otherwise later lines will crash unpredictably. 
        audit_df = pd.DataFrame(rows)
        return audit_df, pd.DataFrame(), False

    # Convert the timestamp column into a UTC DatetimeIndex for efficient 
    # time-series validation.
    ts = pd.DatetimeIndex(pd.to_datetime(df["timestamp_utc"], utc=True))
    # Treat every non-timestamp column as an outage feature. 
    feature_cols = [c for c in df.columns if c != "timestamp_utc"]

    # Record the observed time range.
    observed_start = ts.min()
    observed_end = ts.max()
    # Construct the complete hourly UTC index that should exist between 
    # the first and last observed timestamps. 
    expected_index = pd.date_range(observed_start, observed_end, freq="h", tz="UTC")
    # Store the length of the time series.
    expected_hours = len(expected_index)

    # Confirm that the time series is not empty and record its observed coverage. 
    add("row_count_positive", len(df) > 0, len(df), "> 0")
    add("observed_period_start", True, str(observed_start), "recorded")
    add("observed_period_end", True, str(observed_end), "recorded")
    add("observed_hours", True, len(df), "recorded")

    # If the dataset has exactly one row per hour,
    # its row count should equal the length of the hourly index. 
    add("expected_hours_from_start_to_end", len(df) == expected_hours, len(df), expected_hours)

    # Confirm that the timestamps are already ordered chronologically. 
    # Confirm that timestamps aren't duplicated. 
    add("timestamps_monotonic", ts.is_monotonic_increasing, ts.is_monotonic_increasing, True)
    add("duplicate_timestamps", not ts.has_duplicates, int(ts.duplicated().sum()), 0)

    # Hour-to-hour continuity checks require at least two timestamps. 
    if len(ts) > 1:
        # Calculate the time difference bewteen each timestamp and the preivous one. 
        diffs = pd.Series(ts).diff().dropna()
        # Identify steps that are greater than one hour. 
        bad_diffs = diffs[diffs != pd.Timedelta(hours=1)]
        # Find hours that are in the range, but absent from the data. 
        missing_hours = expected_index.difference(ts)
        # Find timestamps that exist in the data, but not in the expected index.
        extra_hours = ts.difference(expected_index)

        add("hourly_spacing", len(bad_diffs) == 0, len(bad_diffs), 0)
        add("missing_hours", len(missing_hours) == 0, len(missing_hours), 0)
        add("extra_hours", len(extra_hours) == 0, len(extra_hours), 0)

    # Compare the dataset's actual schema with the expected schema. 
    actual_cols = set(df.columns)
    # Expected columns absent from the dataset. 
    missing_cols = EXPECTED_COLUMNS - actual_cols
    # Columns present in the dataset but not included in the expected schema. 
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

    # Identify feature columns that are not stored as numeric dtypes. 
    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    add("all_features_numeric", len(non_numeric) == 0, "; ".join(non_numeric), "all numeric")

    # Find feature columns containing only missing values. 
    all_null = [c for c in feature_cols if df[c].isna().all()]
    # Find feature columns containing at least one missing value. 
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

    # Compare configured outage features iwth broad possible range. 
    # These checks are warnings because unusualvalues should be reviewed, 
    # but they should not prevent the block from saving. 
    for col, (low, high) in RANGE_EXPECTATIONS.items():
        # Skip missing values. 
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

    # Check that outage features contain no negative apacity values. 
    # Record number of zeros, missing counts, and peak values - informational but won't fail the entire block. 
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

    # Verify that all the outage columns are present in the dataframe. 
    # Only run the total outage check if every fuel-specific outage column exists in the dataframe. 
    if set(OUTAGE_COLUMNS).issubset(df.columns):
        # Sum across the columns for each row. 
        implied_total = df[OUTAGE_COLUMNS].sum(axis=1, min_count=1)
        # Calculate the difference between the stored total-outage value,
        # This difference must be negligible for the audit to complete. 
        diff = (df["total_outage"] - implied_total).abs()

        add(
            "domain_total_outage_matches_sum",
            diff.max(skipna=True) < 1e-6,
            observed=f"max_abs_diff={diff.max(skipna=True):.6g}",
            expected="0",
            severity="error",
        )

    # Create summary, with descriptive metrics of the feature columns at 
    # 10th, 25th, ..., 99th percentiles, and transpose those results to a row-format. 
    summary = (
        df[feature_cols]
        .describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99])
        .T
        .reset_index()
    )

    # Rename summary columns. 
    summary = summary.rename(columns={
        "index": "feature",
        "1%": "p01",
        "50%": "median",
        "99%": "p99",
    })

    # Create new summary columns for missing count, missing pct, and the datatype for each column.
    summary["missing_count"] = df[feature_cols].isna().sum().values
    summary["missing_pct"] = (summary["missing_count"] / len(df)) * 100
    summary["dtype"] = [str(df[c].dtype) for c in feature_cols]

    # Convert audit rows into a dataframe. 
    # Each dictionary in rows becomes one row in audit_df. 
    audit_df = pd.DataFrame(rows)
    # Approve the dataset only if every error-severity audit check passes. 
    audit_pass = audit_df.loc[audit_df["severity"].eq("error"), "pass"].all()

    # Return the detailed audit table, feature summary, and overall pass result. 
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
    # Recalculate the overall audit pass/fail result using only checks with error-level severity.
    # Warnings and informational checks do not affect the final pass status. 
    audit_pass = audit_df.loc[audit_df["severity"] == "error", "pass"].all()
    # Select every audit check that failed, regardless of severity. 
    failed = audit_df.loc[~audit_df["pass"]]

    # Convert the cleaned timestamp column into a UTC DatetimeIndex. 
    # Helps with converage and time-range calculations
    ts = pd.DatetimeIndex(pd.to_datetime(clean["timestamp_utc"], utc=True))
    # Expected Hours is the length of the date range bewteen the first and last timestamps. 
    expected_hours = len(pd.date_range(ts.min(), ts.max(), freq="h", tz="UTC"))

    # Formating. 
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

    # Print the details for all failed audit checks. 
    print("\nFailed checks:")
    if failed.empty:
        print("  None")
    else:
        for _, row in failed.iterrows():
            print(
                f"  - {row['check']} [{row['severity']}] "
                f"observed={row['observed']} expected={row['expected']}"
            )

    # Printing the completed audit checklist, including passed and failed attempts. 
    print("\nAudit checklist:")
    for _, row in audit_df.iterrows():
        icon = "✓" if row["pass"] else "✗"
        sev = row["severity"]
        check = row["check"]
        print(f"  {icon} {check} ({sev})")

    # Create copy of feature summary, and key columns. 
    # This summary will be used for terminal display.
    print("\nFeature statistics:")
    compact = feature_summary[
        ["feature", "missing_count", "mean", "median", "p01", "p99", "min", "max"]
    ].copy()

    # Rename the missing_count column. 
    compact = compact.rename(columns={"missing_count": "missing"})

    # Round the value in each column to 2 decimal places. 
    for col in ["mean", "median", "p01", "p99", "min", "max"]:
        compact[col] = compact[col].round(2)

    # Print the compact table without displaying its DataFrame index. 
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

    # Record the starting time so total pipeline runtime can be calculated. 
    start_time = time.perf_counter()

    # Reuse existing processed outputs unless the caller explicitly requests that they be rebuilt. 
    if OUTPUT_CSV.exists() and OUTPUT_PARQUET.exists() and not overwrite:
        return {
            "dataset": "outages",
            "status": "skipped_existing",
            "pass": True,
            "csv_file": str(OUTPUT_CSV),
            "parquet_file": str(OUTPUT_PARQUET),
        }

    # Try the pipeline.
    # If anything fails, jump to the exception block. 
    try:
        # Load the raw outage source file. 
        raw = load_raw_outages()
        # Transform the raw source file into the standardized hourly outage dataset. 
        clean = clean_outages(raw)
        # Run validation checks and generate descriptive feature statistics. 
        audit_df, summary_df, audit_pass = audit_outages(clean)
        # Print a readable/ formatted results section to the terminal. 
        print_audit_report(audit_df, summary_df, clean)

        # Ensure the audit output directory exists before writing audit files. 
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        # Save the detailed audit checks and feature summary regardless of 
        # whether the cleaned dataset passed approval. 
        # It's good to save the audit summary/df first - even if the pipeline failed this will help diagnose it. 
        audit_df.to_csv(AUDIT_FILE, index=False)
        summary_df.to_csv(SUMMARY_FILE, index=False)

        # Stop the pipeline if any error-severity audit check failed. 
        if not audit_pass:
            return {
                "dataset": "outages",
                "status": "audit_failed",
                "pass": False,
                "audit_file": str(AUDIT_FILE),
                "summary_file": str(SUMMARY_FILE),
                "processing_seconds": round(time.perf_counter() - start_time, 3),
            }

        # Ensure the preprocessing output directory exists before saving
        # the approvaed clean dataset. 
        PREPROCESSING_DIR.mkdir(parents=True, exist_ok=True)

        # Save the approved dataset in both CSV and parquet formats. 
        clean.to_csv(OUTPUT_CSV, index=False)
        clean.to_parquet(OUTPUT_PARQUET, index=False)

        # Return a structured success record describing the saved outputs, 
        # dataset dimensions, coverage, source file, and runtime. 
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

    except Exception as e:
        # Catch unexpected pipeline errors and return a structured failure
        # record instead of terminating without context. 
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
    # Define parser for argparse command-line prompts. 
    # Create a prompt '--overwrite', if supplied "overwrite=True"
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


# Python assigns __name__ the value "__main__" when this file is executed
# directly. When the file is imported, __name__ contains the module name,
# which prevents main() from running automatically during the import.
if __name__ == "__main__":
    main()