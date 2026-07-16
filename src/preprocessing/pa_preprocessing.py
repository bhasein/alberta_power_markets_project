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

# Imports
from pathlib import Path
import argparse
import pandas as pd 
import time
 
# Path objects represent filesystem locations, and the / operator appends
# folders or filenames without manually constructing platform-specific strings.
PROJECT_ROOT = Path("/Users/brodiehasein/alberta_power_markets_project")


RAW_PA_FILE = PROJECT_ROOT / "data" / "raw" / "P&A Table_Full Data_data.csv"


PREPROCESSING_DIR = PROJECT_ROOT / "data" / "preprocessing"
AUDIT_DIR = PROJECT_ROOT / "data" / "audits"


OUTPUT_CSV = PREPROCESSING_DIR / "pa_hourly.csv"
OUTPUT_PARQUET = PREPROCESSING_DIR / "pa_hourly.parquet"


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
        # Throw an error if the required file is missing.
        raise FileNotFoundError(f"Missing raw P&A file: {raw_file}")

    # Read the csv with pandas (file is utf-16 encoded and tab-separated).
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

    # Python set for the required columns.
    required_raw_columns = {
        "Date (MST)",
        "AIL",
        "Gas Price",
        "Price",
        "Spark Spread",
    }

    # Checks for missing columns between required/raw file.
    missing = required_raw_columns - set(raw.columns)
    if missing:
        # Print error comment with set of missing columns.
        raise ValueError(f"Missing raw P&A columns: {missing}")

    # Make a copy instead of modiying the original dataframe - raw remains untouched.
    df = raw.copy()

    out = pd.DataFrame()

    # Converts string timestamps into datetime objects. 
    # If a row contains non-date like strings (ex. 'hello'), then errors="coerce" turn it into 'NaT'
    local_ts = pd.to_datetime(df["Date (MST)"], errors="coerce")

    # New timestamp_utc column
    out["timestamp_utc"] = (
        local_ts
        # tz_localize specifies that timestamps are fixed MST (UTC-7)
        .dt.tz_localize("Etc/GMT+7")   
        # Converts timestamps to UTC
        # Ex. 2015-01-01 00:00 MST --> 2015-01-01 07:00 UTC
        .dt.tz_convert("UTC")
    )

    # Creates key feature columns, converting from strings to numeric values
    # Errors="coerce" handles non numeric values (ex. "N/A" becomes NaN)
    out["ail_mw"] = pd.to_numeric(df["AIL"], errors="coerce")
    out["gas_price_cad_gj"] = pd.to_numeric(df["Gas Price"], errors="coerce")
    out["pool_price_cad_mwh"] = pd.to_numeric(df["Price"], errors="coerce")
    out["spark_spread"] = pd.to_numeric(df["Spark Spread"], errors="coerce")

    # Sort values by timestamp, reset row values, discard old index
    out = out.sort_values("timestamp_utc").reset_index(drop=True)
    # Defensive measure removes duplicate columns
    out = out.loc[:, ~out.columns.duplicated()]

    # Return cleaned dataframe
    return out


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

    # Empty rows list for standardized audit-result dictionaries
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
            dictionary structure so the enclosing function can assemble one
            consistent audit DataFrame at the end.
        """

        # Append one standardized audit-result dictionary to the shared rows list
        rows.append({
            "check": check,
            "pass": bool(passed),
            "severity": severity,
            "observed": observed,
            "expected": expected,
            "notes": notes,
        })

    # Check whether the required UTC timestamp column exists
    add(
        "timestamp_column_exists",
        "timestamp_utc" in df.columns,
        "timestamp_utc" in df.columns,
        True,
    )

    # Stop the audit early if the timestamp column is missing
    # The remaining timeline checks depend on a valid timestamp column
    if "timestamp_utc" not in df.columns:
        audit_df = pd.DataFrame(rows)
        return audit_df, pd.DataFrame(), False

    # Parse the timestamp column as UTC datetimes and store it as a DatetimeIndex
    ts = pd.DatetimeIndex(
        pd.to_datetime(
            df["timestamp_utc"],
            utc=True,
        )
    )

    # Store every cleaned-data column except the UTC timestamp column
    feature_cols = [
        c
        for c in df.columns
        if c != "timestamp_utc"
    ]

    # Record the earliest and latest observed timestamps
    observed_start = ts.min()
    observed_end = ts.max()

    # Construct the complete hourly UTC index expected between the observed endpoints
    expected_index = pd.date_range(
        observed_start,
        observed_end,
        freq="h",
        tz="UTC",
    )

    # Count the number of hourly observations expected across the full period
    expected_hours = len(expected_index)

    # Add basic dataset-size and coverage metadata checks
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

    # Check whether timestamps are ordered chronologically
    add(
        "timestamps_monotonic",
        ts.is_monotonic_increasing,
        ts.is_monotonic_increasing,
        True,
    )

    # Check whether every timestamp is unique
    add(
        "duplicate_timestamps",
        not ts.has_duplicates,
        int(ts.duplicated().sum()),
        0,
    )

    # Timeline-spacing checks require at least two timestamp observations
    if len(ts) > 1:
        # Calculate the interval between each pair of consecutive timestamps
        # The first difference is always NaT and is therefore removed
        diffs = pd.Series(ts).diff().dropna()

        # Store intervals that are not exactly one hour
        bad_diffs = diffs[
            diffs != pd.Timedelta(hours=1)
        ]

        # Store expected hourly timestamps that are absent from the observed data
        missing_hours = expected_index.difference(ts)

        # Store observed timestamps that do not belong to the expected hourly index
        extra_hours = ts.difference(expected_index)

        # Add hourly spacing, missing-hour, and extra-hour checks
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

    # Convert the cleaned DataFrame column names into a set
    actual_cols = set(df.columns)

    # Find required columns that are absent from the cleaned dataset
    missing_cols = EXPECTED_COLUMNS - actual_cols

    # Find columns present in the cleaned dataset but absent from the expected schema
    extra_cols = actual_cols - EXPECTED_COLUMNS

    # Add required-column and unexpected-column checks
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

    # Store feature columns whose pandas data types are not numeric
    non_numeric = [
        c
        for c in feature_cols
        if not pd.api.types.is_numeric_dtype(df[c])
    ]

    # Check whether every non-timestamp feature has a numeric data type
    add(
        "all_features_numeric",
        len(non_numeric) == 0,
        "; ".join(non_numeric),
        "all numeric",
    )

    # Store feature columns in which every value is missing
    all_null = [
        c
        for c in feature_cols
        if df[c].isna().all()
    ]

    # Store feature columns containing at least one missing value
    any_null = [
        c
        for c in feature_cols
        if df[c].isna().any()
    ]

    # Treat completely empty feature columns as an error-level failure
    add(
        "no_all_null_feature_columns",
        len(all_null) == 0,
        "; ".join(all_null),
        "",
    )

    # Treat partially missing feature columns as a warning for manual review
    add(
        "no_partial_null_feature_columns",
        len(any_null) == 0,
        "; ".join(any_null),
        "",
        severity="warning",
    )

    # Check each configured feature against its broad plausible-value range
    for col, (low, high) in RANGE_EXPECTATIONS.items():
        # Only run the range check when the configured feature exists
        if col in df.columns:
            # Record the observed minimum and maximum values
            col_min = df[col].min()
            col_max = df[col].max()

            # Pass when both observed extremes remain inside the configured range
            add(
                f"range_check__{col}",
                (col_min >= low) and (col_max <= high),
                observed=f"min={col_min:.6g}, max={col_max:.6g}",
                expected=f"[{low}, {high}]",
                severity="warning",
            )

    # Record the number of pool-price observations above important spike thresholds
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

        # Add informational records for each pool-price spike threshold
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

    # Record the number of hours with an exact zero pool price
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

    # Record the number of hours with a negative natural-gas price
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

    # Validate that Alberta Internal Load remains positive and record its peak
    if "ail_mw" in df.columns:
        # Record the observed minimum and maximum AIL values
        ail_min = df["ail_mw"].min()
        ail_max = df["ail_mw"].max()

        # Require all observed AIL values to remain above zero
        add(
            "domain_ail_positive",
            ail_min > 0,
            f"min={ail_min:.6g}",
            "> 0",
        )

        # Record the maximum observed AIL as informational audit evidence
        add(
            "domain_ail_peak_recorded",
            True,
            f"max={ail_max:.6g}",
            "recorded",
            severity="info",
        )

    # Check broad consistency among pool price, gas price, and reported spark spread
    if {
        "pool_price_cad_mwh",
        "gas_price_cad_gj",
        "spark_spread",
    }.issubset(df.columns):
        # Rearrange the spark-spread relationship to estimate its implied heat rate
        # Zero gas prices are replaced with missing values to avoid division by zero
        implied_heat_rate = (
            df["pool_price_cad_mwh"]
            - df["spark_spread"]
        ) / df["gas_price_cad_gj"].replace(0, pd.NA)

        # Remove observations for which an implied heat rate cannot be calculated
        implied_heat_rate = implied_heat_rate.dropna()

        # Use the median to reduce sensitivity to unusual or near-zero gas-price hours
        median_hr = implied_heat_rate.median()

        # Check whether the median implied heat rate falls inside a broad plausible range
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

    # Calculate descriptive statistics for every non-timestamp feature
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

    # Rename the feature-name and selected percentile columns
    summary = summary.rename(columns={
        "index": "feature",
        "1%": "p01",
        "50%": "median",
        "99%": "p99",
    })

    # Add the number of missing observations for each feature
    summary["missing_count"] = (
        df[feature_cols]
        .isna()
        .sum()
        .values
    )

    # Convert each feature's missing count into a percentage of total rows
    summary["missing_pct"] = (
        summary["missing_count"] / len(df)
    ) * 100

    # Record the pandas data type of each feature
    summary["dtype"] = [
        str(df[c].dtype)
        for c in feature_cols
    ]

    # Convert the accumulated audit-result dictionaries into a DataFrame
    audit_df = pd.DataFrame(rows)

    # Overall audit passes only when every error-level check passes
    # Warning-level and information-level checks do not block approval
    audit_pass = audit_df.loc[
        audit_df["severity"].eq("error"),
        "pass",
    ].all()

    # Return the complete audit checklist, feature summary, and final pass result
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

    # Determines if error-level checks passed
    audit_pass = audit_df.loc[audit_df["severity"] == "error", "pass"].all()
    # Store failed audit checks
    failed = audit_df.loc[~audit_df["pass"]]

    # Convert clean timestamp column to datetime object, wrap resulting seriies in a DatetimeIndex 
    # Useful for functions like ts.min(), ts.max()
    ts = pd.DatetimeIndex(pd.to_datetime(clean["timestamp_utc"], utc=True))
    # Store the length of the number of hours in the df
    expected_hours = len(pd.date_range(ts.min(), ts.max(), freq="h", tz="UTC"))

    # Formating
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
    # Create a copy dataframe with only selected list of variables
    compact = feature_summary[
        ["feature", "missing_count", "mean", "median", "p01", "p99", "min", "max"]
    ].copy()

    # Dictionary rename map
    rename_map = {
        "missing_count": "missing",
        "mean": "mean",
        "median": "median",
        "p01": "p01",
        "p99": "p99",
        "min": "min",
        "max": "max",
    }

    # rebuild compact df with new names
    compact = compact.rename(columns=rename_map)

    # Round the value of each column to two decimal places
    for col in ["mean", "median", "p01", "p99", "min", "max"]:
        compact[col] = compact[col].round(2)

    # Fixed width textual table, index=False means row indecies aren't printed
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
    # time.perf_counter() returns a precise timer value (elapsed time)
    start_time = time.perf_counter()

    # Checks if output.csv exists, and output.parquet exists and if the request was to --overwrite existing files.
    if OUTPUT_CSV.exists() and OUTPUT_PARQUET.exists() and not overwrite:
        # Following is returned immediately (avoids the loading, cleaning, auditing, or saving processes below)
        return {
            "dataset": "pa",
            "status": "skipped_existing",
            "pass": True,
            "csv_file": str(OUTPUT_CSV),
            "parquet_file": str(OUTPUT_PARQUET),
        }
    # If --overwrite is called, or the above conditions are false, the processing will run again

    # Try to catch unexpected errors
    try:
        # Calls data loading function
        raw = load_raw_pa()
        # Passes raw dataframe into the cleaning function
        clean = clean_pa(raw)
        # Calls audit process on the cleaned file
        audit_df, summary_df, audit_pass = audit_pa(clean)
        # Calls the print function on the audit and summary variables passes over from audit_pa()
        print_audit_report(audit_df, summary_df, clean)

        # Creates audit output folder exists
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        # Saves audit evidence
        audit_df.to_csv(AUDIT_FILE, index=False)
        summary_df.to_csv(SUMMARY_FILE, index=False)

        # Returns dictionary if the audit fails
        if not audit_pass:
            return {
                "dataset": "pa",
                "status": "audit_failed",
                "pass": False,
                "audit_file": str(AUDIT_FILE),
                "summary_file": str(SUMMARY_FILE),
                "processing_seconds": round(time.perf_counter() - start_time, 3),
            }

        # If the audit is passed, execution continues to creating/verifying preprocessing folder
        PREPROCESSING_DIR.mkdir(parents=True, exist_ok=True)

        # Output to csv and parquet, index=False avoids writing the pandas row index
        clean.to_csv(OUTPUT_CSV, index=False)
        clean.to_parquet(OUTPUT_PARQUET, index=False)

        # Dictionary which summarizes a successful run
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

    # Jump to this section if any statement inside the try block fails
    except Exception as e:
        # Return this dictionary for a failed attempt
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
    # argparse helps read command-line options.
    parser = argparse.ArgumentParser(description="Preprocess P&A hourly market data.")
    
    # Creates optional command-line flag called '--overwrite'
    parser.add_argument("--overwrite", action="store_true")
    # This line reads the command line
    args = parser.parse_args()

    # Run process_pa with the overwrite value carrying over from args.overwrite (True of False)
    result = process_pa(overwrite=args.overwrite)

    # Formating
    print("\n" + "=" * 80)
    print("P&A PREPROCESSING RESULT")
    print("=" * 80)

    for key, value in result.items():
        print(f"{key}: {value}")

    print("=" * 80) 


# Python assigns __name__ the value "__main__" when this file is executed
# directly. When the file is imported, __name__ contains the module name,
# which prevents main() from running automatically during the import.
if __name__ == "__main__":
    main()