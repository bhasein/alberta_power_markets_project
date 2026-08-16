"""
================================================================================
PURPOSE:
    Preprocess and audit AESO hourly intertie-capability data.

WHY THIS FILE IS USEFUL:
    AESO intertie-capability data records the amount of transfer capacity
    available between Alberta and neighbouring markets. The raw file includes
    both Available Transfer Capability (ATC) and Total Transfer Capability
    (TTC) for Saskatchewan and WECC import/export directions. This file
    standardizes those measurements, converts the fixed-MST timestamps to UTC,
    checks the expected sign conventions and ATC-versus-TTC relationships,
    and produces a clean hourly dataset for downstream market analysis.

PIPELINE OVERVIEW:
    raw AESO intertie-capability file
        --> load_raw_intertie_capability() reads the source table
        --> clean_intertie_capability()    standardizes timestamps and measures
        --> audit_intertie_capability()    validates coverage and domain rules
        --> print_audit_report()           presents the checks for review
        --> process_intertie_capability()  writes audit and approved products
        --> main()                         exposes the pipeline through the CLI
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
    INTERTIE_CAPABILITY_CSV,
    INTERTIE_CAPABILITY_PARQUET,
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

RAW_INTERTIE_FILE = PROJECT_ROOT / "data" / "raw" / "Intertie Table.csv"
OUTPUT_CSV = INTERTIE_CAPABILITY_CSV
OUTPUT_PARQUET = INTERTIE_CAPABILITY_PARQUET
AUDIT_DIR = PREPROCESSING_AUDITS_DIR
AUDIT_FILE = AUDIT_DIR / "intertie_capability_audit_checks.csv"
SUMMARY_FILE = AUDIT_DIR / "intertie_capability_feature_summary.csv"

EXPECTED_COLUMNS = {
    "timestamp_utc",
    "atc_sk_export",
    "atc_sk_import",
    "atc_wecc_export",
    "atc_wecc_import",
    "ttc_sk_export",
    "ttc_sk_import",
    "ttc_wecc_export",
    "ttc_wecc_import",
}

RAW_COLUMN_MAP = {
    "ATC SK Export": "atc_sk_export",
    "ATC SK Import": "atc_sk_import",
    "ATC Export": "atc_wecc_export",
    "ATC Import": "atc_wecc_import",
    "TTC SK Export": "ttc_sk_export",
    "TTC SK Import": "ttc_sk_import",
    "TTC Export": "ttc_wecc_export",
    "TTC Import": "ttc_wecc_import",
}

RANGE_EXPECTATIONS = {

    "atc_sk_export": (-2000, 0),
    "atc_sk_import": (0, 2000),
    "atc_wecc_export": (-3000, 0),
    "atc_wecc_import": (0, 3000),
    "ttc_sk_export": (-2000, 0),
    "ttc_sk_import": (0, 2000),
    "ttc_wecc_export": (-3000, 0),
    "ttc_wecc_import": (0, 3000),
}

EXPORT_COLUMNS = {
    "atc_sk_export",
    "atc_wecc_export",
    "ttc_sk_export",
    "ttc_wecc_export",
}

IMPORT_COLUMNS = {
    "atc_sk_import",
    "atc_wecc_import",
    "ttc_sk_import",
    "ttc_wecc_import",
}


def load_raw_intertie_capability(
    raw_file: Path = RAW_INTERTIE_FILE,
) -> pd.DataFrame:
    """
    General purpose:
        Read the raw AESO intertie-capability table from disk using the
        encoding and delimiter used by the source file.

    Role in the pipeline:
        This is the ingestion stage. It verifies that the source file exists
        and returns the untouched raw table for schema validation and
        standardization in clean_intertie_capability().
    """
    if not raw_file.exists():
        raise FileNotFoundError(f"Missing raw intertie capability file: {raw_file}")

    return pd.read_csv(raw_file, encoding="utf-16", sep="\t", low_memory=False)


def clean_intertie_capability(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    General purpose:
        Standardize the raw intertie-capability table by identifying its
        timestamp column, converting fixed-MST timestamps to UTC, renaming
        capability fields, and converting measurements to numeric values.

    Role in the pipeline:
        This is the transformation stage. It converts the source table into
        one clean chronological record per UTC hour using the exact schema
        expected by audit_intertie_capability() and downstream models.
    """
    raw = raw.copy()

    raw.columns = (
        raw.columns
        .str.strip()
        .str.replace("\ufeff", "", regex=False)
    )

    timestamp_candidates = [
        "Date (MST)",
        "Date - MST",
        "Date_Begin_Local",
        "Date",
        "timestamp",
    ]

    timestamp_col = None
    for col in timestamp_candidates:
        if col in raw.columns:
            timestamp_col = col
            break

    if timestamp_col is None:
        raise ValueError(f"Could not find timestamp column. Columns: {list(raw.columns)}")

    missing = set(RAW_COLUMN_MAP.keys()) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing raw intertie capability columns: {missing}")

    out = pd.DataFrame()

    local_ts = pd.to_datetime(
        raw[timestamp_col],
        format="%m/%d/%Y %I:%M:%S %p",
        errors="coerce",
    )

    out["timestamp_utc"] = (
        local_ts
        .dt.tz_localize("Etc/GMT+7")
        .dt.tz_convert("UTC")
    )

    for raw_col, clean_col in RAW_COLUMN_MAP.items():
        out[clean_col] = pd.to_numeric(
            raw[raw_col]
            .astype(str)
            .str.strip()
            .str.replace(",", "", regex=False),
            errors="coerce",
    )

    out = out.dropna(subset=["timestamp_utc"])
    out, exact_duplicate_rows = deduplicate_or_raise(
        out,
        ["timestamp_utc"],
        dataset_name="intertie capability",
    )
    out = out.sort_values("timestamp_utc").reset_index(drop=True)

    out = out.loc[:, ~out.columns.duplicated()]
    return set_duplicate_stats(
        out,
        exact_duplicate_rows=exact_duplicate_rows,
    )


def audit_intertie_capability(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    bool,
]:
    """
    General purpose:
        Run a complete validation suite against the cleaned intertie data,
        including timeline continuity, schema, missingness, value ranges,
        sign conventions, and ATC-versus-TTC relationships.

    Role in the pipeline:
        This is the quality-control gate between cleaning and saving. Its
        boolean pass/fail result determines whether process_intertie_capability()
        may write the cleaned dataset as an approved preprocessing product.
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
    add("no_partial_null_feature_columns", len(any_null) == 0, "; ".join(any_null), "", severity="warning")

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
            zero_count = int((df[col] == 0).sum())

            add(
                f"domain_zero_hours__{col}",
                True,
                observed=zero_count,
                expected="recorded",
                severity="info",
            )

            add(
                f"domain_min_recorded__{col}",
                True,
                observed=f"min={df[col].min(skipna=True):.6g}",
                expected="recorded",
                severity="info",
            )

            add(
                f"domain_max_recorded__{col}",
                True,
                observed=f"max={df[col].max(skipna=True):.6g}",
                expected="recorded",
                severity="info",
            )

            if col in EXPORT_COLUMNS:
                positive_export_count = int((df[col] > 0).sum())

                add(
                    f"domain_export_non_positive__{col}",
                    positive_export_count == 0,
                    observed=positive_export_count,
                    expected=0,
                    severity="warning",
                    notes="AESO export capability is encoded as negative or zero MW.",
                )

            if col in IMPORT_COLUMNS:
                negative_import_count = int((df[col] < 0).sum())

                add(
                    f"domain_import_non_negative__{col}",
                    negative_import_count == 0,
                    observed=negative_import_count,
                    expected=0,
                    severity="warning",
                    notes="AESO import capability is encoded as positive or zero MW.",
                )

    capability_pairs = [
        ("atc_sk_export", "ttc_sk_export", "export"),
        ("atc_sk_import", "ttc_sk_import", "import"),
        ("atc_wecc_export", "ttc_wecc_export", "export"),
        ("atc_wecc_import", "ttc_wecc_import", "import"),
    ]

    for atc_col, ttc_col, direction in capability_pairs:
        if {atc_col, ttc_col}.issubset(df.columns):

            if direction == "export":

                valid_mask = df[[atc_col, ttc_col]].notna().all(axis=1)
                violations = int((df.loc[valid_mask, atc_col].abs() > df.loc[valid_mask, ttc_col].abs()).sum())

                add(
                    f"domain_abs_atc_not_above_abs_ttc__{atc_col}",
                    violations == 0,
                    observed=violations,
                    expected=0,
                    severity="warning",
                    notes="Exports are negative; this compares absolute transfer capability.",
                )

            else:
                valid_mask = df[[atc_col, ttc_col]].notna().all(axis=1)
                violations = int((df.loc[valid_mask, atc_col] > df.loc[valid_mask, ttc_col]).sum())

                add(
                    f"domain_atc_not_above_ttc__{atc_col}",
                    violations == 0,
                    observed=violations,
                    expected=0,
                    severity="warning",
                    notes="Imports are positive; ATC should generally be less than or equal to TTC.",
                )

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
    audit_pass = audit_passes(audit_df)

    return audit_df, summary, bool(audit_pass)


def print_audit_report(
    audit_df: pd.DataFrame,
    feature_summary: pd.DataFrame,
    clean: pd.DataFrame,
) -> None:
    """
    General purpose:
        Print a formatted, human-readable summary of the audit checks and
        feature statistics to the terminal.

    Role in the pipeline:
        This is the presentation stage. It does not modify the data or the
        pass/fail result; it exposes the findings from
        audit_intertie_capability() for manual review.
    """
    audit_pass = audit_passes(audit_df)
    failed = audit_df.loc[~audit_df["pass"]]

    ts = pd.DatetimeIndex(pd.to_datetime(clean["timestamp_utc"], utc=True))
    expected_hours = len(pd.date_range(ts.min(), ts.max(), freq="h", tz="UTC"))

    print("\n" + "=" * 80)
    print("INTERTIE CAPABILITY AUDIT")
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


def process_intertie_capability(
    overwrite: bool = False,
) -> dict:
    """
    General purpose:
        Run the complete intertie-capability pipeline: load the raw source,
        clean it, audit it, print the audit report, and conditionally write
        the approved dataset and audit evidence.

    Role in the pipeline:
        This is the top-level orchestrator called by main(). It reuses existing
        outputs unless overwrite is requested and only writes the final cleaned
        products after every error-level audit check passes.
    """
    start_time = time.perf_counter()

    expected_manifest = build_manifest(
        dataset="intertie_capability",
        source_paths=[RAW_INTERTIE_FILE],
        code_paths=preprocessing_code_paths(Path(__file__)),
    )

    if not overwrite and outputs_are_current(
        [OUTPUT_CSV, OUTPUT_PARQUET, AUDIT_FILE, SUMMARY_FILE],
        expected_manifest,
    ):
        return {
            "dataset": "intertie_capability",
            "status": "skipped_existing",
            "pass": True,
            "csv_file": str(OUTPUT_CSV),
            "parquet_file": str(OUTPUT_PARQUET),
        }

    try:
        raw = load_raw_intertie_capability()
        clean = clean_intertie_capability(raw)
        audit_df, summary_df, audit_pass = audit_intertie_capability(clean)
        print_audit_report(audit_df, summary_df, clean)

        write_audit_artifacts(
            {AUDIT_FILE: audit_df, SUMMARY_FILE: summary_df}
        )

        if not audit_pass:
            return {
                "dataset": "intertie_capability",
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
            "dataset": "intertie_capability",
            "status": "saved",
            "pass": True,
            "rows": len(clean),
            "features": len(clean.columns) - 1,
            "start": str(clean["timestamp_utc"].min()),
            "end": str(clean["timestamp_utc"].max()),
            "raw_file": str(RAW_INTERTIE_FILE),
            "raw_file_size_mb": round(RAW_INTERTIE_FILE.stat().st_size / 1024**2, 3),
            "csv_file": str(OUTPUT_CSV),
            "parquet_file": str(OUTPUT_PARQUET),
            "audit_file": str(AUDIT_FILE),
            "summary_file": str(SUMMARY_FILE),
            "processing_seconds": round(time.perf_counter() - start_time, 3),
        }

    except DuplicateConflictError as exc:
        write_audit_artifacts({AUDIT_FILE: duplicate_failure_audit(exc)})
        return {
            "dataset": "intertie_capability",
            "status": "audit_failed",
            "pass": False,
            "error": str(exc),
            "audit_file": str(AUDIT_FILE),
            "processing_seconds": round(time.perf_counter() - start_time, 3),
        }

    except Exception as e:
        return {
            "dataset": "intertie_capability",
            "status": "error",
            "pass": False,
            "error": repr(e),
            "processing_seconds": round(time.perf_counter() - start_time, 3),
        }


def main() -> None:
    """
    General purpose:
        Provide a command-line entry point for running the intertie-capability
        preprocessing pipeline directly from a terminal.

    Role in the pipeline:
        This is the outermost wrapper. It parses the --overwrite option,
        calls process_intertie_capability(), and prints the returned status
        dictionary so the user can see what the pipeline did.
    """
    parser = argparse.ArgumentParser(description="Preprocess AESO intertie capability data.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    result = process_intertie_capability(overwrite=args.overwrite)

    print("\n" + "=" * 80)
    print("INTERTIE CAPABILITY PREPROCESSING RESULT")
    print("=" * 80)

    for key, value in result.items():
        print(f"{key}: {value}")

    print("=" * 80)


if __name__ == "__main__":
    main()
