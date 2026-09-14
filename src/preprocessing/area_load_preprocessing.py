"""Preprocess and audit AESO hourly regional load with a stable 2025 cutover.

Purpose
-------
The historical regional-load product contains the six regional series already
used throughout the project through December 2024. The newer AESO Load Chart
export contains Meter Load, behind-the-fence (BTF) Load, and Actual Load for
six regions through 2026.

This pipeline preserves the existing canonical regional series through 2024
and switches to observed Meter Load from the Load Chart export beginning
2025-01-01 00:00 MST. This prevents the source transition from silently
rewriting historical notebook results while removing the former frozen
December-2024 extension.

Pipeline
--------
existing canonical regional-load product + Load Chart long-form export
    -> retain existing regional values through 2024
    -> read and validate the Load Chart export
    -> reshape Load Chart regions and measures to one row per UTC hour
    -> append observed regional Meter Load from 2025 onward
    -> audit identities, coverage, continuity, overlap, and cutover behavior
    -> overwrite the canonical regional-load CSV and Parquet table

The existing processed regional-load product is therefore required once as the
historical source for the cutover migration. Subsequent runs preserve the same
pre-2025 history from the canonical output itself.
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    PREPROCESSING_AUDITS_DIR,
    RAW_DIR,
    REGIONAL_LOAD_CSV,
    REGIONAL_LOAD_PARQUET,
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


# ---------------------------------------------------------------------
# Paths and schema
# ---------------------------------------------------------------------

DATASET_NAME = "regional_load"
RAW_LOAD_CHART_FILE = RAW_DIR / "Load Chart_Full Data_data.csv"

OUTPUT_CSV = REGIONAL_LOAD_CSV
OUTPUT_PARQUET = REGIONAL_LOAD_PARQUET

AUDIT_FILE = PREPROCESSING_AUDITS_DIR / "regional_load_audit_checks.csv"
SUMMARY_FILE = PREPROCESSING_AUDITS_DIR / "regional_load_feature_summary.csv"
FILE_SUMMARY_FILE = PREPROCESSING_AUDITS_DIR / "regional_load_source_summary.csv"
OVERLAP_FILE = PREPROCESSING_AUDITS_DIR / "regional_load_overlap_audit.csv"

REGIONS = [
    "Calgary",
    "Central",
    "Edmonton",
    "Northeast",
    "Northwest",
    "South",
]

REGION_SLUGS = {region: region.lower() for region in REGIONS}
CANONICAL_REGION_COLUMNS = [
    f"{REGION_SLUGS[region]}_load_mw"
    for region in REGIONS
]

MEASURE_COLUMNS = {
    "Actual Load": "actual_load_mw",
    "BTF Load": "btf_load_mw",
    "Meter Load": "meter_load_mw",
}

REQUIRED_RAW_COLUMNS = {
    "Region",
    "Date - MST",
    "Date",
    "Date (MPT)",
    "Date (MST)",
    *MEASURE_COLUMNS,
}

CUTOVER_UTC = pd.Timestamp("2025-01-01 07:00:00", tz="UTC")


# ---------------------------------------------------------------------
# Raw Load Chart parsing and reshaping
# ---------------------------------------------------------------------

def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize superficial whitespace without altering AESO field names."""

    out = frame.copy()
    out.columns = (
        out.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return out


def read_load_chart(path: Path = RAW_LOAD_CHART_FILE) -> pd.DataFrame:
    """Read the UTF-16, tab-delimited Load Chart export despite its CSV suffix."""

    if not path.exists():
        raise FileNotFoundError(f"Missing AESO Load Chart export: {path}")

    attempts = [
        {"sep": "\t", "encoding": "utf-16"},
        {"sep": "\t", "encoding": "utf-8-sig"},
        {"sep": ",", "encoding": "utf-8-sig"},
    ]
    errors: list[str] = []

    for options in attempts:
        try:
            frame = normalize_columns(
                pd.read_csv(path, low_memory=False, **options)
            )
            if REQUIRED_RAW_COLUMNS.issubset(frame.columns):
                return frame
        except Exception as exc:  # pragma: no cover
            errors.append(f"{options}: {exc!r}")

    raise ValueError(
        "Could not read the Load Chart export with its expected schema. "
        f"Attempts: {errors}"
    )


def parse_fixed_mst(values: pd.Series) -> pd.Series:
    """Parse hour-beginning fixed-MST timestamps and convert them to UTC."""

    parsed = pd.to_datetime(
        values.astype("string").str.strip(),
        format="%m/%d/%Y %I:%M:%S %p",
        errors="coerce",
    )
    return parsed.dt.tz_localize("Etc/GMT+7").dt.tz_convert("UTC")


def clean_load_chart(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the long-form Load Chart observations."""

    raw = normalize_columns(raw)
    missing = sorted(REQUIRED_RAW_COLUMNS - set(raw.columns))
    if missing:
        raise ValueError(f"Load Chart export is missing columns: {missing}")

    clean = raw[list(REQUIRED_RAW_COLUMNS)].copy()
    clean["Region"] = clean["Region"].astype("string").str.strip()
    clean["timestamp_utc"] = parse_fixed_mst(clean["Date - MST"])

    for column in MEASURE_COLUMNS:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    clean = clean.dropna(subset=["timestamp_utc", "Region"])
    clean, exact_duplicates = deduplicate_or_raise(
        clean,
        ["timestamp_utc", "Region"],
        ignore_columns=["Date", "Date (MPT)", "Date (MST)", "Date - MST"],
        dataset_name="AESO Load Chart regional load",
    )
    clean = clean.sort_values(["timestamp_utc", "Region"]).reset_index(drop=True)
    return set_duplicate_stats(clean, exact_duplicate_rows=exact_duplicates)


def reshape_load_chart(clean: pd.DataFrame) -> pd.DataFrame:
    """Reshape region rows and load measures to one row per UTC hour."""

    available_regions = [*REGIONS, "Losses", "System Load"]
    source = clean.loc[clean["Region"].isin(available_regions)].copy()

    wide_parts: list[pd.DataFrame] = []

    for raw_measure, suffix in MEASURE_COLUMNS.items():
        pivot = source.pivot(
            index="timestamp_utc",
            columns="Region",
            values=raw_measure,
        )

        rename = {
            region: f"{REGION_SLUGS[region]}_{suffix}"
            for region in REGIONS
        }
        rename.update(
            {
                "Losses": f"losses_{suffix}",
                "System Load": f"system_{suffix}",
            }
        )
        wide_parts.append(pivot.rename(columns=rename))

    wide = pd.concat(wide_parts, axis=1).sort_index().reset_index()

    six_meter = [
        f"{REGION_SLUGS[region]}_meter_load_mw"
        for region in REGIONS
    ]
    six_actual = [
        f"{REGION_SLUGS[region]}_actual_load_mw"
        for region in REGIONS
    ]

    wide["total_meter_region_load_mw"] = (
        wide[six_meter].sum(axis=1, min_count=6)
    )
    wide["total_actual_region_load_mw"] = (
        wide[six_actual].sum(axis=1, min_count=6)
    )
    wide["actual_load_plus_losses_mw"] = (
        wide["total_actual_region_load_mw"]
        + wide["losses_meter_load_mw"]
    )
    wide["system_meter_reconciliation_error_mw"] = (
        wide["total_meter_region_load_mw"]
        - wide["system_meter_load_mw"]
    )

    return wide


# ---------------------------------------------------------------------
# Historical preservation and 2025 cutover
# ---------------------------------------------------------------------

def load_legacy_regional_history() -> tuple[pd.DataFrame, Path]:
    """Load the existing canonical product and retain its pre-2025 history."""

    if OUTPUT_PARQUET.exists():
        source_path = OUTPUT_PARQUET
        history = pd.read_parquet(source_path)
    elif OUTPUT_CSV.exists():
        source_path = OUTPUT_CSV
        history = pd.read_csv(source_path)
    else:
        raise FileNotFoundError(
            "No existing canonical regional-load output was found. "
            "This cutover processor requires the previous regional-load "
            "Parquet or CSV once so that pre-2025 values can be preserved."
        )

    if "timestamp_utc" not in history.columns:
        raise ValueError(
            "Existing regional-load product does not contain timestamp_utc."
        )

    missing = [
        column
        for column in CANONICAL_REGION_COLUMNS
        if column not in history.columns
    ]
    if missing:
        raise ValueError(
            "Existing regional-load product is missing required historical "
            f"columns: {missing}"
        )

    history["timestamp_utc"] = pd.to_datetime(
        history["timestamp_utc"],
        utc=True,
    )

    history = history.loc[
        history["timestamp_utc"] < CUTOVER_UTC
    ].copy()

    if "regional_load_imputed" in history.columns:
        imputed = "regional_load_imputed"
    elif "area_load_imputed" in history.columns:
        imputed = "area_load_imputed"
    else:
        history["regional_load_imputed"] = np.int8(0)
        imputed = "regional_load_imputed"

    keep = [
        "timestamp_utc",
        *CANONICAL_REGION_COLUMNS,
        imputed,
    ]

    history = history[keep].copy()

    if imputed != "regional_load_imputed":
        history = history.rename(
            columns={imputed: "regional_load_imputed"}
        )

    history["regional_load_imputed"] = (
        pd.to_numeric(
            history["regional_load_imputed"],
            errors="coerce",
        )
        .fillna(0)
        .astype(np.int8)
    )

    history = (
        history
        .sort_values("timestamp_utc")
        .drop_duplicates("timestamp_utc", keep="last")
        .reset_index(drop=True)
    )

    return history, source_path


def build_overlap_audit(
    legacy: pd.DataFrame,
    load_chart: pd.DataFrame,
) -> pd.DataFrame:
    """Quantify differences between Load Chart measures and retained history."""

    overlap = legacy.merge(
        load_chart,
        on="timestamp_utc",
        how="inner",
    )
    overlap = overlap.loc[
        overlap["timestamp_utc"] < CUTOVER_UTC
    ]

    rows: list[dict] = []

    for region in REGIONS:
        slug = REGION_SLUGS[region]
        legacy_column = f"{slug}_load_mw"

        for measure in ["meter_load_mw", "actual_load_mw"]:
            candidate = f"{slug}_{measure}"
            valid = overlap[[legacy_column, candidate]].dropna()
            difference = valid[candidate] - valid[legacy_column]

            rows.append(
                {
                    "region": region,
                    "candidate_measure": measure,
                    "hours": len(valid),
                    "mean_difference_mw": difference.mean(),
                    "mean_absolute_difference_mw": (
                        difference.abs().mean()
                    ),
                    "maximum_absolute_difference_mw": (
                        difference.abs().max()
                    ),
                    "correlation": (
                        valid[legacy_column].corr(valid[candidate])
                    ),
                    "exact_match_pct": (
                        np.isclose(
                            valid[legacy_column],
                            valid[candidate],
                            atol=1e-6,
                        ).mean()
                        * 100
                    ),
                    "decision": (
                        "audit_only; existing history retained through 2024"
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_canonical_regional_load(
    legacy: pd.DataFrame,
    load_chart: pd.DataFrame,
) -> pd.DataFrame:
    """Splice preserved history to observed post-2024 Meter Load."""

    historical = legacy.loc[
        legacy["timestamp_utc"] < CUTOVER_UTC
    ].copy()

    historical["regional_load_source"] = (
        "preserved_canonical_history"
    )
    historical["regional_load_definition"] = (
        "legacy_reported_region"
    )

    recent = load_chart.loc[
        load_chart["timestamp_utc"] >= CUTOVER_UTC
    ].copy()

    for region in REGIONS:
        slug = REGION_SLUGS[region]
        recent[f"{slug}_load_mw"] = (
            recent[f"{slug}_meter_load_mw"]
        )

    recent["regional_load_imputed"] = np.int8(0)
    recent["regional_load_source"] = "load_chart_full_data"
    recent["regional_load_definition"] = "meter_load"

    base_columns = [
        "timestamp_utc",
        *CANONICAL_REGION_COLUMNS,
        "regional_load_imputed",
        "regional_load_source",
        "regional_load_definition",
    ]

    canonical = pd.concat(
        [
            historical[base_columns],
            recent[base_columns],
        ],
        ignore_index=True,
    )

    diagnostics = load_chart.drop(
        columns=[
            column
            for column in CANONICAL_REGION_COLUMNS
            if column in load_chart.columns
        ],
        errors="ignore",
    )

    canonical = canonical.merge(
        diagnostics,
        on="timestamp_utc",
        how="left",
    )

    canonical["total_region_load_mw"] = canonical[
        CANONICAL_REGION_COLUMNS
    ].sum(
        axis=1,
        min_count=len(CANONICAL_REGION_COLUMNS),
    )

    return (
        canonical
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------
# Audits
# ---------------------------------------------------------------------

def audit_regional_load(
    canonical: pd.DataFrame,
    raw_clean: pd.DataFrame,
    load_chart: pd.DataFrame,
    legacy: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Audit source shape, accounting identities, continuity, and cutover."""

    rows: list[dict] = []
    add = partial(add_check, rows)

    add_duplicate_checks(rows, raw_clean)

    observed_regions = set(
        raw_clean["Region"].dropna().unique()
    )
    expected_regions = {
        *REGIONS,
        "Losses",
        "System Load",
    }

    add(
        "expected_regions_present",
        expected_regions.issubset(observed_regions),
        "; ".join(sorted(observed_regions)),
        "; ".join(sorted(expected_regions)),
    )

    add(
        "raw_timestamps_parse",
        raw_clean["timestamp_utc"].notna().all(),
        int(raw_clean["timestamp_utc"].isna().sum()),
        0,
    )

    six_actual = [
        f"{REGION_SLUGS[r]}_actual_load_mw"
        for r in REGIONS
    ]
    six_btf = [
        f"{REGION_SLUGS[r]}_btf_load_mw"
        for r in REGIONS
    ]
    six_meter = [
        f"{REGION_SLUGS[r]}_meter_load_mw"
        for r in REGIONS
    ]

    load_measure_columns = [
        *six_actual,
        *six_btf,
        *six_meter,
    ]

    add(
        "six_regions_complete_in_load_chart",
        not load_chart[
            load_measure_columns
        ].isna().any().any(),
        int(
            load_chart[
                load_measure_columns
            ].isna().sum().sum()
        ),
        0,
    )

    identity_errors = []

    for region in REGIONS:
        slug = REGION_SLUGS[region]

        error = (
            load_chart[f"{slug}_actual_load_mw"]
            - load_chart[f"{slug}_btf_load_mw"]
            - load_chart[f"{slug}_meter_load_mw"]
        ).abs()

        identity_errors.append(error)

    identity_frame = pd.concat(
        identity_errors,
        axis=1,
    )

    maximum_identity_error = (
        identity_frame.max().max()
    )
    identity_match_pct = (
        identity_frame.le(0.01).stack().mean()
        * 100
    )

    add(
        "actual_equals_meter_plus_btf",
        maximum_identity_error < 50.0,
        (
            f"within_0.01_mw={identity_match_pct:.3f}%; "
            f"max_abs_error={maximum_identity_error:.6g}"
        ),
        "maximum source discrepancy < 50 MW",
        severity="warning",
        notes=(
            "Most records satisfy Actual Load = Meter Load + BTF Load "
            "to rounding precision; a small set of AESO source rows differ."
        ),
    )

    system_error = load_chart[
        "system_meter_reconciliation_error_mw"
    ].abs()

    add(
        "six_region_meter_sum_reconciles_to_system_load",
        system_error.max(skipna=True) < 30.0,
        (
            f"mean_abs_error={system_error.mean():.6g}; "
            f"max_abs_error={system_error.max(skipna=True):.6g}"
        ),
        "max < 30 MW",
        severity="warning",
        notes=(
            "The AESO export contains small rounding or aggregation "
            "differences."
        ),
    )

    timestamps = pd.DatetimeIndex(
        canonical["timestamp_utc"]
    )

    expected_index = pd.date_range(
        timestamps.min(),
        timestamps.max(),
        freq="h",
        tz="UTC",
    )

    add(
        "canonical_rows_positive",
        len(canonical) > 0,
        len(canonical),
        "> 0",
    )

    add(
        "canonical_timestamps_unique",
        timestamps.is_unique,
        int(timestamps.duplicated().sum()),
        0,
    )

    add(
        "canonical_timestamps_monotonic",
        timestamps.is_monotonic_increasing,
        timestamps.is_monotonic_increasing,
        True,
    )

    add(
        "canonical_hourly_coverage",
        timestamps.equals(expected_index),
        len(timestamps),
        len(expected_index),
    )

    add(
        "canonical_region_values_complete",
        not canonical[
            CANONICAL_REGION_COLUMNS
        ].isna().any().any(),
        int(
            canonical[
                CANONICAL_REGION_COLUMNS
            ].isna().sum().sum()
        ),
        0,
    )

    add(
        "canonical_region_values_nonnegative",
        not canonical[
            CANONICAL_REGION_COLUMNS
        ].lt(0).any().any(),
        int(
            canonical[
                CANONICAL_REGION_COLUMNS
            ].lt(0).sum().sum()
        ),
        0,
    )

    retained = canonical.loc[
        canonical["timestamp_utc"] < CUTOVER_UTC
    ]

    retained_check = retained.merge(
        legacy[
            [
                "timestamp_utc",
                *CANONICAL_REGION_COLUMNS,
            ]
        ],
        on="timestamp_utc",
        suffixes=("", "_legacy"),
    )

    historical_max_error = max(
        (
            retained_check[column]
            - retained_check[f"{column}_legacy"]
        ).abs().max()
        for column in CANONICAL_REGION_COLUMNS
    )

    add(
        "pre_2025_values_unchanged",
        historical_max_error < 1e-9,
        f"max_abs_error={historical_max_error:.6g}",
        "0 MW",
    )

    recent = canonical.loc[
        canonical["timestamp_utc"] >= CUTOVER_UTC
    ]

    recent_max_error = max(
        (
            recent[
                f"{REGION_SLUGS[region]}_load_mw"
            ]
            - recent[
                f"{REGION_SLUGS[region]}_meter_load_mw"
            ]
        ).abs().max()
        for region in REGIONS
    )

    add(
        "post_2024_uses_observed_meter_load",
        recent_max_error < 1e-9,
        f"max_abs_error={recent_max_error:.6g}",
        "0 MW",
    )

    add(
        "no_frozen_extension",
        "area_load_frozen" not in canonical.columns,
        "area_load_frozen" in canonical.columns,
        False,
    )

    numeric = canonical.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    summary = (
        canonical[numeric]
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
        .reset_index(names="feature")
    )

    summary = summary.rename(
        columns={
            "1%": "p01",
            "50%": "median",
            "99%": "p99",
        }
    )

    summary["missing_count"] = (
        canonical[numeric]
        .isna()
        .sum()
        .to_numpy()
    )

    summary["missing_pct"] = (
        summary["missing_count"]
        / len(canonical)
        * 100
    )

    audit = pd.DataFrame(rows)

    return audit, summary, audit_passes(audit)


def build_source_summary(
    canonical: pd.DataFrame,
    raw_clean: pd.DataFrame,
    historical_source_path: Path,
) -> pd.DataFrame:
    """Summarize input provenance and the canonical source split."""

    return pd.DataFrame(
        [
            {
                "source_family": "preserved_canonical_history",
                "files": 1,
                "paths": str(historical_source_path),
                "canonical_start_utc": canonical.loc[
                    canonical["regional_load_source"].eq(
                        "preserved_canonical_history"
                    ),
                    "timestamp_utc",
                ].min(),
                "canonical_end_utc": canonical.loc[
                    canonical["regional_load_source"].eq(
                        "preserved_canonical_history"
                    ),
                    "timestamp_utc",
                ].max(),
            },
            {
                "source_family": "load_chart_full_data",
                "files": 1,
                "paths": str(RAW_LOAD_CHART_FILE),
                "raw_rows": len(raw_clean),
                "canonical_start_utc": canonical.loc[
                    canonical["regional_load_source"].eq(
                        "load_chart_full_data"
                    ),
                    "timestamp_utc",
                ].min(),
                "canonical_end_utc": canonical.loc[
                    canonical["regional_load_source"].eq(
                        "load_chart_full_data"
                    ),
                    "timestamp_utc",
                ].max(),
            },
        ]
    )


# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------

def process_area_load(overwrite: bool = False) -> dict:
    """Build, audit, and save the canonical regional-load product."""

    started = time.perf_counter()

    source_paths = [RAW_LOAD_CHART_FILE]
    code_paths = preprocessing_code_paths(
        Path(__file__)
    )

    manifest = build_manifest(
        dataset=DATASET_NAME,
        source_paths=source_paths,
        code_paths=code_paths,
        configuration={
            "cutover_utc": str(CUTOVER_UTC),
            "pre_cutover_source": (
                "existing canonical regional-load history"
            ),
            "post_cutover_measure": "Meter Load",
            "timestamp_policy": (
                "fixed MST converted to UTC"
            ),
        },
    )

    outputs = [
        OUTPUT_CSV,
        OUTPUT_PARQUET,
        AUDIT_FILE,
        SUMMARY_FILE,
        FILE_SUMMARY_FILE,
        OVERLAP_FILE,
    ]

    if (
        not overwrite
        and outputs_are_current(outputs, manifest)
    ):
        return {
            "dataset": DATASET_NAME,
            "status": "skipped_existing",
            "pass": True,
            "parquet_file": str(OUTPUT_PARQUET),
            "csv_file": str(OUTPUT_CSV),
        }

    try:
        legacy, historical_source_path = (
            load_legacy_regional_history()
        )

        raw = read_load_chart()
        raw_clean = clean_load_chart(raw)
        load_chart = reshape_load_chart(raw_clean)

        overlap = build_overlap_audit(
            legacy,
            load_chart,
        )

        canonical = build_canonical_regional_load(
            legacy,
            load_chart,
        )

        audit, summary, passed = (
            audit_regional_load(
                canonical,
                raw_clean,
                load_chart,
                legacy,
            )
        )

        source_summary = build_source_summary(
            canonical,
            raw_clean,
            historical_source_path,
        )

        artifacts = {
            AUDIT_FILE: audit,
            SUMMARY_FILE: summary,
            FILE_SUMMARY_FILE: source_summary,
            OVERLAP_FILE: overlap,
        }

        write_audit_artifacts(artifacts)

        if not passed:
            return {
                "dataset": DATASET_NAME,
                "status": "audit_failed",
                "pass": False,
                "audit_file": str(AUDIT_FILE),
                "processing_seconds": round(
                    time.perf_counter() - started,
                    3,
                ),
            }

        write_tabular_outputs(
            canonical,
            parquet_path=OUTPUT_PARQUET,
            csv_path=OUTPUT_CSV,
            manifest=manifest,
            provenance_artifacts=list(artifacts),
        )

        return {
            "dataset": DATASET_NAME,
            "status": "saved",
            "pass": True,
            "rows": len(canonical),
            "features": len(canonical.columns) - 1,
            "start": str(
                canonical["timestamp_utc"].min()
            ),
            "end": str(
                canonical["timestamp_utc"].max()
            ),
            "legacy_hours": int(
                canonical[
                    "regional_load_source"
                ].eq(
                    "preserved_canonical_history"
                ).sum()
            ),
            "new_source_hours": int(
                canonical[
                    "regional_load_source"
                ].eq(
                    "load_chart_full_data"
                ).sum()
            ),
            "imputed_hours": int(
                canonical[
                    "regional_load_imputed"
                ].sum()
            ),
            "parquet_file": str(
                OUTPUT_PARQUET
            ),
            "csv_file": str(
                OUTPUT_CSV
            ),
            "audit_file": str(
                AUDIT_FILE
            ),
            "overlap_file": str(
                OVERLAP_FILE
            ),
            "processing_seconds": round(
                time.perf_counter() - started,
                3,
            ),
        }

    except DuplicateConflictError as exc:
        write_audit_artifacts(
            {
                AUDIT_FILE: duplicate_failure_audit(
                    exc
                )
            }
        )

        return {
            "dataset": DATASET_NAME,
            "status": "audit_failed",
            "pass": False,
            "error": str(exc),
            "audit_file": str(AUDIT_FILE),
            "processing_seconds": round(
                time.perf_counter() - started,
                3,
            ),
        }

    except Exception as exc:
        return {
            "dataset": DATASET_NAME,
            "status": "error",
            "pass": False,
            "error": repr(exc),
            "processing_seconds": round(
                time.perf_counter() - started,
                3,
            ),
        }


def main() -> None:
    """Run regional-load preprocessing from the command line."""

    parser = argparse.ArgumentParser(
        description=(
            "Preprocess AESO regional load "
            "with a stable 2025 cutover."
        )
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()
    result = process_area_load(
        overwrite=args.overwrite
    )

    print("\n" + "=" * 80)
    print("REGIONAL LOAD PREPROCESSING RESULT")
    print("=" * 80)

    for key, value in result.items():
        print(f"{key}: {value}")

    print("=" * 80)

    if not result.get("pass", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()