# src/feature_engineering/market_features.py

"""
Build one canonical hourly Alberta market-feature dataset.

Inputs
------
The script searches for these preprocessed datasets:

1. P&A / price-load-gas data
   - pa_preprocessed.parquet
   - price_ail_gas_preprocessed.parquet
   - p_and_a_preprocessed.parquet

2. Intertie data
   - intertie_preprocessed.parquet
   - interties_preprocessed.parquet

3. Outage data
   - outages_preprocessed.parquet

Outputs
-------
Canonical feature output:

    data/processed/feature_engineering/market/market_features_hourly.parquet

Optional full CSV output:

    data/processed/feature_engineering/market/market_features_hourly.csv

Audit outputs:

    data/audits/feature_engineering/market_features_audit_checks.csv
    data/audits/feature_engineering/market_features_feature_summary.csv
    data/audits/feature_engineering/market_features_source_summary.csv

Run
---
Standard run:

    python src/feature_engineering/market_features.py

Overwrite existing outputs:

    python src/feature_engineering/market_features.py \
        --overwrite

Write CSV as well:

    python src/feature_engineering/market_features.py \
        --overwrite \
        --write-csv

Verbose logging:

    python src/feature_engineering/market_features.py \
        --verbose

Design
------
The P&A table is the hourly backbone. Intertie and outage data are left-joined
onto it. Hours outside a source's historical coverage remain missing and are
identified with explicit source-availability flags.

The file keeps feature logic modular. Add or remove functions from
FEATURE_BUILDERS without changing the input/output pipeline.

Information timing
------------------
This dataset intentionally contains targets, contemporaneous/ex-post fields,
and historical predictors. It is not a ready-made forecasting design matrix.
The feature summary labels every numeric column as target, target-derived,
contemporaneous, or historical so modeling code can enforce an as-of policy.
"""

# ============================================================================
# Imports
# ============================================================================

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

try:
    from .shared import (
        add_changes_through_current_hour as add_changes,
        add_check,
        add_lags,
        add_prior_rolling_statistics as add_rolling_statistics,
        apply_feature_builders as run_feature_builders,
        audit_passed,
        build_manifest,
        classify_feature_timing,
        configure_logging,
        ensure_directories,
        feature_code_paths,
        ensure_src_on_path,
        existing_outputs_satisfy_request as outputs_satisfy_request,
        find_first_column,
        load_parquet_table,
        merge_hourly_sources,
        numeric_feature_summary,
        numericize_except_timestamp,
        output_is_current,
        read_existing_parquet,
        require_columns,
        resolve_file,
        safe_divide,
        save_feature_outputs as write_feature_outputs,
        save_tables,
        source_summary,
        write_manifest,
    )
except ImportError:  # Support direct execution of this file.
    from shared import (
        add_changes_through_current_hour as add_changes,
        add_check,
        add_lags,
        add_prior_rolling_statistics as add_rolling_statistics,
        apply_feature_builders as run_feature_builders,
        audit_passed,
        build_manifest,
        classify_feature_timing,
        configure_logging,
        ensure_directories,
        feature_code_paths,
        ensure_src_on_path,
        existing_outputs_satisfy_request as outputs_satisfy_request,
        find_first_column,
        load_parquet_table,
        merge_hourly_sources,
        numeric_feature_summary,
        numericize_except_timestamp,
        output_is_current,
        read_existing_parquet,
        require_columns,
        resolve_file,
        safe_divide,
        save_feature_outputs as write_feature_outputs,
        save_tables,
        source_summary,
        write_manifest,
    )

ensure_src_on_path(__file__)


# ============================================================================
# Logging
# ============================================================================

LOGGER = logging.getLogger(__name__)


# ============================================================================
# Project paths
# ============================================================================

from config import (
    PREPROCESSING_DIR,
    FEATURES_DIR,
    FEATURE_ENGINEERING_AUDITS_DIR,
    PROJECT_ROOT,
)


OUTPUT_DIR = FEATURES_DIR / "market"

OUTPUT_PARQUET = OUTPUT_DIR / "market_features_hourly.parquet"
OUTPUT_CSV = OUTPUT_DIR / "market_features_hourly.csv"

# Backward-compatible alias retained for code that imports OUTPUT_FILE.
OUTPUT_FILE = OUTPUT_PARQUET

AUDIT_DIR = FEATURE_ENGINEERING_AUDITS_DIR

AUDIT_FILE = AUDIT_DIR / "market_features_audit_checks.csv"
FEATURE_SUMMARY_FILE = AUDIT_DIR / "market_features_feature_summary.csv"
SOURCE_SUMMARY_FILE = AUDIT_DIR / "market_features_source_summary.csv"


PA_FILE_CANDIDATES = [
    PREPROCESSING_DIR / "pa_hourly_preprocessed.parquet",
    PREPROCESSING_DIR / "pa_hourly.parquet",
]


INTERTIE_FILE_CANDIDATES = [
    PREPROCESSING_DIR / "interties_hour_ahead.parquet",
    PREPROCESSING_DIR / "interties_hour_ahead_preprocessed.parquet",
]


OUTAGE_FILE_CANDIDATES = [
    PREPROCESSING_DIR / "outages_preprocessed.parquet",
]


# ============================================================================
# Configuration
# ============================================================================

LOCAL_TIMEZONE = "America/Edmonton"
DATASET_NAME = "market_features"
FEATURE_INFORMATION_POLICY = "mixed_ex_post_and_historical"

REQUIRED_BACKBONE_COLUMNS = [
    "timestamp_utc",
    "ail_mw",
    "pool_price",
    "gas_price",
    "spark_spread",
    "pa_available",
]

PRICE_THRESHOLDS = {
    "price_above_100": 100.0,
    "price_above_300": 300.0,
    "price_above_500": 500.0,
    "price_above_900": 900.0,
}

PRICE_LAGS = [1, 2, 3, 6, 12, 24, 48, 168]
LOAD_LAGS = [1, 3, 6, 12, 24, 48, 168]
INTERTIE_LAGS = [1, 3, 6, 12, 24]
OUTAGE_LAGS = [1, 3, 6, 12, 24, 48]

ROLLING_WINDOWS = [3, 6, 12, 24, 72, 168]


# ============================================================================
# Source loading and standardization
# ============================================================================

def load_pa_data(
    path: Path,
) -> pd.DataFrame:
    """Load canonical price, load, gas, and spread fields."""
    frame = numericize_except_timestamp(
        load_parquet_table(
            path,
            "P&A",
        )
    )

    aliases = {
        "ail_mw": [
            "ail",
            "actual_ail",
            "ail_mw",
        ],
        "pool_price": [
            "pool_price",
            "pool_price_cad_mwh",
            "price",
            "actual_pool_price",
        ],
        "gas_price": [
            "gas_price",
            "gas_price_cad_gj",
            "aeco_price",
            "aeco_price_cad_gj",
        ],
        "spark_spread": [
            "spark_spread",
        ],
    }

    rename = {}

    for canonical, candidates in aliases.items():
        observed = find_first_column(
            frame.columns,
            candidates,
        )

        if observed is not None:
            rename[observed] = canonical

    frame = frame.rename(
        columns=rename
    )

    require_columns(
        frame,
        [
            "timestamp_utc",
            "ail_mw",
            "pool_price",
            "gas_price",
        ],
        "P&A",
    )

    if "spark_spread" not in frame.columns:
        frame["spark_spread"] = np.nan

    keep = [
        "timestamp_utc",
        "ail_mw",
        "pool_price",
        "gas_price",
        "spark_spread",
    ]

    frame = frame[keep].copy()
    frame["pa_available"] = 1

    return frame


def load_intertie_data(
    path: Path,
) -> pd.DataFrame:
    """Load validated non-negative directional intertie flows."""
    frame = numericize_except_timestamp(
        load_parquet_table(
            path,
            "intertie",
        )
    )

    import_columns = [
        "import_bc",
        "import_mt",
        "import_sk",
    ]

    export_columns = [
        "export_bc",
        "export_mt",
        "export_sk",
    ]

    directional_columns = (
        import_columns
        + export_columns
    )

    missing_directional_columns = sorted(
        set(directional_columns)
        - set(frame.columns)
    )

    if missing_directional_columns:
        raise ValueError(
            "Intertie data is missing required directional columns: "
            f"{missing_directional_columns}"
        )

    negative_counts = {
        column: int(
            frame[column]
            .lt(0)
            .sum()
        )
        for column in directional_columns
        if frame[column].lt(0).any()
    }

    if negative_counts:
        raise ValueError(
            "Preprocessed intertie data still contains negative "
            "directional flows. Signed-flow correction must occur "
            "in the intertie preprocessing pipeline. "
            f"Negative counts: {negative_counts}"
        )

    LOGGER.info(
        "Validated non-negative directional intertie flows."
    )

    frame["intertie_available"] = 1

    return frame


def load_outage_data(
    path: Path,
) -> pd.DataFrame:
    """Load hourly outage fields and standardize the total name."""
    frame = numericize_except_timestamp(
        load_parquet_table(
            path,
            "outages",
        )
    )

    if "total_outage" in frame.columns:
        frame = frame.rename(
            columns={
                "total_outage": "total_outage_mw",
            }
        )

    frame["outage_available"] = 1

    return frame


# ============================================================================
# Intertie features
# ============================================================================

def identify_directional_columns(
    columns: Iterable[str],
    direction: str,
) -> list[str]:
    """Find individual import or export columns without aggregates."""

    matches = []

    for column in columns:
        lower = column.lower()

        if column == "timestamp_utc":
            continue

        if lower.endswith("_raw"):
            continue

        if "total" in lower or "net_" in lower:
            continue

        if (
            lower.startswith(f"{direction}_")
            or lower.endswith(f"_{direction}")
            or f"_{direction}_" in lower
        ):
            matches.append(column)

    return sorted(matches)


def add_intertie_aggregate_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Add total, net, lagged, and change interchange features."""

    output = frame.copy()

    import_columns = identify_directional_columns(
        output.columns,
        "import",
    )

    export_columns = identify_directional_columns(
        output.columns,
        "export",
    )

    # Directional columns have already been corrected to non-negative flows
    # by preprocessing, so their sums have a direct physical interpretation.
    if import_columns:
        output["total_imports_mw"] = output[
            import_columns
        ].sum(
            axis=1,
            min_count=1,
        )

    if export_columns:
        output["total_exports_mw"] = output[
            export_columns
        ].sum(
            axis=1,
            min_count=1,
        )

    # Positive net imports mean Alberta is receiving more energy than it is
    # exporting; gross interchange measures total cross-border activity.
    if {
        "total_imports_mw",
        "total_exports_mw",
    }.issubset(output.columns):
        output["net_imports_mw"] = (
            output["total_imports_mw"]
            - output["total_exports_mw"]
        )

        output["gross_interchange_mw"] = (
            output["total_imports_mw"]
            + output["total_exports_mw"]
        )

        output["is_net_importing"] = (
            output["net_imports_mw"] > 0
        ).astype("int8")

        output["is_net_exporting"] = (
            output["net_imports_mw"] < 0
        ).astype("int8")

    # Lags are historical; change fields include the current observed hour.
    for column in [
        "total_imports_mw",
        "total_exports_mw",
        "net_imports_mw",
    ]:
        output = add_lags(
            output,
            column,
            INTERTIE_LAGS,
        )

        output = add_changes(
            output,
            column,
            [1, 3, 6, 24],
        )

    return output


# ============================================================================
# Outage features
# ============================================================================

def add_outage_aggregate_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Add fuel-group outage aggregates and temporal features."""

    output = frame.copy()

    # Group fuel-level outage fields into interpretable supply categories.
    outage_columns = [
        column
        for column in output.columns
        if column.endswith("_outage")
    ]

    thermal_keywords = [
        "coal",
        "cogeneration",
        "combined_cycle",
        "dual_fuel",
        "gas_fired_steam",
        "simple_cycle",
        "other",
    ]

    renewable_keywords = [
        "wind",
        "solar",
        "hydro",
    ]

    storage_keywords = [
        "storage",
    ]

    thermal_columns = [
        column
        for column in outage_columns
        if any(
            keyword in column
            for keyword in thermal_keywords
        )
    ]

    renewable_columns = [
        column
        for column in outage_columns
        if any(
            keyword in column
            for keyword in renewable_keywords
        )
    ]

    storage_columns = [
        column
        for column in outage_columns
        if any(
            keyword in column
            for keyword in storage_keywords
        )
    ]

    if thermal_columns:
        output["thermal_outage_mw"] = output[
            thermal_columns
        ].sum(
            axis=1,
            min_count=1,
        )

    if renewable_columns:
        output["renewable_outage_mw"] = output[
            renewable_columns
        ].sum(
            axis=1,
            min_count=1,
        )

    if storage_columns:
        output["storage_outage_mw"] = output[
            storage_columns
        ].sum(
            axis=1,
            min_count=1,
        )

    # Prefer the source-reported total when available; otherwise reconstruct it
    # from the complete set of fuel-level outage fields.
    if "total_outage_mw" not in output.columns and outage_columns:
        output["total_outage_mw"] = output[
            outage_columns
        ].sum(
            axis=1,
            min_count=1,
        )

    for column in [
        "total_outage_mw",
        "thermal_outage_mw",
        "renewable_outage_mw",
    ]:
        output = add_lags(
            output,
            column,
            OUTAGE_LAGS,
        )

        output = add_changes(
            output,
            column,
            [1, 3, 6, 24],
        )

    return output


# ============================================================================
# P&A features
# ============================================================================

def add_price_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Add target-derived price diagnostics and historical price features."""

    output = frame.copy()

    if "pool_price" not in output.columns:
        return output

    # These flags and the log transform use the current pool price. They are
    # descriptive target-derived fields, not forecasting-safe predictors.
    output["price_negative"] = (
        output["pool_price"] < 0
    ).astype("int8")

    for name, threshold in PRICE_THRESHOLDS.items():
        output[name] = (
            output["pool_price"]
            >= threshold
        ).astype("int8")

    output["price_at_cap"] = (
        output["pool_price"]
        >= 999.0
    ).astype("int8")

    output["log1p_nonnegative_price"] = np.log1p(
        output["pool_price"].clip(lower=0.0)
    )

    # Lag and prior-window features are safe historical summaries; current
    # minus lagged changes remain target-derived at timestamp t.
    output = add_lags(
        output,
        "pool_price",
        PRICE_LAGS,
    )

    output = add_changes(
        output,
        "pool_price",
        [1, 3, 6, 12, 24],
    )

    output = add_rolling_statistics(
        output,
        "pool_price",
        [3, 6, 24, 72, 168],
    )

    return output


def add_load_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Add contemporaneous load changes and historical load features."""

    output = frame.copy()

    if "ail_mw" not in output.columns:
        return output

    # Separate known historical load from ramps that use current-hour AIL.
    output = add_lags(
        output,
        "ail_mw",
        LOAD_LAGS,
    )

    output = add_changes(
        output,
        "ail_mw",
        [1, 3, 6, 12, 24],
        prefix="ail_ramp",
    )

    output = add_rolling_statistics(
        output,
        "ail_mw",
        [3, 6, 24, 72, 168],
    )

    output["ail_above_prior_24h_mean"] = (
        output["ail_mw"]
        > output["ail_mw_mean_prior_24h"]
    ).astype("int8")

    return output


def add_gas_and_spread_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Add historical gas-price and spark-spread features."""

    output = frame.copy()

    if "gas_price" in output.columns:
        output = add_lags(
            output,
            "gas_price",
            [1, 24, 168],
        )

        output = add_changes(
            output,
            "gas_price",
            [24, 168],
        )

    if "spark_spread" in output.columns:
        output = add_lags(
            output,
            "spark_spread",
            [1, 3, 6, 24, 168],
        )

        output = add_rolling_statistics(
            output,
            "spark_spread",
            [24, 72, 168],
        )

    return output


# ============================================================================
# Cross-source market features
# ============================================================================

def add_market_pressure_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Add interpretable cross-source supply-pressure proxies."""

    output = frame.copy()

    # Imports reduce the amount of demand that must be served domestically.
    if {
        "ail_mw",
        "net_imports_mw",
    }.issubset(output.columns):
        output["import_adjusted_load_mw"] = (
            output["ail_mw"]
            - output["net_imports_mw"]
        )

    # Outages increase the demand-equivalent pressure on available supply.
    if {
        "ail_mw",
        "total_outage_mw",
    }.issubset(output.columns):
        output["outage_adjusted_load_mw"] = (
            output["ail_mw"]
            + output["total_outage_mw"]
        )

    # Thermal outages and net imports are combined into a compact scarcity
    # proxy; this is explanatory rather than a physical reserve calculation.
    if {
        "ail_mw",
        "thermal_outage_mw",
        "net_imports_mw",
    }.issubset(output.columns):
        output["supply_pressure_proxy_mw"] = (
            output["ail_mw"]
            + output["thermal_outage_mw"]
            - output["net_imports_mw"]
        )

        output = add_lags(
            output,
            "supply_pressure_proxy_mw",
            [1, 3, 6, 24],
        )

        output = add_changes(
            output,
            "supply_pressure_proxy_mw",
            [1, 3, 6, 24],
        )

        output = add_rolling_statistics(
            output,
            "supply_pressure_proxy_mw",
            [24, 72, 168],
        )

    # Scale level variables by AIL to make hours with different system demand
    # more comparable. Division by zero produces missing values.
    if {
        "total_imports_mw",
        "ail_mw",
    }.issubset(output.columns):
        output["imports_as_share_of_ail"] = np.divide(
            output["total_imports_mw"],
            output["ail_mw"],
            out=np.full(
                len(output),
                np.nan,
                dtype=float,
            ),
            where=output["ail_mw"].to_numpy(dtype=float) != 0,
        )

    if {
        "total_outage_mw",
        "ail_mw",
    }.issubset(output.columns):
        output["outages_as_share_of_ail"] = np.divide(
            output["total_outage_mw"],
            output["ail_mw"],
            out=np.full(
                len(output),
                np.nan,
                dtype=float,
            ),
            where=output["ail_mw"].to_numpy(dtype=float) != 0,
        )

    return output


FEATURE_BUILDERS: list[
    Callable[
        [pd.DataFrame],
        pd.DataFrame,
    ]
] = [
    add_intertie_aggregate_features,
    add_outage_aggregate_features,
    add_price_features,
    add_load_features,
    add_gas_and_spread_features,
    add_market_pressure_features,
]


# ============================================================================
# Merge and audit
# ============================================================================

def merge_sources(
    pa: pd.DataFrame,
    intertie: pd.DataFrame,
    outages: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join optional sources to the complete P&A backbone."""
    return merge_hourly_sources(
        pa,
        [intertie, outages],
        ["pa_available", "intertie_available", "outage_available"],
    )


def audit_market_features(
    frame: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    bool,
]:
    """Audit the market table and build its numeric feature catalog."""
    rows: list[dict] = []

    timestamps = pd.DatetimeIndex(
        pd.to_datetime(
            frame["timestamp_utc"],
            utc=True,
        )
    )

    expected = pd.date_range(
        timestamps.min(),
        timestamps.max(),
        freq="h",
        tz="UTC",
    )

    add_check(
        rows,
        "row_count_positive",
        len(frame) > 0,
        len(frame),
        "> 0",
    )

    add_check(
        rows,
        "timestamps_unique",
        not timestamps.has_duplicates,
        int(timestamps.duplicated().sum()),
        0,
    )

    add_check(
        rows,
        "timestamps_monotonic",
        timestamps.is_monotonic_increasing,
        timestamps.is_monotonic_increasing,
        True,
    )

    add_check(
        rows,
        "complete_pa_backbone",
        len(expected.difference(timestamps)) == 0,
        len(expected.difference(timestamps)),
        0,
    )

    add_check(
        rows,
        "pool_price_complete",
        frame["pool_price"].notna().all(),
        int(frame["pool_price"].isna().sum()),
        0,
    )

    add_check(
        rows,
        "ail_complete",
        frame["ail_mw"].notna().all(),
        int(frame["ail_mw"].isna().sum()),
        0,
    )

    add_check(
        rows,
        "gas_price_complete",
        frame["gas_price"].notna().all(),
        int(frame["gas_price"].isna().sum()),
        0,
    )

    for flag in [
        "pa_available",
        "intertie_available",
        "outage_available",
    ]:
        invalid = int(
            (~frame[flag].isin([0, 1])).sum()
        )

        add_check(
            rows,
            f"{flag}_binary",
            invalid == 0,
            invalid,
            0,
        )

        add_check(
            rows,
            f"{flag}_hours",
            True,
            int(frame[flag].sum()),
            "recorded",
            severity="info",
        )

    if "total_outage_mw" in frame.columns:
        negative = int(
            frame["total_outage_mw"]
            .lt(0)
            .sum()
        )

        add_check(
            rows,
            "total_outage_non_negative",
            negative == 0,
            negative,
            0,
            severity="warning",
        )

    # Check every cleaned directional intertie column directly.
    #
    # These are error-level checks because negative values should have
    # already been reclassified to the opposite direction.
    directional_intertie_columns = [
        "import_bc",
        "import_mt",
        "import_sk",
        "export_bc",
        "export_mt",
        "export_sk",
    ]

    for column in directional_intertie_columns:
        if column not in frame.columns:
            add_check(
                rows,
                f"{column}_present",
                False,
                "missing",
                "present",
            )
            continue

        negative_count = int(
            frame[column]
            .lt(0)
            .sum()
        )

        add_check(
            rows,
            f"{column}_non_negative",
            negative_count == 0,
            negative_count,
            0,
        )

    if "total_imports_mw" in frame.columns:
        negative = int(
            frame["total_imports_mw"]
            .lt(0)
            .sum()
        )

        add_check(
            rows,
            "total_imports_non_negative",
            negative == 0,
            negative,
            0,
        )

    if "total_exports_mw" in frame.columns:
        negative = int(
            frame["total_exports_mw"]
            .lt(0)
            .sum()
        )

        add_check(
            rows,
            "total_exports_non_negative",
            negative == 0,
            negative,
            0,
        )

    summary = numeric_feature_summary(
        frame,
        timing_classifier=lambda column: classify_feature_timing(
            column,
            target_columns={"pool_price"},
            target_derived_prefixes={
                "price_",
                "log1p_nonnegative_price",
                "pool_price_change_",
                "spark_spread",
            },
        ),
    )

    audit = pd.DataFrame(rows)
    return audit, summary, audit_passed(audit)


def print_report(
    frame: pd.DataFrame,
    audit: pd.DataFrame,
    passed: bool,
) -> None:
    """Print a compact human-readable market audit report."""
    timestamps = pd.DatetimeIndex(
        frame["timestamp_utc"]
    )

    failed = audit.loc[
        ~audit["pass"]
    ]

    print("\n" + "=" * 80)
    print("MARKET FEATURES AUDIT")
    print("=" * 80)
    print(f"Overall pass       : {passed}")
    print(f"Rows               : {len(frame):,}")
    print(f"Columns            : {len(frame.columns):,}")
    print(f"Start UTC          : {timestamps.min()}")
    print(f"End UTC            : {timestamps.max()}")
    print(f"Intertie coverage  : {int(frame['intertie_available'].sum()):,}")
    print(f"Outage coverage    : {int(frame['outage_available'].sum()):,}")

    print("\nFailed checks:")

    if failed.empty:
        print("  None")
    else:
        for _, row in failed.iterrows():
            print(
                f"  - {row['check']} [{row['severity']}] "
                f"observed={row['observed']} expected={row['expected']}"
            )

    print("=" * 80)


# ============================================================================
# Pipeline
# ============================================================================

def build_market_features(
    overwrite: bool = False,
    write_csv: bool = False,
) -> dict[str, Any]:
    """Load, merge, engineer, audit, and save market features."""

    started = time.perf_counter()

    LOGGER.info("Starting market-feature pipeline.")
    LOGGER.debug("Project root: %s", PROJECT_ROOT)
    LOGGER.debug("Preprocessing directory: %s", PREPROCESSING_DIR)
    LOGGER.debug("Feature output directory: %s", OUTPUT_DIR)
    LOGGER.debug("Audit output directory: %s", AUDIT_DIR)

    ensure_directories(OUTPUT_DIR, AUDIT_DIR)

    pa_path = resolve_file(PA_FILE_CANDIDATES, "P&A")
    intertie_path = resolve_file(INTERTIE_FILE_CANDIDATES, "intertie")
    outage_path = resolve_file(OUTAGE_FILE_CANDIDATES, "outage")
    expected_manifest = build_manifest(
        dataset=DATASET_NAME,
        source_paths=[pa_path, intertie_path, outage_path],
        code_paths=feature_code_paths(Path(__file__)),
        configuration={
            "feature_information_policy": FEATURE_INFORMATION_POLICY,
            "price_lags": PRICE_LAGS,
            "load_lags": LOAD_LAGS,
            "rolling_windows": ROLLING_WINDOWS,
        },
    )

    # Skip only when all explicitly requested feature outputs already exist.
    if (
        not overwrite
        and outputs_satisfy_request(
            OUTPUT_PARQUET,
            OUTPUT_CSV,
            write_csv=write_csv,
            expected_manifest=expected_manifest,
            required_artifacts=[
                AUDIT_FILE,
                FEATURE_SUMMARY_FILE,
                SOURCE_SUMMARY_FILE,
            ],
        )
    ):
        LOGGER.info(
            "Requested market-feature outputs already exist. "
            "Use --overwrite to rebuild them."
        )

        return {
            "dataset": DATASET_NAME,
            "status": "skipped_existing",
            "pass": True,
            "rows": None,
            "columns": None,
            "parquet_file": str(OUTPUT_PARQUET),
            "csv_file": (
                str(OUTPUT_CSV)
                if write_csv
                else "not requested"
            ),
            "audit_file": (
                str(AUDIT_FILE)
                if AUDIT_FILE.exists()
                else "not available"
            ),
            "feature_summary_file": (
                str(FEATURE_SUMMARY_FILE)
                if FEATURE_SUMMARY_FILE.exists()
                else "not available"
            ),
            "source_summary_file": (
                str(SOURCE_SUMMARY_FILE)
                if SOURCE_SUMMARY_FILE.exists()
                else "not available"
            ),
            "processing_seconds": round(
                time.perf_counter()
                - started,
                3,
            ),
        }

    # If the canonical Parquet already exists and CSV is the only missing
    # requested representation, create it directly without rebuilding.
    if (
        not overwrite
        and output_is_current(OUTPUT_PARQUET, expected_manifest)
        and write_csv
        and not OUTPUT_CSV.exists()
    ):
        frame = read_existing_parquet(OUTPUT_PARQUET)

        LOGGER.info(
            "Creating missing CSV from existing canonical Parquet."
        )

        frame.to_csv(
            OUTPUT_CSV,
            index=False,
        )
        write_manifest(OUTPUT_CSV, expected_manifest)

        return {
            "dataset": DATASET_NAME,
            "status": "csv_created_from_existing_parquet",
            "pass": True,
            "rows": len(frame),
            "columns": len(frame.columns),
            "start_utc": str(
                frame["timestamp_utc"].min()
            ),
            "end_utc": str(
                frame["timestamp_utc"].max()
            ),
            "parquet_file": str(OUTPUT_PARQUET),
            "csv_file": str(OUTPUT_CSV),
            "audit_file": (
                str(AUDIT_FILE)
                if AUDIT_FILE.exists()
                else "not available"
            ),
            "feature_summary_file": (
                str(FEATURE_SUMMARY_FILE)
                if FEATURE_SUMMARY_FILE.exists()
                else "not available"
            ),
            "source_summary_file": (
                str(SOURCE_SUMMARY_FILE)
                if SOURCE_SUMMARY_FILE.exists()
                else "not available"
            ),
            "processing_seconds": round(
                time.perf_counter()
                - started,
                3,
            ),
        }

    # ------------------------------------------------------------------------
    # Source resolution and loading
    # ------------------------------------------------------------------------

    pa = load_pa_data(
        pa_path
    )

    intertie = load_intertie_data(
        intertie_path
    )

    outages = load_outage_data(
        outage_path
    )

    source_summary_frame = pd.DataFrame(
        [
            source_summary(
                "pa",
                pa,
                pa_path,
            ),
            source_summary(
                "intertie",
                intertie,
                intertie_path,
            ),
            source_summary(
                "outages",
                outages,
                outage_path,
            ),
        ]
    )

    LOGGER.info(
        "Loaded sources: P&A=%s rows, intertie=%s rows, outages=%s rows.",
        f"{len(pa):,}",
        f"{len(intertie):,}",
        f"{len(outages):,}",
    )

    # ------------------------------------------------------------------------
    # Merge and feature construction
    # ------------------------------------------------------------------------

    LOGGER.info(
        "Merging intertie and outage data onto the P&A hourly backbone."
    )

    master = merge_sources(
        pa,
        intertie,
        outages,
    )

    LOGGER.info(
        "Applying %s market feature builders.",
        len(FEATURE_BUILDERS),
    )

    master = run_feature_builders(master, FEATURE_BUILDERS)

    LOGGER.info(
        "Market feature table constructed with %s rows and %s columns.",
        f"{len(master):,}",
        f"{len(master.columns):,}",
    )

    # ------------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------------

    audit, feature_summary, passed = audit_market_features(
        master
    )

    print_report(
        master,
        audit,
        passed,
    )

    # Audit files are retained even when the feature audit fails.
    save_tables(
        {
            AUDIT_FILE: audit,
            FEATURE_SUMMARY_FILE: feature_summary,
            SOURCE_SUMMARY_FILE: source_summary_frame,
        },
        {
            AUDIT_FILE: "market-feature audit checks",
            FEATURE_SUMMARY_FILE: "market-feature numeric summary",
            SOURCE_SUMMARY_FILE: "market-feature source summary",
        },
    )

    if not passed:
        LOGGER.error(
            "Market-feature audit failed. "
            "Canonical feature outputs were not written."
        )

        return {
            "dataset": DATASET_NAME,
            "status": "audit_failed",
            "pass": False,
            "rows": len(master),
            "columns": len(master.columns),
            "start_utc": str(
                master["timestamp_utc"].min()
            ),
            "end_utc": str(
                master["timestamp_utc"].max()
            ),
            "pa_file": str(pa_path),
            "intertie_file": str(intertie_path),
            "outage_file": str(outage_path),
            "parquet_file": "not written",
            "csv_file": "not written",
            "audit_file": str(AUDIT_FILE),
            "feature_summary_file": str(FEATURE_SUMMARY_FILE),
            "source_summary_file": str(SOURCE_SUMMARY_FILE),
            "processing_seconds": round(
                time.perf_counter()
                - started,
                3,
            ),
        }

    # ------------------------------------------------------------------------
    # Canonical feature outputs
    # ------------------------------------------------------------------------

    write_feature_outputs(
        master,
        OUTPUT_PARQUET,
        OUTPUT_CSV,
        write_csv,
        "market-feature",
        manifest=expected_manifest,
    )
    for artifact in [AUDIT_FILE, FEATURE_SUMMARY_FILE, SOURCE_SUMMARY_FILE]:
        write_manifest(artifact, expected_manifest)
    provenance_file = write_manifest(OUTPUT_PARQUET, expected_manifest)

    processing_seconds = round(
        time.perf_counter()
        - started,
        3,
    )

    LOGGER.info(
        "Market-feature pipeline completed successfully in %.3f seconds.",
        processing_seconds,
    )

    return {
        "dataset": DATASET_NAME,
        "status": "saved",
        "pass": True,
        "rows": len(master),
        "columns": len(master.columns),
        "start_utc": str(
            master["timestamp_utc"].min()
        ),
        "end_utc": str(
            master["timestamp_utc"].max()
        ),
        "pa_hours": int(
            master["pa_available"].sum()
        ),
        "intertie_hours": int(
            master["intertie_available"].sum()
        ),
        "outage_hours": int(
            master["outage_available"].sum()
        ),
        "pa_file": str(pa_path),
        "intertie_file": str(intertie_path),
        "outage_file": str(outage_path),
        "parquet_file": str(OUTPUT_PARQUET),
        "csv_file": (
            str(OUTPUT_CSV)
            if write_csv
            else "not requested"
        ),
        "audit_file": str(AUDIT_FILE),
        "feature_summary_file": str(FEATURE_SUMMARY_FILE),
        "source_summary_file": str(SOURCE_SUMMARY_FILE),
        "manifest_file": str(provenance_file),
        "processing_seconds": processing_seconds,
    }


# ============================================================================
# CLI
# ============================================================================

def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Build canonical hourly Alberta market features "
            "from P&A, intertie, and outage data."
        )
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Rebuild and overwrite existing market-feature outputs."
        ),
    )

    parser.add_argument(
        "--write-csv",
        action="store_true",
        help=(
            "Also write the full market feature table to CSV."
        ),
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Enable verbose DEBUG-level logging."
        ),
    )

    return parser


def print_pipeline_result(
    result: dict[str, Any],
) -> None:
    """Print the final pipeline result dictionary."""

    print(
        "\n"
        + "=" * 80
    )

    print(
        "MARKET FEATURE RESULT"
    )

    print(
        "=" * 80
    )

    for key, value in result.items():
        print(
            f"{key}: {value}"
        )

    print(
        "=" * 80
    )


def main() -> None:
    """Run the market-feature pipeline from the command line."""

    parser = build_argument_parser()

    args = parser.parse_args()

    configure_logging(
        verbose=args.verbose
    )

    try:
        result = build_market_features(
            overwrite=args.overwrite,
            write_csv=args.write_csv,
        )

    except Exception:
        LOGGER.exception(
            "Market-feature pipeline terminated with an unexpected error."
        )
        raise

    print_pipeline_result(
        result
    )

    if not result.get(
        "pass",
        False,
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
