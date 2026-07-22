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

    data/features/market/market_features_hourly.parquet

Optional full CSV output:

    data/features/market/market_features_hourly.csv

Audit outputs:

    data/audits/market_features_audit_checks.csv
    data/audits/market_features_feature_summary.csv
    data/audits/market_features_source_summary.csv

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
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd


# ============================================================================
# Logging
# ============================================================================

LOGGER = logging.getLogger(__name__)


def configure_logging(
    verbose: bool = False,
) -> None:
    """Configure console logging for the pipeline."""

    logging.basicConfig(
        level=(
            logging.DEBUG
            if verbose
            else logging.INFO
        ),
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


# ============================================================================
# Paths
# ============================================================================

# This file is expected to live at:
#
#     PROJECT_ROOT/src/feature_engineering/market_features.py
#
# parents[0] -> feature_engineering
# parents[1] -> src
# parents[2] -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREPROCESSING_DIR = PROJECT_ROOT / "data" / "preprocessing"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
OUTPUT_DIR = FEATURES_DIR / "market"
AUDIT_DIR = PROJECT_ROOT / "data" / "audits"

OUTPUT_PARQUET = OUTPUT_DIR / "market_features_hourly.parquet"
OUTPUT_CSV = OUTPUT_DIR / "market_features_hourly.csv"
CALENDAR_FILE = (FEATURES_DIR / "calendar" / "calendar_features_hourly.parquet")

# Backward-compatible alias retained for code that imports OUTPUT_FILE.
OUTPUT_FILE = OUTPUT_PARQUET

AUDIT_FILE = AUDIT_DIR / "market_features_audit_checks.csv"
FEATURE_SUMMARY_FILE = AUDIT_DIR / "market_features_feature_summary.csv"
SOURCE_SUMMARY_FILE = AUDIT_DIR / "market_features_source_summary.csv"

PA_FILE_CANDIDATES = [
    PREPROCESSING_DIR / "pa_hourly_preprocessed.parquet",
    PREPROCESSING_DIR / "pa_preprocessed.parquet",
    PREPROCESSING_DIR / "price_ail_gas_preprocessed.parquet",
    PREPROCESSING_DIR / "p_and_a_preprocessed.parquet",
]

INTERTIE_FILE_CANDIDATES = [
    PREPROCESSING_DIR / "interties_hour_ahead.parquet",
    PREPROCESSING_DIR / "interties_hour_ahead_preprocessed.parquet",
    PREPROCESSING_DIR / "intertie_preprocessed.parquet",
    PREPROCESSING_DIR / "interties_preprocessed.parquet",
]

OUTAGE_FILE_CANDIDATES = [
    PREPROCESSING_DIR / "outages_preprocessed.parquet",
]


# ============================================================================
# Configuration
# ============================================================================

LOCAL_TIMEZONE = "America/Edmonton"
DATASET_NAME = "market_features"

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
# General helpers
# ============================================================================

def ensure_output_directories() -> None:
    """Create feature and audit output directories if needed."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def add_check(
    rows: list[dict],
    check: str,
    passed: bool,
    observed=None,
    expected=None,
    severity: str = "error",
    notes: str = "",
) -> None:
    rows.append(
        {
            "check": check,
            "pass": bool(passed),
            "severity": severity,
            "observed": observed,
            "expected": expected,
            "notes": notes,
        }
    )


def normalize_column_name(column: str) -> str:
    return (
        str(column)
        .strip()
        .lower()
        .replace("\ufeff", "")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output.columns = [
        normalize_column_name(column)
        for column in output.columns
    ]
    return output


def resolve_file(
    candidates: Iterable[Path],
    dataset_name: str,
) -> Path:
    for path in candidates:
        if path.exists():
            LOGGER.info(
                "Resolved %s input: %s",
                dataset_name,
                path,
            )
            return path

    candidate_text = "\n".join(
        f"  - {path}"
        for path in candidates
    )

    raise FileNotFoundError(
        f"Could not find preprocessed {dataset_name} data. "
        f"Checked:\n{candidate_text}"
    )


def find_first_column(
    columns: Iterable[str],
    candidates: Iterable[str],
) -> str | None:
    available = set(columns)

    for candidate in candidates:
        normalized = normalize_column_name(candidate)

        if normalized in available:
            return normalized

    return None


def require_columns(
    frame: pd.DataFrame,
    required: Iterable[str],
    dataset_name: str,
) -> None:
    missing = sorted(
        set(required)
        - set(frame.columns)
    )

    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing}"
        )


def load_parquet_table(
    path: Path,
    dataset_name: str,
) -> pd.DataFrame:
    LOGGER.info(
        "Loading %s data from %s.",
        dataset_name,
        path,
    )

    frame = normalize_columns(
        pd.read_parquet(path)
    )

    timestamp_column = find_first_column(
        frame.columns,
        [
            "timestamp_utc",
            "timestamp",
            "date_begin_gmt",
            "datetime_utc",
        ],
    )

    if timestamp_column is None:
        raise ValueError(
            f"{dataset_name} does not contain a recognizable UTC timestamp."
        )

    if timestamp_column != "timestamp_utc":
        frame = frame.rename(
            columns={
                timestamp_column: "timestamp_utc",
            }
        )

    frame["timestamp_utc"] = pd.to_datetime(
        frame["timestamp_utc"],
        utc=True,
        errors="coerce",
    )

    invalid_timestamps = int(
        frame["timestamp_utc"]
        .isna()
        .sum()
    )

    if invalid_timestamps:
        raise ValueError(
            f"{dataset_name} contains {invalid_timestamps} invalid timestamps."
        )

    frame = (
        frame
        .drop_duplicates(
            subset=["timestamp_utc"],
            keep="last",
        )
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )

    return frame


def numericize_except_timestamp(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    output = frame.copy()

    for column in output.columns:
        if column == "timestamp_utc":
            continue

        output[column] = pd.to_numeric(
            output[column],
            errors="coerce",
        )

    return output


def add_lags(
    frame: pd.DataFrame,
    column: str,
    lags: Iterable[int],
    prefix: str | None = None,
) -> pd.DataFrame:
    if column not in frame.columns:
        return frame

    output = frame.copy()
    name = prefix or column

    for lag in lags:
        output[
            f"{name}_lag_{lag}h"
        ] = output[column].shift(lag)

    return output


def add_changes(
    frame: pd.DataFrame,
    column: str,
    lags: Iterable[int],
    prefix: str | None = None,
) -> pd.DataFrame:
    if column not in frame.columns:
        return frame

    output = frame.copy()
    name = prefix or column

    for lag in lags:
        output[
            f"{name}_change_{lag}h"
        ] = output[column].diff(lag)

    return output


def add_rolling_statistics(
    frame: pd.DataFrame,
    column: str,
    windows: Iterable[int],
    prefix: str | None = None,
) -> pd.DataFrame:
    if column not in frame.columns:
        return frame

    output = frame.copy()
    name = prefix or column

    # Shift one hour before rolling so every statistic uses only prior data.
    historical = output[column].shift(1)

    for window in windows:
        rolling = historical.rolling(
            window,
            min_periods=max(1, min(window, 3)),
        )

        output[
            f"{name}_mean_prior_{window}h"
        ] = rolling.mean()

        output[
            f"{name}_std_prior_{window}h"
        ] = rolling.std()

        output[
            f"{name}_min_prior_{window}h"
        ] = rolling.min()

        output[
            f"{name}_max_prior_{window}h"
        ] = rolling.max()

    return output


# ============================================================================
# Source loading and standardization
# ============================================================================

def load_pa_data(
    path: Path,
) -> pd.DataFrame:
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

def load_calendar_data(
    path: Path,
) -> pd.DataFrame:
    """
    Load the canonical hourly calendar-feature dataset.

    Calendar features must be built first by calendar_features.py.
    """

    if not path.exists():
        raise FileNotFoundError(
            "Canonical calendar features were not found at:\n"
            f"  {path}\n"
            "Run calendar_features.py first."
        )

    LOGGER.info(
        "Loading canonical calendar features from %s.",
        path,
    )

    calendar = pd.read_parquet(
        path
    )

    calendar = normalize_columns(
        calendar
    )

    require_columns(
        calendar,
        [
            "timestamp_utc",
            "hour_alberta",
            "day_of_week_alberta",
            "month_alberta",
            "year_alberta",
            "is_weekend",
            "is_business_hour",
            "is_morning_ramp",
            "is_evening_peak",
        ],
        "calendar features",
    )

    calendar["timestamp_utc"] = pd.to_datetime(
        calendar["timestamp_utc"],
        utc=True,
        errors="coerce",
    )

    invalid_timestamps = int(
        calendar["timestamp_utc"]
        .isna()
        .sum()
    )

    if invalid_timestamps:
        raise ValueError(
            "Calendar features contain "
            f"{invalid_timestamps} invalid timestamps."
        )

    calendar = (
        calendar
        .drop_duplicates(
            subset=["timestamp_utc"],
            keep="last",
        )
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )

    return calendar

# ============================================================================
# Intertie features
# ============================================================================

def identify_directional_columns(
    columns: Iterable[str],
    direction: str,
) -> list[str]:
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
    output = frame.copy()

    import_columns = identify_directional_columns(
        output.columns,
        "import",
    )

    export_columns = identify_directional_columns(
        output.columns,
        "export",
    )

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
    output = frame.copy()

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
    output = frame.copy()

    if "pool_price" not in output.columns:
        return output

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
    output = frame.copy()

    if "ail_mw" not in output.columns:
        return output

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
    output = frame.copy()

    if {
        "ail_mw",
        "net_imports_mw",
    }.issubset(output.columns):
        output["import_adjusted_load_mw"] = (
            output["ail_mw"]
            - output["net_imports_mw"]
        )

    if {
        "ail_mw",
        "total_outage_mw",
    }.issubset(output.columns):
        output["outage_adjusted_load_mw"] = (
            output["ail_mw"]
            + output["total_outage_mw"]
        )

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


def apply_feature_builders(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    output = frame.copy()

    for builder in FEATURE_BUILDERS:
        output = builder(
            output
        )

    return output


# ============================================================================
# Merge and audit
# ============================================================================

def merge_sources(
    pa: pd.DataFrame,
    intertie: pd.DataFrame,
    outages: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    output = pa.copy()

    output = output.merge(
        intertie,
        on="timestamp_utc",
        how="left",
        validate="one_to_one",
    )

    output = output.merge(
        outages,
        on="timestamp_utc",
        how="left",
        validate="one_to_one",
    )

    output = output.merge(
        calendar,
        on="timestamp_utc",
        how="left",
        validate="one_to_one",
    )

    for flag in [
        "pa_available",
        "intertie_available",
        "outage_available",
    ]:
        if flag not in output.columns:
            output[flag] = 0

        output[flag] = (
            output[flag]
            .fillna(0)
            .astype("int8")
        )

    output = (
        output
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )

    return output


def source_summary(
    name: str,
    frame: pd.DataFrame,
    path: Path,
) -> dict:
    timestamps = pd.DatetimeIndex(
        frame["timestamp_utc"]
    )

    expected = pd.date_range(
        timestamps.min(),
        timestamps.max(),
        freq="h",
        tz="UTC",
    )

    return {
        "source": name,
        "path": str(path),
        "rows": len(frame),
        "columns": len(frame.columns),
        "start_utc": str(timestamps.min()),
        "end_utc": str(timestamps.max()),
        "duplicate_timestamps": int(
            timestamps.duplicated().sum()
        ),
        "missing_hours_within_source": len(
            expected.difference(timestamps)
        ),
    }


def audit_market_features(
    frame: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    bool,
]:
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

    numeric_columns = [
        column
        for column in frame.columns
        if column != "timestamp_utc"
        and pd.api.types.is_numeric_dtype(
            frame[column]
        )
    ]

    summary = (
        frame[numeric_columns]
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
        .rename(
            columns={
                "index": "feature",
                "1%": "p01",
                "50%": "median",
                "99%": "p99",
            }
        )
    )

    summary["missing_count"] = (
        frame[numeric_columns]
        .isna()
        .sum()
        .values
    )

    summary["missing_pct"] = (
        summary["missing_count"]
        / len(frame)
        * 100
    )

    summary["dtype"] = [
        str(frame[column].dtype)
        for column in numeric_columns
    ]

    audit = pd.DataFrame(rows)

    error_checks = audit.loc[
        audit["severity"].eq("error"),
        "pass",
    ]

    passed = (
        bool(error_checks.all())
        if not error_checks.empty
        else True
    )

    return audit, summary, passed


def print_report(
    frame: pd.DataFrame,
    audit: pd.DataFrame,
    passed: bool,
) -> None:
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
# Output helpers
# ============================================================================

def save_audit_outputs(
    audit: pd.DataFrame,
    feature_summary: pd.DataFrame,
    source_summary_frame: pd.DataFrame,
) -> None:
    """Write all market-feature audit and source-summary outputs."""

    ensure_output_directories()

    LOGGER.info(
        "Writing market-feature audit checks to %s.",
        AUDIT_FILE,
    )

    audit.to_csv(
        AUDIT_FILE,
        index=False,
    )

    LOGGER.info(
        "Writing market-feature numeric summary to %s.",
        FEATURE_SUMMARY_FILE,
    )

    feature_summary.to_csv(
        FEATURE_SUMMARY_FILE,
        index=False,
    )

    LOGGER.info(
        "Writing market-feature source summary to %s.",
        SOURCE_SUMMARY_FILE,
    )

    source_summary_frame.to_csv(
        SOURCE_SUMMARY_FILE,
        index=False,
    )


def save_feature_outputs(
    frame: pd.DataFrame,
    write_csv: bool,
) -> None:
    """Write canonical Parquet and optional CSV feature outputs."""

    ensure_output_directories()

    LOGGER.info(
        "Writing canonical market-feature Parquet to %s.",
        OUTPUT_PARQUET,
    )

    frame.to_parquet(
        OUTPUT_PARQUET,
        index=False,
    )

    if write_csv:
        LOGGER.info(
            "Writing optional market-feature CSV to %s.",
            OUTPUT_CSV,
        )

        frame.to_csv(
            OUTPUT_CSV,
            index=False,
        )


def existing_outputs_satisfy_request(
    write_csv: bool,
) -> bool:
    """
    Return True when every requested feature output already exists.

    Parquet is always required. CSV is required only when write_csv=True.
    """

    parquet_exists = OUTPUT_PARQUET.exists()

    csv_requirement_satisfied = (
        OUTPUT_CSV.exists()
        if write_csv
        else True
    )

    return (
        parquet_exists
        and csv_requirement_satisfied
    )


def read_existing_parquet_for_csv() -> pd.DataFrame:
    """
    Load the canonical Parquet when only a missing CSV output is requested.
    """

    LOGGER.info(
        "Loading existing market-feature Parquet from %s.",
        OUTPUT_PARQUET,
    )

    frame = pd.read_parquet(
        OUTPUT_PARQUET
    )

    if "timestamp_utc" in frame.columns:
        frame["timestamp_utc"] = pd.to_datetime(
            frame["timestamp_utc"],
            utc=True,
        )

    return frame


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

    ensure_output_directories()

    # Skip only when all explicitly requested feature outputs already exist.
    if (
        not overwrite
        and existing_outputs_satisfy_request(
            write_csv=write_csv
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
        and OUTPUT_PARQUET.exists()
        and write_csv
        and not OUTPUT_CSV.exists()
    ):
        frame = read_existing_parquet_for_csv()

        LOGGER.info(
            "Creating missing CSV from existing canonical Parquet."
        )

        frame.to_csv(
            OUTPUT_CSV,
            index=False,
        )

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

    pa_path = resolve_file(
        PA_FILE_CANDIDATES,
        "P&A",
    )

    intertie_path = resolve_file(
        INTERTIE_FILE_CANDIDATES,
        "intertie",
    )

    outage_path = resolve_file(
        OUTAGE_FILE_CANDIDATES,
        "outage",
    )

    pa = load_pa_data(
        pa_path
    )

    intertie = load_intertie_data(
        intertie_path
    )

    outages = load_outage_data(
        outage_path
    )

    calendar = load_calendar_data(
        CALENDAR_FILE
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
        calendar,
    )

    LOGGER.info(
        "Applying %s market feature builders.",
        len(FEATURE_BUILDERS),
    )

    master = apply_feature_builders(
        master
    )

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
    save_audit_outputs(
        audit,
        feature_summary,
        source_summary_frame,
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

    save_feature_outputs(
        master,
        write_csv=write_csv,
    )

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
