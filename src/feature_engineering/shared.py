"""
Shared contracts and utilities for hourly feature-engineering pipelines.

Folder-wide style convention
----------------------------
Feature modules use framed top-level section headers, concise function
docstrings, and inline comments for domain rationale or non-obvious timing and
unit semantics. Comments should explain why a transformation exists rather
than narrating ordinary pandas or NumPy operations.
"""

# ============================================================================
# Imports
# ============================================================================

from __future__ import annotations

import logging
import sys
import calendar
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from pipeline_shared import (
    MANIFEST_VERSION,
    build_manifest,
    manifest_path,
    output_is_current,
    outputs_are_current,
    write_manifest,
    write_manifests,
)

# ============================================================================
# Configuration
# ============================================================================

TIMESTAMP_CANDIDATES = (
    "timestamp_utc",
    "timestamp",
    "date_begin_gmt",
    "datetime_utc",
)


# ============================================================================
# Import and hourly-index helpers
# ============================================================================

def ensure_src_on_path(module_file: str | Path) -> None:
    """Make ``src`` importable when a feature module is run as a script."""

    src_dir = Path(module_file).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def configure_logging(verbose: bool = False) -> None:
    """Configure the standard feature-pipeline console logger."""

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def ensure_directories(*paths: Path) -> None:
    """Create each supplied output directory if it does not exist."""

    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def feature_code_paths(module_path: Path) -> list[Path]:
    """Return all code and configuration files governing a feature stage."""

    module_path = module_path.resolve()
    return [
        module_path,
        Path(__file__).resolve(),
        Path(__file__).resolve().parents[1] / "pipeline_shared.py",
        module_path.parents[1] / "config.py",
    ]


def add_check(
    rows: list[dict[str, Any]],
    check: str,
    passed: bool,
    observed: Any = None,
    expected: Any = None,
    severity: str = "error",
    notes: str = "",
) -> None:
    """Append one standardized audit result."""

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


def add_period_check(
    rows: list[dict[str, Any]],
    period: str,
    check: str,
    passed: bool,
    observed: Any = None,
    expected: Any = None,
    severity: str = "error",
    notes: str = "",
) -> None:
    """Append one standardized audit result associated with a period."""

    rows.append(
        {
            "period": period,
            "check": check,
            "pass": bool(passed),
            "severity": severity,
            "observed": observed,
            "expected": expected,
            "notes": notes,
        }
    )


def audit_passed(audit: pd.DataFrame) -> bool:
    """Return whether every error-level audit check passed."""

    if audit.empty or not {"severity", "pass"}.issubset(audit.columns):
        return True
    error_checks = audit.loc[audit["severity"].eq("error"), "pass"]
    return bool(error_checks.all()) if not error_checks.empty else True


def apply_feature_builders(
    frame: pd.DataFrame,
    builders: Iterable[Callable[[pd.DataFrame], pd.DataFrame]],
) -> pd.DataFrame:
    """Apply registered feature builders in dependency order."""

    output = frame.copy()
    for builder in builders:
        output = builder(output)
    return output


def normalize_column_name(column: str) -> str:
    """Normalize one source heading to snake case."""

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
    """Return a copy with normalized column names."""

    output = frame.copy()
    output.columns = [normalize_column_name(column) for column in output.columns]
    return output


def find_first_column(
    columns: Iterable[str],
    candidates: Iterable[str],
) -> str | None:
    """Return the first normalized candidate heading that is present."""

    available = set(columns)
    for candidate in candidates:
        normalized = normalize_column_name(candidate)
        if normalized in available:
            return normalized
    return None


def resolve_file(
    candidates: Iterable[Path],
    dataset_name: str,
) -> Path:
    """Return the first available candidate input file."""

    candidate_paths = list(candidates)
    for path in candidate_paths:
        if path.exists():
            logging.getLogger(__name__).info(
                "Resolved %s input: %s",
                dataset_name,
                path,
            )
            return path
    candidate_text = "\n".join(f"  - {path}" for path in candidate_paths)
    raise FileNotFoundError(
        f"Could not find preprocessed {dataset_name} data. "
        f"Checked:\n{candidate_text}"
    )


def load_parquet_table(path: Path, dataset_name: str) -> pd.DataFrame:
    """Load and standardize a timestamp-keyed Parquet source."""

    logging.getLogger(__name__).info("Loading %s data from %s.", dataset_name, path)
    frame = normalize_columns(pd.read_parquet(path))
    timestamp_column = find_first_column(frame.columns, TIMESTAMP_CANDIDATES)
    if timestamp_column is None:
        raise ValueError(
            f"{dataset_name} does not contain a recognizable UTC timestamp."
        )
    if timestamp_column != "timestamp_utc":
        frame = frame.rename(columns={timestamp_column: "timestamp_utc"})
    frame["timestamp_utc"] = pd.to_datetime(
        frame["timestamp_utc"],
        utc=True,
        errors="coerce",
    )
    invalid_timestamps = int(frame["timestamp_utc"].isna().sum())
    if invalid_timestamps:
        raise ValueError(
            f"{dataset_name} contains {invalid_timestamps} invalid timestamps."
        )
    return (
        frame.drop_duplicates(subset=["timestamp_utc"], keep="last")
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )


def numericize_except_timestamp(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce non-key columns to numeric values."""

    output = frame.copy()
    for column in output.columns:
        if column != "timestamp_utc":
            output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def merge_hourly_sources(
    backbone: pd.DataFrame,
    sources: Iterable[pd.DataFrame],
    availability_flags: Iterable[str],
) -> pd.DataFrame:
    """Left-join hourly sources and standardize their availability flags."""

    output = backbone.copy()
    for source in sources:
        output = output.merge(
            source,
            on="timestamp_utc",
            how="left",
            validate="one_to_one",
        )
    for flag in availability_flags:
        if flag not in output.columns:
            output[flag] = 0
        output[flag] = output[flag].fillna(0).astype("int8")
    return output.sort_values("timestamp_utc").reset_index(drop=True)


def source_summary(name: str, frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    """Summarize source coverage and hourly key integrity."""

    timestamps = pd.DatetimeIndex(frame["timestamp_utc"])
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
        "duplicate_timestamps": int(timestamps.duplicated().sum()),
        "missing_hours_within_source": len(expected.difference(timestamps)),
    }


def numeric_feature_summary(
    frame: pd.DataFrame,
    timing_classifier: Callable[[str], str] | None = None,
) -> pd.DataFrame:
    """Build the standard descriptive catalog for numeric feature columns."""

    numeric_columns = [
        column
        for column in frame.columns
        if column != "timestamp_utc"
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    summary = (
        frame[numeric_columns]
        .describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99])
        .T.reset_index()
        .rename(
            columns={
                "index": "feature",
                "1%": "p01",
                "50%": "median",
                "99%": "p99",
            }
        )
    )
    summary["missing_count"] = frame[numeric_columns].isna().sum().values
    summary["missing_pct"] = summary["missing_count"] / len(frame) * 100
    summary["dtype"] = [str(frame[column].dtype) for column in numeric_columns]
    if timing_classifier is not None:
        summary["feature_timing"] = [
            timing_classifier(column) for column in numeric_columns
        ]
    return summary


def save_tables(
    tables: Mapping[Path, pd.DataFrame],
    descriptions: Mapping[Path, str] | None = None,
    manifest: Mapping[str, Any] | None = None,
) -> None:
    """Write CSV evidence tables and optionally bind their provenance."""

    logger = logging.getLogger(__name__)
    for path, frame in tables.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        description = descriptions.get(path, path.name) if descriptions else path.name
        logger.info("Writing %s to %s.", description, path)
        frame.to_csv(path, index=False)
    if manifest is not None:
        write_manifests(list(tables), manifest)


def save_feature_outputs(
    frame: pd.DataFrame,
    parquet_path: Path,
    csv_path: Path,
    write_csv: bool,
    dataset_label: str,
    manifest: Mapping[str, Any] | None = None,
) -> None:
    """Write a canonical Parquet and its optional CSV representation."""

    logger = logging.getLogger(__name__)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing canonical %s Parquet to %s.", dataset_label, parquet_path)
    frame.to_parquet(parquet_path, index=False)
    if write_csv:
        logger.info("Writing optional %s CSV to %s.", dataset_label, csv_path)
        frame.to_csv(csv_path, index=False)
    if manifest is not None:
        paths = [parquet_path, *([csv_path] if write_csv else [])]
        write_manifests(paths, manifest)


def existing_outputs_satisfy_request(
    parquet_path: Path,
    csv_path: Path,
    write_csv: bool,
    expected_manifest: Mapping[str, Any],
    required_artifacts: Sequence[Path] = (),
) -> bool:
    """Return whether all requested data and evidence artifacts are current."""

    requested = [
        parquet_path,
        *required_artifacts,
        *([csv_path] if write_csv else []),
    ]
    return outputs_are_current(requested, expected_manifest)


def read_existing_parquet(path: Path) -> pd.DataFrame:
    """Load a canonical feature Parquet and restore its UTC timestamp type."""

    logging.getLogger(__name__).info("Loading existing feature Parquet from %s.", path)
    frame = pd.read_parquet(path)
    if "timestamp_utc" in frame.columns:
        frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    return frame


def monthly_file_period(path: Path) -> str:
    """Extract and validate YYYY-MM from a standardized monthly filename."""

    parts = path.stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Cannot identify year and month from {path.name}")
    period = f"{parts[-2]}-{parts[-1]}"
    try:
        pd.Period(period, freq="M")
    except ValueError as exc:
        raise ValueError(f"Cannot identify year and month from {path.name}") from exc
    return period


def expected_month_hours(period: str) -> int:
    """Return the calendar-hour count for a YYYY-MM period."""

    year, month = map(int, period.split("-"))
    return calendar.monthrange(year, month)[1] * 24


def get_monthly_files(monthly_dir: Path) -> list[Path]:
    """Return standardized ERA5 monthly files in chronological order."""

    files = sorted(monthly_dir.glob("era5_alberta_standardized_*.nc"))
    if not files:
        raise FileNotFoundError(
            f"No standardized monthly ERA5 files found in {monthly_dir}"
        )
    return files


def load_grid(reference_file: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load latitude and longitude coordinates from one ERA5 file."""

    import xarray as xr

    with xr.open_dataset(reference_file) as dataset:
        missing = {"timestamp", "lat", "lon"} - set(dataset.coords) - set(
            dataset.dims
        )
        if missing:
            raise ValueError(
                f"{reference_file.name} is missing coordinates: {sorted(missing)}"
            )
        latitudes = np.asarray(dataset["lat"].values, dtype=float)
        longitudes = np.asarray(dataset["lon"].values, dtype=float)
    return latitudes, longitudes


def weather_array(site_dataset: Any, variable: str) -> np.ndarray | None:
    """Return one weather field as a timestamp-by-site array."""

    if variable not in site_dataset.data_vars:
        return None
    array = site_dataset[variable].transpose("timestamp", "weather_site_id").values
    return np.asarray(array, dtype=float)


def validate_continuous_hourly_frame(
    frame: pd.DataFrame,
    timestamp_column: str = "timestamp_utc",
) -> None:
    """Require a sorted, unique, uninterrupted hourly UTC timeline."""

    if timestamp_column not in frame.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_column}")

    timestamps = pd.DatetimeIndex(
        pd.to_datetime(frame[timestamp_column], utc=True, errors="raise")
    )
    if timestamps.has_duplicates:
        raise ValueError("Hourly feature input contains duplicate timestamps.")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("Hourly feature input must be sorted by timestamp.")
    if len(timestamps) > 1:
        spacing = pd.Series(timestamps).diff().dropna()
        if not spacing.eq(pd.Timedelta(hours=1)).all():
            raise ValueError(
                "Row-based hourly features require an uninterrupted hourly timeline."
            )


def add_lags(
    frame: pd.DataFrame,
    column: str,
    lags: Iterable[int],
    prefix: str | None = None,
) -> pd.DataFrame:
    """Add prior-hour values; one row is guaranteed to equal one hour."""

    if column not in frame.columns:
        return frame
    validate_continuous_hourly_frame(frame)
    output = frame.copy()
    name = prefix or column
    for lag in lags:
        if lag <= 0:
            raise ValueError("Lag lengths must be positive integers.")
        output[f"{name}_lag_{lag}h"] = output[column].shift(lag)
    return output


def add_changes_through_current_hour(
    frame: pd.DataFrame,
    column: str,
    lags: Iterable[int],
    prefix: str | None = None,
) -> pd.DataFrame:
    """Add current-minus-prior changes, explicitly classified as ex-post."""

    if column not in frame.columns:
        return frame
    validate_continuous_hourly_frame(frame)
    output = frame.copy()
    name = prefix or column
    for lag in lags:
        if lag <= 0:
            raise ValueError("Change horizons must be positive integers.")
        output[f"{name}_change_{lag}h"] = output[column].diff(lag)
    return output


def add_prior_rolling_statistics(
    frame: pd.DataFrame,
    column: str,
    windows: Iterable[int],
    prefix: str | None = None,
    minimum_observations: int = 3,
) -> pd.DataFrame:
    """Add rolling statistics ending one hour before the current row."""

    if column not in frame.columns:
        return frame
    validate_continuous_hourly_frame(frame)
    output = frame.copy()
    name = prefix or column
    historical = output[column].shift(1)
    for window in windows:
        if window <= 0:
            raise ValueError("Rolling windows must be positive integers.")
        rolling = historical.rolling(
            window,
            min_periods=min(window, minimum_observations),
        )
        output[f"{name}_mean_prior_{window}h"] = rolling.mean()
        output[f"{name}_std_prior_{window}h"] = rolling.std()
        output[f"{name}_min_prior_{window}h"] = rolling.min()
        output[f"{name}_max_prior_{window}h"] = rolling.max()
    return output


# ============================================================================
# Numeric and spatial helpers
# ============================================================================

def safe_divide(
    numerator: np.ndarray | pd.Series,
    denominator: np.ndarray | pd.Series,
) -> np.ndarray:
    """Divide finite values elementwise and return NaN for zero denominators."""

    numerator_array = np.asarray(numerator, dtype=float)
    denominator_array = np.asarray(denominator, dtype=float)
    if numerator_array.shape != denominator_array.shape:
        raise ValueError("Numerator and denominator must have the same shape.")
    result = np.full(numerator_array.shape, np.nan, dtype=float)
    valid = (
        np.isfinite(numerator_array)
        & np.isfinite(denominator_array)
        & (denominator_array != 0)
    )
    np.divide(
        numerator_array,
        denominator_array,
        out=result,
        where=valid,
    )
    return result


def nearest_coordinate(
    value: float,
    coordinates: np.ndarray,
) -> tuple[float, int]:
    """Return the nearest coordinate value and its positional index."""

    position = int(np.abs(np.asarray(coordinates, dtype=float) - value).argmin())
    return float(coordinates[position]), position


def haversine_distance_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Return great-circle distance in kilometres using the haversine formula."""

    radius_km = 6371.0088
    latitude_1 = np.radians(lat1)
    latitude_2 = np.radians(lat2)
    delta_latitude = latitude_2 - latitude_1
    delta_longitude = np.radians(lon2 - lon1)
    haversine = (
        np.sin(delta_latitude / 2.0) ** 2
        + np.cos(latitude_1)
        * np.cos(latitude_2)
        * np.sin(delta_longitude / 2.0) ** 2
    )
    central_angle = 2.0 * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0)))
    return float(radius_km * central_angle)


def weighted_average(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return row-wise weighted averages while excluding missing values."""

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.shape != weights.shape:
        raise ValueError("Values and weights must have the same shape.")
    valid = np.isfinite(values) & np.isfinite(weights)
    valid_weights = np.where(valid, weights, 0.0)
    numerator = np.nansum(values * valid_weights, axis=1)
    denominator = valid_weights.sum(axis=1)
    result = np.full(values.shape[0], np.nan, dtype=float)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def weighted_standard_deviation(
    values: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Return row-wise population standard deviations using supplied weights."""

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mean = weighted_average(values, weights)
    valid = np.isfinite(values) & np.isfinite(weights)
    valid_weights = np.where(valid, weights, 0.0)
    numerator = np.nansum((values - mean[:, None]) ** 2 * valid_weights, axis=1)
    denominator = valid_weights.sum(axis=1)
    variance = np.full(values.shape[0], np.nan, dtype=float)
    np.divide(numerator, denominator, out=variance, where=denominator > 0)
    return np.sqrt(variance)


# ============================================================================
# Schema and feature-metadata helpers
# ============================================================================

def require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    dataset_name: str,
) -> None:
    """Raise a clear error when a required output schema is incomplete."""

    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {missing}")


def classify_feature_timing(
    feature: str,
    *,
    target_columns: Iterable[str] = (),
    target_derived_prefixes: Iterable[str] = (),
    known_ahead_columns: Iterable[str] = (),
) -> str:
    """Classify when a feature is available relative to its timestamp."""

    if feature in set(target_columns):
        return "target"
    if "_lag_" in feature or "_prior_" in feature:
        return "historical"
    if any(feature.startswith(prefix) for prefix in target_derived_prefixes):
        return "ex_post_target_derived"
    if feature in set(known_ahead_columns):
        return "known_ahead"
    return "contemporaneous"
