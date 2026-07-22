# src/feature_engineering/calendar_features.py

"""
Build canonical hourly calendar features for the Alberta power-market project.

Purpose
-------
Calendar features describe when each market hour occurs. They are objective
time metadata and should be created before market exploration so every EDA
and modeling notebook uses the same definitions.

The output contains:

1. UTC and Alberta-local time references.
2. Hour, weekday, month, quarter, week, and day-of-year fields.
3. Weekend, weekday, business-day, holiday, and adjacent-holiday flags.
4. Broad operating-period flags such as overnight, morning ramp, business
   hours, afternoon, evening peak, and late evening.
5. Meteorological season and heating/cooling-season flags.
6. Cyclical sine/cosine encodings suitable for regression and machine learning.

Important timezone convention
-----------------------------
The canonical key remains timestamp_utc.

Alberta-local calendar attributes are calculated using America/Edmonton,
which correctly handles Mountain Standard Time and daylight-saving time.

By default, the output spans the common weather-feature horizon:

    2015-01-01 00:00 UTC
    through
    2026-06-30 23:00 UTC

The range can be changed from the command line.

Holiday support
---------------
The script uses the `holidays` Python package to generate Canadian statutory
holidays for Alberta.

Install it if needed:

    pip install holidays

Outputs
-------
Canonical feature output:

    data/features/calendar/calendar_features_hourly.parquet

Optional full CSV output:

    data/features/calendar/calendar_features_hourly.csv

Audit outputs:

    data/audits/calendar_features_audit_checks.csv
    data/audits/calendar_features_summary.csv
    data/audits/calendar_holiday_dates.csv

Run
---
Standard run:

    python src/feature_engineering/calendar_features.py

Overwrite existing outputs:

    python src/feature_engineering/calendar_features.py \
        --overwrite

Custom range:

    python src/feature_engineering/calendar_features.py \
        --start "2015-01-01 00:00:00+00:00" \
        --end "2026-06-30 23:00:00+00:00" \
        --overwrite

Write CSV as well:

    python src/feature_engineering/calendar_features.py \
        --overwrite \
        --write-csv

Verbose logging:

    python src/feature_engineering/calendar_features.py \
        --verbose
"""

# ============================================================================
# Imports
# ============================================================================

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import holidays
except ImportError as exc:
    raise ImportError(
        "The calendar feature pipeline requires the `holidays` package. "
        "Install it with: pip install holidays"
    ) from exc


# ============================================================================
# Logging
# ============================================================================

LOGGER = logging.getLogger(__name__)


def configure_logging(verbose: bool = False) -> None:
    """
    Configure console logging for the pipeline.

    INFO is used by default.

    DEBUG can be enabled with the command-line --verbose flag.
    """

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


# ============================================================================
# Project paths
# ============================================================================

# This file is expected to live at:
#
#     PROJECT_ROOT/src/feature_engineering/calendar_features.py
#
# parents[0] -> feature_engineering
# parents[1] -> src
# parents[2] -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES_DIR = PROJECT_ROOT / "data" / "features"
OUTPUT_DIR = FEATURES_DIR / "calendar"

OUTPUT_PARQUET = OUTPUT_DIR / "calendar_features_hourly.parquet"
OUTPUT_CSV = OUTPUT_DIR / "calendar_features_hourly.csv"

AUDIT_DIR = PROJECT_ROOT / "data" / "audits"

AUDIT_FILE = AUDIT_DIR / "calendar_features_audit_checks.csv"
SUMMARY_FILE = AUDIT_DIR / "calendar_features_summary.csv"
HOLIDAY_DATES_FILE = AUDIT_DIR / "calendar_holiday_dates.csv"


# ============================================================================
# Configuration
# ============================================================================

TIMEZONE = "America/Edmonton"

DEFAULT_START_UTC = "2015-01-01 00:00:00+00:00"
DEFAULT_END_UTC = "2026-06-30 23:00:00+00:00"

DATASET_NAME = "calendar_features"

VALID_SEASONS = {
    "winter",
    "spring",
    "summer",
    "fall",
}

VALID_PERIODS_OF_DAY = {
    "overnight",
    "morning_ramp",
    "daytime",
    "evening_peak",
    "late_evening",
}

PERIOD_FLAG_COLUMNS = [
    "is_overnight",
    "is_morning_ramp",
    "is_business_hour",
    "is_afternoon",
    "is_evening_peak",
    "is_late_evening",
]

BINARY_COLUMNS = [
    "is_weekday",
    "is_weekend",
    "is_holiday",
    "is_business_day",
    "is_day_before_holiday",
    "is_day_after_holiday",
    "is_month_start",
    "is_month_end",
    "is_quarter_start",
    "is_quarter_end",
    "is_year_start",
    "is_year_end",
    "is_heating_season",
    "is_cooling_season",
    "is_daylight_saving_time",
    *PERIOD_FLAG_COLUMNS,
]

CYCLICAL_COLUMNS = [
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]

REQUIRED_COLUMNS = [
    "timestamp_utc",
    "timestamp_alberta",
    "local_date",
    "year_alberta",
    "quarter_alberta",
    "month_alberta",
    "month_name_alberta",
    "week_of_year_alberta",
    "day_of_year_alberta",
    "day_of_month_alberta",
    "day_of_week_alberta",
    "day_name_alberta",
    "hour_alberta",
    "utc_offset_hours",
    "is_daylight_saving_time",
    "is_weekday",
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "is_quarter_start",
    "is_quarter_end",
    "is_year_start",
    "is_year_end",
    "holiday_name",
    "is_holiday",
    "is_day_before_holiday",
    "is_day_after_holiday",
    "is_business_day",
    "season",
    "is_heating_season",
    "is_cooling_season",
    "period_of_day",
    *PERIOD_FLAG_COLUMNS,
    *CYCLICAL_COLUMNS,
]


# ============================================================================
# General helpers
# ============================================================================

def ensure_output_directories() -> None:
    """Create feature and audit output directories if they do not exist."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def add_check(
    rows: list[dict[str, Any]],
    check: str,
    passed: bool,
    observed: Any = None,
    expected: Any = None,
    severity: str = "error",
    notes: str = "",
) -> None:
    """Append one audit check."""

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


def parse_utc_timestamp(
    value: str | pd.Timestamp,
    field_name: str,
) -> pd.Timestamp:
    """Parse a timezone-aware timestamp and convert it to UTC."""

    # Convert the input value to a pandas Timestamp object.
    timestamp = pd.Timestamp(value)

    # Verify that the timestamp includes timezone information.
    # Raise an error if the timestamp is timezone-naive.
    if timestamp.tzinfo is None:
        raise ValueError(
            f"{field_name} must include timezone information. "
            "Example: 2015-01-01 00:00:00+00:00"
        )

    # Convert the timestamp to UTC and return it.
    return timestamp.tz_convert("UTC")


def validate_range(
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
) -> None:
    """Validate the requested calendar coverage."""

    # Verify that the timestamps have a start and end date in the correct order.
    if end_utc < start_utc:
        raise ValueError(
            "End timestamp must be on or after start timestamp. "
            f"Observed start={start_utc}, end={end_utc}"
        )

    # Verify that the start timestamp is aligned to the hour.
    if (
        start_utc.minute != 0
        or start_utc.second != 0
        or start_utc.microsecond != 0
        or start_utc.nanosecond != 0
    ):
        raise ValueError(
            f"Start timestamp must be aligned to the hour: {start_utc}"
        )

    # Verify that the end timestamp is aligned to the hour.
    if (
        end_utc.minute != 0
        or end_utc.second != 0
        or end_utc.microsecond != 0
        or end_utc.nanosecond != 0
    ):
        raise ValueError(
            f"End timestamp must be aligned to the hour: {end_utc}"
        )


def expected_hour_count(
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
) -> int:
    """Return the expected number of inclusive hourly timestamps."""

    return len(
        pd.date_range(
            start=start_utc,
            end=end_utc,
            freq="h",
            tz="UTC",
        )
    )


def meteorological_season(
    month: pd.Series,
) -> pd.Series:
    """Map month numbers to meteorological seasons."""

    # Create a list of boolean masks, one for each meteorological season.
    # Each mask identifies which rows belong to that season.
    conditions = [
        month.isin([12, 1, 2]),
        month.isin([3, 4, 5]),
        month.isin([6, 7, 8]),
        month.isin([9, 10, 11]),
    ]

    # Define the season assigned when the corresponding condition is True.
    choices = [
        "winter",
        "spring",
        "summer",
        "fall",
    ]

    # Create a Series containing the season for each month.
    # If a month does not match any condition, assign "unknown".
    return pd.Series(
        # np.select() essentially performs:
        # if condition 1 -> winter
        # if condition 2 -> spring
        # and so on.
        np.select(
            conditions,
            choices,
            default="unknown",
        ),
        index=month.index,
        dtype="string",
    )


def period_of_day(
    hour: pd.Series,
) -> pd.Series:
    """Create one mutually exclusive period-of-day category."""

    # Create a list of boolean masks, one for each period of the day.
    # Each mask identifies which rows belong to that period.
    conditions = [
        hour.between(0, 5),
        hour.between(6, 9),
        hour.between(10, 15),
        hour.between(16, 19),
        hour.between(20, 23),
    ]

    # Define the period assigned when each condition is True.
    choices = [
        "overnight",
        "morning_ramp",
        "daytime",
        "evening_peak",
        "late_evening",
    ]

    # np.select assigns each condition to its corresponding choice.
    # The default value is set to "unknown".
    return pd.Series(
        np.select(
            conditions,
            choices,
            default="unknown",
        ),
        index=hour.index,
        dtype="string",
    )


# ============================================================================
# Holiday construction
# ============================================================================

def build_alberta_holiday_table(
    local_dates: pd.Series,
) -> pd.DataFrame:
    """
    Build one row per Alberta holiday date required by the dataset.

    The `holidays` package includes observed holiday dates where applicable.
    """

    if local_dates.empty:
        return pd.DataFrame(
            columns=[
                "local_date",
                "holiday_name",
            ]
        )

    # Determine the earliest year represented in the input dates.
    minimum_year = int(
        local_dates.dt.year.min()
    )

    # Determine the latest year represented in the input dates.
    maximum_year = int(
        local_dates.dt.year.max()
    )

    # Create an Alberta holiday calendar covering the required years.
    #
    # Include one year before and after the data range to safely capture
    # holidays near dataset boundaries when constructing the before-holiday
    # and after-holiday flags.
    holiday_calendar = holidays.CA(
        subdiv="AB",
        years=range(
            minimum_year - 1,
            maximum_year + 2,
        ),
        observed=True,
    )

    # Build a list of dictionaries, one for each holiday in the calendar.
    # Each dictionary stores the holiday date and its corresponding name.
    rows = [
        {
            "local_date": pd.Timestamp(date_value),
            "holiday_name": str(name),
        }
        for date_value, name in sorted(
            holiday_calendar.items()
        )
    ]

    # Convert the list of holiday records into a DataFrame.
    holiday_table = pd.DataFrame(rows)

    # Return an empty DataFrame with the expected schema if no holidays exist.
    if holiday_table.empty:
        return pd.DataFrame(
            columns=[
                "local_date",
                "holiday_name",
            ]
        )

    # Create a local date column using pandas datetime values.
    holiday_table["local_date"] = pd.to_datetime(
        holiday_table["local_date"]
    ).dt.normalize()

    # Some dates can have multiple holiday names.
    #
    # For example, an observed holiday could occur on the same day as another
    # holiday. Group rows by date, combine duplicate holiday names, sort them
    # alphabetically, and join them into a single string separated by " | ".
    holiday_table = (
        holiday_table
        .groupby(
            "local_date",
            as_index=False,
        )["holiday_name"]
        .agg(
            lambda values: " | ".join(
                sorted(
                    set(values)
                )
            )
        )
        .sort_values("local_date")
        .reset_index(drop=True)
    )

    return holiday_table


# ============================================================================
# Feature construction
# ============================================================================

def build_calendar_features(
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the complete hourly calendar feature table."""

    # Verify that the input timestamp range is valid before generating
    # the calendar features.
    validate_range(
        start_utc,
        end_utc,
    )

    LOGGER.info(
        "Building hourly calendar range from %s through %s.",
        start_utc,
        end_utc,
    )

    # Create an hourly UTC DatetimeIndex covering the entire requested time
    # range, including both the start and end timestamps.
    timestamp_utc = pd.date_range(
        start=start_utc,
        end=end_utc,
        freq="h",
        tz="UTC",
    )

    # Create the calendar DataFrame using the hourly UTC timestamps.
    calendar = pd.DataFrame(
        {
            "timestamp_utc": timestamp_utc,
        }
    )

    # Convert the UTC timezone to Alberta local time.
    #
    # Calendar features depend on local time because demand, business activity,
    # holidays, and operating periods occur according to the local clock.
    timestamp_local = (
        calendar["timestamp_utc"]
        .dt.tz_convert(TIMEZONE)
    )

    # Create the Alberta-local timestamp column.
    calendar["timestamp_alberta"] = timestamp_local

    # ------------------------------------------------------------------------
    # Local-time primitives
    # ------------------------------------------------------------------------

    # Remove timezone information after converting to Alberta local time,
    # then normalize the timestamp to local midnight.
    #
    # This gives a timezone-naive local calendar date that can be matched
    # cleanly to the holiday table.
    calendar["local_date"] = (
        timestamp_local
        .dt.tz_localize(None)
        .dt.normalize()
    )

    calendar["year_alberta"] = (
        timestamp_local
        .dt.year
        .astype("int16")
    )

    calendar["quarter_alberta"] = (
        timestamp_local
        .dt.quarter
        .astype("int8")
    )

    calendar["month_alberta"] = (
        timestamp_local
        .dt.month
        .astype("int8")
    )

    calendar["month_name_alberta"] = (
        timestamp_local
        .dt.month_name()
        .str.lower()
        .astype("string")
    )

    calendar["week_of_year_alberta"] = (
        timestamp_local
        .dt.isocalendar()
        .week
        .astype("int16")
    )

    calendar["day_of_year_alberta"] = (
        timestamp_local
        .dt.dayofyear
        .astype("int16")
    )

    calendar["day_of_month_alberta"] = (
        timestamp_local
        .dt.day
        .astype("int8")
    )

    calendar["day_of_week_alberta"] = (
        timestamp_local
        .dt.dayofweek
        .astype("int8")
    )

    calendar["day_name_alberta"] = (
        timestamp_local
        .dt.day_name()
        .str.lower()
        .astype("string")
    )

    # Intraday hour according to the Alberta-local clock.
    calendar["hour_alberta"] = (
        timestamp_local
        .dt.hour
        .astype("int8")
    )

    # Alberta is UTC-7 during Mountain Standard Time and UTC-6 during
    # Mountain Daylight Time.
    calendar["utc_offset_hours"] = (
        timestamp_local
        .map(
            lambda value: (
                value.utcoffset().total_seconds()
                / 3600.0
            )
        )
        .astype("int8")
    )

    # Explicit daylight-saving-time flag.
    calendar["is_daylight_saving_time"] = (
        timestamp_local
        .map(
            lambda value: (
                value.dst().total_seconds() != 0
            )
        )
        .astype("int8")
    )

    # ------------------------------------------------------------------------
    # Weekday and calendar-boundary flags
    # ------------------------------------------------------------------------

    calendar["is_weekday"] = (
        calendar["day_of_week_alberta"]
        .between(0, 4)
        .astype("int8")
    )

    calendar["is_weekend"] = (
        1
        - calendar["is_weekday"]
    ).astype("int8")

    calendar["is_month_start"] = (
        timestamp_local
        .dt.is_month_start
        .astype("int8")
    )

    calendar["is_month_end"] = (
        timestamp_local
        .dt.is_month_end
        .astype("int8")
    )

    calendar["is_quarter_start"] = (
        timestamp_local
        .dt.is_quarter_start
        .astype("int8")
    )

    calendar["is_quarter_end"] = (
        timestamp_local
        .dt.is_quarter_end
        .astype("int8")
    )

    calendar["is_year_start"] = (
        timestamp_local
        .dt.is_year_start
        .astype("int8")
    )

    calendar["is_year_end"] = (
        timestamp_local
        .dt.is_year_end
        .astype("int8")
    )

    # ------------------------------------------------------------------------
    # Holidays
    # ------------------------------------------------------------------------

    LOGGER.info("Building Alberta statutory-holiday table.")

    holiday_table = build_alberta_holiday_table(
        calendar["local_date"]
    )

    holiday_name_map = (
        holiday_table
        .set_index("local_date")["holiday_name"]
        .to_dict()
    )

    holiday_dates = set(
        holiday_table["local_date"]
    )

    calendar["holiday_name"] = (
        calendar["local_date"]
        .map(holiday_name_map)
        .fillna("")
        .astype("string")
    )

    # Holiday effects.
    calendar["is_holiday"] = (
        calendar["local_date"]
        .isin(holiday_dates)
        .astype("int8")
    )

    calendar["is_day_before_holiday"] = (
        calendar["local_date"]
        .add(
            pd.Timedelta(days=1)
        )
        .isin(holiday_dates)
        .astype("int8")
    )

    calendar["is_day_after_holiday"] = (
        calendar["local_date"]
        .sub(
            pd.Timedelta(days=1)
        )
        .isin(holiday_dates)
        .astype("int8")
    )

    # Commercial and industrial business days.
    calendar["is_business_day"] = (
        calendar["is_weekday"].eq(1)
        & calendar["is_holiday"].eq(0)
    ).astype("int8")

    # ------------------------------------------------------------------------
    # Season and operating-period features
    # ------------------------------------------------------------------------

    calendar["season"] = meteorological_season(
        calendar["month_alberta"]
    )

    # Rough weather-driven demand-regime proxies.
    #
    # The flags are broad seasonal indicators rather than direct substitutes
    # for actual temperature or heating/cooling-degree variables.
    calendar["is_heating_season"] = (
        calendar["month_alberta"]
        .isin(
            [
                10,
                11,
                12,
                1,
                2,
                3,
                4,
            ]
        )
        .astype("int8")
    )

    calendar["is_cooling_season"] = (
        calendar["month_alberta"]
        .isin(
            [
                6,
                7,
                8,
            ]
        )
        .astype("int8")
    )

    hour = calendar["hour_alberta"]

    calendar["period_of_day"] = period_of_day(hour)

    # Overnight hours.
    calendar["is_overnight"] = (
        hour
        .between(0, 5)
        .astype("int8")
    )

    # Load, price, and scarcity behavior may differ during morning-ramp hours.
    calendar["is_morning_ramp"] = (
        hour
        .between(6, 9)
        .astype("int8")
    )

    # Business hours require both a business day and an hour between
    # 08:00 and 16:59 Alberta time.
    calendar["is_business_hour"] = (
        calendar["is_business_day"].eq(1)
        & hour.between(8, 16)
    ).astype("int8")

    calendar["is_afternoon"] = (
        hour
        .between(12, 16)
        .astype("int8")
    )

    calendar["is_evening_peak"] = (
        hour
        .between(17, 20)
        .astype("int8")
    )

    calendar["is_late_evening"] = (
        hour
        .between(21, 23)
        .astype("int8")
    )

    # ------------------------------------------------------------------------
    # Cyclical encodings
    # ------------------------------------------------------------------------

    """
    Cyclical encodings represent values that wrap around.

    Mathematically, hours 23:00 and 00:00 differ by 23 when represented as
    ordinary integers, but in reality they are only one hour apart. Placing
    hours around a circle allows 23:00 and 00:00 to sit next to each other,
    similar to their positions on a clock.

    The same concept can be applied to hours, weekdays, months, and days
    of the year.

    In power forecasting, electricity demand follows repeated cycles. This
    is also why some lagged features may be more predictive at t-24 hours
    than at t-18 hours.
    """

    two_pi = 2.0 * np.pi

    calendar["hour_sin"] = np.sin(
        two_pi
        * calendar["hour_alberta"]
        / 24.0
    )

    calendar["hour_cos"] = np.cos(
        two_pi
        * calendar["hour_alberta"]
        / 24.0
    )

    calendar["day_of_week_sin"] = np.sin(
        two_pi
        * calendar["day_of_week_alberta"]
        / 7.0
    )

    calendar["day_of_week_cos"] = np.cos(
        two_pi
        * calendar["day_of_week_alberta"]
        / 7.0
    )

    calendar["month_sin"] = np.sin(
        two_pi
        * (
            calendar["month_alberta"]
            - 1
        )
        / 12.0
    )

    calendar["month_cos"] = np.cos(
        two_pi
        * (
            calendar["month_alberta"]
            - 1
        )
        / 12.0
    )

    # A denominator of 366 preserves a consistent feature definition across
    # leap and non-leap years.
    calendar["day_of_year_sin"] = np.sin(
        two_pi
        * (
            calendar["day_of_year_alberta"]
            - 1
        )
        / 366.0
    )

    calendar["day_of_year_cos"] = np.cos(
        two_pi
        * (
            calendar["day_of_year_alberta"]
            - 1
        )
        / 366.0
    )

    LOGGER.info(
        "Calendar feature table constructed with %s rows and %s columns.",
        f"{len(calendar):,}",
        f"{len(calendar.columns):,}",
    )

    return (
        calendar,
        holiday_table,
    )


# ============================================================================
# Audit helpers
# ============================================================================

def build_numeric_summary(
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build descriptive statistics for all numeric calendar features.

    The table includes standard distribution statistics, selected
    percentiles, missing-value counts, missing percentages, and dtypes.
    """

    numeric_columns = (
        calendar
        .select_dtypes(
            include=[
                np.number,
                "bool",
            ]
        )
        .columns
        .tolist()
    )

    if not numeric_columns:
        return pd.DataFrame(
            columns=[
                "feature",
                "count",
                "mean",
                "std",
                "min",
                "p01",
                "25%",
                "median",
                "75%",
                "p99",
                "max",
                "missing_count",
                "missing_pct",
                "dtype",
            ]
        )

    # Create descriptive statistics for each numeric calendar column,
    # including selected percentiles.
    #
    # Transpose the result so each feature is represented by one row, then
    # standardize selected column names.
    summary = (
        calendar[numeric_columns]
        .describe(
            percentiles=[
                0.01,
                0.25,
                0.50,
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

    # Sum the number of missing values in each numeric column.
    summary["missing_count"] = (
        calendar[numeric_columns]
        .isna()
        .sum()
        .reindex(numeric_columns)
        .to_numpy()
    )

    # Percentage of missing values in each numeric column.
    summary["missing_pct"] = (
        summary["missing_count"]
        / len(calendar)
        * 100.0
    )

    # Store each numeric column's datatype.
    summary["dtype"] = [
        str(calendar[column].dtype)
        for column in numeric_columns
    ]

    return summary


# ============================================================================
# Audit
# ============================================================================

def audit_calendar_features(
    calendar: pd.DataFrame,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    bool,
]:
    """Audit calendar coverage, values, schema, and logical consistency."""

    LOGGER.info("Running calendar-feature audit.")

    rows: list[dict[str, Any]] = []

    expected_index = pd.date_range(
        start=start_utc,
        end=end_utc,
        freq="h",
        tz="UTC",
    )

    timestamp_index = pd.DatetimeIndex(
        pd.to_datetime(
            calendar["timestamp_utc"],
            utc=True,
        )
    )

    # ------------------------------------------------------------------------
    # Basic structure and coverage
    # ------------------------------------------------------------------------

    add_check(
        rows,
        "row_count_positive",
        len(calendar) > 0,
        observed=len(calendar),
        expected="> 0",
    )

    add_check(
        rows,
        "expected_row_count",
        len(calendar) == len(expected_index),
        observed=len(calendar),
        expected=len(expected_index),
    )

    add_check(
        rows,
        "required_columns_present",
        set(REQUIRED_COLUMNS).issubset(calendar.columns),
        observed="; ".join(
            sorted(
                set(REQUIRED_COLUMNS)
                - set(calendar.columns)
            )
        )
        or "all present",
        expected="all required columns present",
    )

    add_check(
        rows,
        "timestamp_start",
        timestamp_index.min() == start_utc,
        observed=str(timestamp_index.min()),
        expected=str(start_utc),
    )

    add_check(
        rows,
        "timestamp_end",
        timestamp_index.max() == end_utc,
        observed=str(timestamp_index.max()),
        expected=str(end_utc),
    )

    add_check(
        rows,
        "timestamps_unique",
        timestamp_index.is_unique,
        observed=int(
            timestamp_index
            .duplicated()
            .sum()
        ),
        expected=0,
    )

    add_check(
        rows,
        "timestamps_monotonic",
        timestamp_index.is_monotonic_increasing,
        observed=timestamp_index.is_monotonic_increasing,
        expected=True,
    )

    missing_hours = expected_index.difference(
        timestamp_index
    )

    extra_hours = timestamp_index.difference(
        expected_index
    )

    add_check(
        rows,
        "missing_hours",
        len(missing_hours) == 0,
        observed=len(missing_hours),
        expected=0,
    )

    add_check(
        rows,
        "extra_hours",
        len(extra_hours) == 0,
        observed=len(extra_hours),
        expected=0,
    )

    if len(timestamp_index) > 1:
        bad_spacing = int(
            pd.Series(timestamp_index)
            .diff()
            .dropna()
            .ne(
                pd.Timedelta(hours=1)
            )
            .sum()
        )
    else:
        bad_spacing = 0

    add_check(
        rows,
        "hourly_spacing",
        bad_spacing == 0,
        observed=bad_spacing,
        expected=0,
    )

    add_check(
        rows,
        "no_missing_values",
        not calendar.isna().any().any(),
        observed=int(
            calendar
            .isna()
            .sum()
            .sum()
        ),
        expected=0,
    )

    # ------------------------------------------------------------------------
    # Timestamp datatype and timezone checks
    # ------------------------------------------------------------------------

    timestamp_utc_dtype = calendar["timestamp_utc"].dtype
    timestamp_alberta_dtype = calendar["timestamp_alberta"].dtype

    add_check(
        rows,
        "timestamp_utc_timezone",
        str(timestamp_utc_dtype).endswith(", UTC]"),
        observed=str(timestamp_utc_dtype),
        expected="datetime64[*, UTC]",
    )

    add_check(
        rows,
        "timestamp_alberta_timezone",
        TIMEZONE in str(timestamp_alberta_dtype),
        observed=str(timestamp_alberta_dtype),
        expected=f"datetime64[*, {TIMEZONE}]",
    )

    # Confirm that converting the Alberta timestamps back to UTC reproduces
    # the canonical timestamp key exactly.
    alberta_back_to_utc = (
        calendar["timestamp_alberta"]
        .dt.tz_convert("UTC")
    )

    add_check(
        rows,
        "alberta_timestamp_matches_utc",
        alberta_back_to_utc.equals(
            calendar["timestamp_utc"]
        ),
        observed=int(
            (
                alberta_back_to_utc
                != calendar["timestamp_utc"]
            )
            .sum()
        ),
        expected=0,
    )

    # ------------------------------------------------------------------------
    # Binary feature validation
    # ------------------------------------------------------------------------

    for column in BINARY_COLUMNS:
        # Confirm the column is present.
        if column not in calendar.columns:
            add_check(
                rows,
                f"binary_column_present__{column}",
                False,
                observed=False,
                expected=True,
            )

            # Skip the remainder of the iteration because there is no column
            # available to validate.
            continue

        # Check for invalid values that are not either zero or one.
        invalid_count = int(
            (
                ~calendar[column]
                .isin(
                    [
                        0,
                        1,
                    ]
                )
            )
            .sum()
        )

        add_check(
            rows,
            f"binary_values__{column}",
            invalid_count == 0,
            observed=invalid_count,
            expected=0,
        )

    # ------------------------------------------------------------------------
    # Calendar logic
    # ------------------------------------------------------------------------

    weekday_weekend_sum = (
        calendar["is_weekday"]
        + calendar["is_weekend"]
    )

    # A row cannot be both a weekday and a weekend.
    add_check(
        rows,
        "weekday_weekend_complement",
        weekday_weekend_sum.eq(1).all(),
        observed=int(
            weekday_weekend_sum
            .ne(1)
            .sum()
        ),
        expected=0,
    )

    expected_business_day = (
        calendar["is_weekday"].eq(1)
        & calendar["is_holiday"].eq(0)
    ).astype("int8")

    # Business days should equal weekdays that are not statutory holidays.
    add_check(
        rows,
        "business_day_logic",
        calendar["is_business_day"]
        .eq(expected_business_day)
        .all(),
        observed=int(
            calendar["is_business_day"]
            .ne(expected_business_day)
            .sum()
        ),
        expected=0,
    )

    # Business hours must be contained within business days.
    invalid_business_hours = int(
        (
            calendar["is_business_hour"].eq(1)
            & calendar["is_business_day"].eq(0)
        )
        .sum()
    )

    add_check(
        rows,
        "business_hour_requires_business_day",
        invalid_business_hours == 0,
        observed=invalid_business_hours,
        expected=0,
    )

    # Validate all hour fields.
    add_check(
        rows,
        "valid_hour_range",
        calendar["hour_alberta"]
        .between(0, 23)
        .all(),
        observed=(
            f"min={calendar['hour_alberta'].min()}, "
            f"max={calendar['hour_alberta'].max()}"
        ),
        expected="[0, 23]",
    )

    add_check(
        rows,
        "valid_day_of_week_range",
        calendar["day_of_week_alberta"]
        .between(0, 6)
        .all(),
        observed=(
            f"min={calendar['day_of_week_alberta'].min()}, "
            f"max={calendar['day_of_week_alberta'].max()}"
        ),
        expected="[0, 6]",
    )

    add_check(
        rows,
        "valid_month_range",
        calendar["month_alberta"]
        .between(1, 12)
        .all(),
        observed=(
            f"min={calendar['month_alberta'].min()}, "
            f"max={calendar['month_alberta'].max()}"
        ),
        expected="[1, 12]",
    )

    add_check(
        rows,
        "valid_quarter_range",
        calendar["quarter_alberta"]
        .between(1, 4)
        .all(),
        observed=(
            f"min={calendar['quarter_alberta'].min()}, "
            f"max={calendar['quarter_alberta'].max()}"
        ),
        expected="[1, 4]",
    )

    add_check(
        rows,
        "valid_day_of_year_range",
        calendar["day_of_year_alberta"]
        .between(1, 366)
        .all(),
        observed=(
            f"min={calendar['day_of_year_alberta'].min()}, "
            f"max={calendar['day_of_year_alberta'].max()}"
        ),
        expected="[1, 366]",
    )

    add_check(
        rows,
        "valid_utc_offset",
        calendar["utc_offset_hours"]
        .isin(
            [
                -7,
                -6,
            ]
        )
        .all(),
        observed="; ".join(
            map(
                str,
                sorted(
                    calendar["utc_offset_hours"]
                    .unique()
                    .tolist()
                ),
            )
        ),
        expected="-7; -6",
    )

    # ------------------------------------------------------------------------
    # Season and period categories
    # ------------------------------------------------------------------------

    observed_seasons = set(
        calendar["season"]
        .unique()
    )

    add_check(
        rows,
        "valid_season_values",
        observed_seasons.issubset(
            VALID_SEASONS
        ),
        observed="; ".join(
            sorted(observed_seasons)
        ),
        expected="; ".join(
            sorted(VALID_SEASONS)
        ),
    )

    observed_periods = set(
        calendar["period_of_day"]
        .unique()
    )

    add_check(
        rows,
        "valid_period_of_day_values",
        observed_periods.issubset(
            VALID_PERIODS_OF_DAY
        ),
        observed="; ".join(
            sorted(observed_periods)
        ),
        expected="; ".join(
            sorted(VALID_PERIODS_OF_DAY)
        ),
    )

    # Every row should belong to a valid period_of_day category.
    add_check(
        rows,
        "period_of_day_no_unknown_values",
        not calendar["period_of_day"]
        .eq("unknown")
        .any(),
        observed=int(
            calendar["period_of_day"]
            .eq("unknown")
            .sum()
        ),
        expected=0,
    )

    # ------------------------------------------------------------------------
    # Cyclical encoding checks
    # ------------------------------------------------------------------------

    # Loop through every cyclical feature, find the smallest and largest value
    # in that column, and ensure all values remain within sine/cosine limits.
    for column in CYCLICAL_COLUMNS:
        if column not in calendar.columns:
            add_check(
                rows,
                f"cyclical_column_present__{column}",
                False,
                observed=False,
                expected=True,
            )
            continue

        column_min = float(
            calendar[column].min()
        )

        column_max = float(
            calendar[column].max()
        )

        add_check(
            rows,
            f"cyclical_range__{column}",
            (
                column_min >= -1.000001
                and column_max <= 1.000001
            ),
            observed=(
                f"min={column_min:.6g}, "
                f"max={column_max:.6g}"
            ),
            expected="[-1, 1]",
        )

    # ------------------------------------------------------------------------
    # Holiday checks
    # ------------------------------------------------------------------------

    holiday_rows = calendar.loc[
        calendar["is_holiday"].eq(1)
    ]

    # Every row marked as a holiday should contain a non-empty holiday name.
    empty_holiday_names = int(
        holiday_rows["holiday_name"]
        .str.len()
        .eq(0)
        .sum()
    )

    add_check(
        rows,
        "holiday_names_present_for_holiday_rows",
        empty_holiday_names == 0,
        observed=empty_holiday_names,
        expected=0,
    )

    # Every row that has a holiday name should also have the holiday flag.
    named_non_holiday_rows = int(
        (
            calendar["holiday_name"]
            .str.len()
            .gt(0)
            & calendar["is_holiday"].eq(0)
        )
        .sum()
    )

    add_check(
        rows,
        "holiday_flag_present_for_named_holidays",
        named_non_holiday_rows == 0,
        observed=named_non_holiday_rows,
        expected=0,
    )

    # ------------------------------------------------------------------------
    # Summary and final status
    # ------------------------------------------------------------------------

    summary = build_numeric_summary(calendar)

    audit = pd.DataFrame(rows)

    error_checks = audit.loc[
        audit["severity"].eq("error"),
        "pass",
    ]

    # Calculate the overall result using only error-level checks.
    #
    # Warning and informational checks would not fail the pipeline if they
    # were added in the future.
    audit_pass = (
        bool(error_checks.all())
        if not error_checks.empty
        else True
    )

    LOGGER.info(
        "Calendar audit completed: pass=%s, checks=%s, failed=%s.",
        audit_pass,
        f"{len(audit):,}",
        f"{int((~audit['pass']).sum()):,}",
    )

    return (
        audit,
        summary,
        audit_pass,
    )


# ============================================================================
# Reporting
# ============================================================================

def print_audit_report(
    calendar: pd.DataFrame,
    audit: pd.DataFrame,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
) -> None:
    """Print a compact pipeline report."""

    error_checks = audit.loc[
        audit["severity"].eq("error"),
        "pass",
    ]

    audit_pass = (
        bool(error_checks.all())
        if not error_checks.empty
        else True
    )

    failed = audit.loc[
        ~audit["pass"]
    ]

    holiday_dates = (
        calendar.loc[
            calendar["is_holiday"].eq(1),
            "local_date",
        ]
        .nunique()
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "CALENDAR FEATURES AUDIT"
    )

    print(
        "=" * 80
    )

    print(
        f"Overall pass    : {audit_pass}"
    )

    print(
        f"Rows            : {len(calendar):,}"
    )

    print(
        f"Columns         : {len(calendar.columns):,}"
    )

    print(
        f"Expected rows   : {expected_hour_count(start_utc, end_utc):,}"
    )

    print(
        f"Start UTC       : {start_utc}"
    )

    print(
        f"End UTC         : {end_utc}"
    )

    print(
        f"Start Alberta   : {calendar['timestamp_alberta'].min()}"
    )

    print(
        f"End Alberta     : {calendar['timestamp_alberta'].max()}"
    )

    print(
        f"Holiday dates   : {holiday_dates:,}"
    )

    print(
        f"DST hours       : "
        f"{int(calendar['is_daylight_saving_time'].sum()):,}"
    )

    print(
        f"Audit checks    : {len(audit):,}"
    )

    print(
        f"Failed checks   : {len(failed):,}"
    )

    print(
        "\nFailed checks:"
    )

    if failed.empty:
        print(
            "  None"
        )
    else:
        for _, row in failed.iterrows():
            print(
                f"  - {row['check']} "
                f"[{row['severity']}] "
                f"observed={row['observed']} "
                f"expected={row['expected']}"
            )

    print(
        "=" * 80
    )


def print_pipeline_result(
    result: dict[str, Any],
) -> None:
    """Print the final pipeline result dictionary."""

    print(
        "\n"
        + "=" * 80
    )

    print(
        "CALENDAR FEATURE RESULT"
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


# ============================================================================
# Output helpers
# ============================================================================

def save_audit_outputs(
    audit: pd.DataFrame,
    summary: pd.DataFrame,
    holiday_table: pd.DataFrame,
) -> None:
    """Write calendar audit and metadata tables."""

    ensure_output_directories()

    LOGGER.info(
        "Writing audit checks to %s.",
        AUDIT_FILE,
    )

    audit.to_csv(
        AUDIT_FILE,
        index=False,
    )

    LOGGER.info(
        "Writing numeric summary to %s.",
        SUMMARY_FILE,
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    LOGGER.info(
        "Writing holiday-date table to %s.",
        HOLIDAY_DATES_FILE,
    )

    holiday_table.to_csv(
        HOLIDAY_DATES_FILE,
        index=False,
    )


def save_feature_outputs(
    calendar: pd.DataFrame,
    write_csv: bool,
) -> None:
    """Write canonical Parquet and optional CSV feature outputs."""

    ensure_output_directories()

    LOGGER.info(
        "Writing canonical calendar Parquet to %s.",
        OUTPUT_PARQUET,
    )

    calendar.to_parquet(
        OUTPUT_PARQUET,
        index=False,
    )

    if write_csv:
        LOGGER.info(
            "Writing optional calendar CSV to %s.",
            OUTPUT_CSV,
        )

        calendar.to_csv(
            OUTPUT_CSV,
            index=False,
        )


def existing_outputs_satisfy_request(
    write_csv: bool,
) -> bool:
    """
    Return True when all requested feature outputs already exist.

    Parquet is always required.

    CSV is required only when write_csv=True.
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
    Load an existing canonical Parquet file when only a missing CSV is needed.

    This avoids rebuilding the full calendar table simply because the user
    later requests the optional CSV representation.
    """

    LOGGER.info(
        "Loading existing calendar Parquet from %s.",
        OUTPUT_PARQUET,
    )

    calendar = pd.read_parquet(
        OUTPUT_PARQUET
    )

    if "timestamp_utc" in calendar.columns:
        calendar["timestamp_utc"] = pd.to_datetime(
            calendar["timestamp_utc"],
            utc=True,
        )

    if "timestamp_alberta" in calendar.columns:
        calendar["timestamp_alberta"] = pd.to_datetime(
            calendar["timestamp_alberta"],
            utc=True,
        ).dt.tz_convert(TIMEZONE)

    if "local_date" in calendar.columns:
        calendar["local_date"] = pd.to_datetime(
            calendar["local_date"]
        )

    return calendar


# ============================================================================
# Pipeline
# ============================================================================

def process_calendar_features(
    start: str = DEFAULT_START_UTC,
    end: str = DEFAULT_END_UTC,
    overwrite: bool = False,
    write_csv: bool = False,
) -> dict[str, Any]:
    """Build, audit, and save canonical calendar features."""

    started = time.perf_counter()

    LOGGER.info("Starting calendar-feature pipeline.")
    LOGGER.debug("Project root: %s", PROJECT_ROOT)
    LOGGER.debug("Feature output directory: %s", OUTPUT_DIR)
    LOGGER.debug("Audit output directory: %s", AUDIT_DIR)

    start_utc = parse_utc_timestamp(
        start,
        "start",
    )

    end_utc = parse_utc_timestamp(
        end,
        "end",
    )

    validate_range(
        start_utc,
        end_utc,
    )

    LOGGER.info(
        "Requested UTC coverage: %s through %s.",
        start_utc,
        end_utc,
    )

    LOGGER.info(
        "Expected inclusive row count: %s.",
        f"{expected_hour_count(start_utc, end_utc):,}",
    )

    ensure_output_directories()

    # ------------------------------------------------------------------------
    # Existing-output handling
    # ------------------------------------------------------------------------

    # Parquet is the canonical output of the pipeline.
    #
    # Skip the pipeline only if every requested feature output already exists.
    # This means that a later --write-csv request can create the CSV even when
    # the Parquet was produced during an earlier run.
    if (
        not overwrite
        and existing_outputs_satisfy_request(
            write_csv=write_csv
        )
    ):
        LOGGER.info(
            "Requested feature outputs already exist. "
            "Use --overwrite to rebuild them."
        )

        return {
            "dataset": DATASET_NAME,
            "status": "skipped_existing",
            "pass": True,
            "rows": None,
            "columns": None,
            "requested_start_utc": str(start_utc),
            "requested_end_utc": str(end_utc),
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
            "summary_file": (
                str(SUMMARY_FILE)
                if SUMMARY_FILE.exists()
                else "not available"
            ),
            "holiday_dates_file": (
                str(HOLIDAY_DATES_FILE)
                if HOLIDAY_DATES_FILE.exists()
                else "not available"
            ),
            "processing_seconds": round(
                time.perf_counter()
                - started,
                3,
            ),
        }

    # When Parquet already exists and the only missing requested output is CSV,
    # create the CSV directly from the canonical Parquet.
    #
    # The calendar is not rebuilt because the Parquet has already passed the
    # pipeline that originally created it.
    if (
        not overwrite
        and OUTPUT_PARQUET.exists()
        and write_csv
        and not OUTPUT_CSV.exists()
    ):
        calendar = read_existing_parquet_for_csv()

        LOGGER.info(
            "Creating missing CSV from existing canonical Parquet."
        )

        calendar.to_csv(
            OUTPUT_CSV,
            index=False,
        )

        return {
            "dataset": DATASET_NAME,
            "status": "csv_created_from_existing_parquet",
            "pass": True,
            "rows": len(calendar),
            "columns": len(calendar.columns),
            "start_utc": str(
                calendar["timestamp_utc"].min()
            ),
            "end_utc": str(
                calendar["timestamp_utc"].max()
            ),
            "parquet_file": str(OUTPUT_PARQUET),
            "csv_file": str(OUTPUT_CSV),
            "audit_file": (
                str(AUDIT_FILE)
                if AUDIT_FILE.exists()
                else "not available"
            ),
            "summary_file": (
                str(SUMMARY_FILE)
                if SUMMARY_FILE.exists()
                else "not available"
            ),
            "holiday_dates_file": (
                str(HOLIDAY_DATES_FILE)
                if HOLIDAY_DATES_FILE.exists()
                else "not available"
            ),
            "processing_seconds": round(
                time.perf_counter()
                - started,
                3,
            ),
        }

    # ------------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------------

    calendar, holiday_table = build_calendar_features(
        start_utc,
        end_utc,
    )

    # ------------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------------

    (
        audit,
        summary,
        audit_pass,
    ) = audit_calendar_features(
        calendar,
        start_utc,
        end_utc,
    )

    print_audit_report(
        calendar,
        audit,
        start_utc,
        end_utc,
    )

    # Audit outputs are written regardless of whether the audit passes.
    #
    # This preserves the evidence needed to diagnose a failed pipeline run.
    save_audit_outputs(
        audit,
        summary,
        holiday_table,
    )

    if not audit_pass:
        LOGGER.error(
            "Calendar-feature audit failed. "
            "Feature outputs were not written."
        )

        return {
            "dataset": DATASET_NAME,
            "status": "audit_failed",
            "pass": False,
            "rows": len(calendar),
            "columns": len(calendar.columns),
            "requested_start_utc": str(start_utc),
            "requested_end_utc": str(end_utc),
            "audit_file": str(AUDIT_FILE),
            "summary_file": str(SUMMARY_FILE),
            "holiday_dates_file": str(HOLIDAY_DATES_FILE),
            "parquet_file": "not written",
            "csv_file": "not written",
            "processing_seconds": round(
                time.perf_counter()
                - started,
                3,
            ),
        }

    # ------------------------------------------------------------------------
    # Feature output
    # ------------------------------------------------------------------------

    save_feature_outputs(
        calendar,
        write_csv=write_csv,
    )

    processing_seconds = round(
        time.perf_counter()
        - started,
        3,
    )

    LOGGER.info(
        "Calendar-feature pipeline completed successfully in %.3f seconds.",
        processing_seconds,
    )

    return {
        "dataset": DATASET_NAME,
        "status": "saved",
        "pass": True,
        "rows": len(calendar),
        "columns": len(calendar.columns),
        "start_utc": str(
            calendar["timestamp_utc"].min()
        ),
        "end_utc": str(
            calendar["timestamp_utc"].max()
        ),
        "start_alberta": str(
            calendar["timestamp_alberta"].min()
        ),
        "end_alberta": str(
            calendar["timestamp_alberta"].max()
        ),
        "holiday_dates": int(
            calendar.loc[
                calendar["is_holiday"].eq(1),
                "local_date",
            ]
            .nunique()
        ),
        "parquet_file": str(OUTPUT_PARQUET),
        "csv_file": (
            str(OUTPUT_CSV)
            if write_csv
            else "not requested"
        ),
        "audit_file": str(AUDIT_FILE),
        "summary_file": str(SUMMARY_FILE),
        "holiday_dates_file": str(HOLIDAY_DATES_FILE),
        "processing_seconds": processing_seconds,
    }


# ============================================================================
# CLI
# ============================================================================

def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Build canonical hourly Alberta calendar features."
        )
    )

    parser.add_argument(
        "--start",
        default=DEFAULT_START_UTC,
        help=(
            "Inclusive timezone-aware UTC start timestamp. "
            f"Default: {DEFAULT_START_UTC}"
        ),
    )

    parser.add_argument(
        "--end",
        default=DEFAULT_END_UTC,
        help=(
            "Inclusive timezone-aware UTC end timestamp. "
            f"Default: {DEFAULT_END_UTC}"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Rebuild and overwrite existing calendar-feature outputs."
        ),
    )

    parser.add_argument(
        "--write-csv",
        action="store_true",
        help=(
            "Also write the full calendar feature table to CSV."
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


def main() -> None:
    """Run the calendar-feature pipeline from the command line."""

    parser = build_argument_parser()

    args = parser.parse_args()

    configure_logging(
        verbose=args.verbose
    )

    try:
        result = process_calendar_features(
            start=args.start,
            end=args.end,
            overwrite=args.overwrite,
            write_csv=args.write_csv,
        )

    except Exception:
        LOGGER.exception(
            "Calendar-feature pipeline terminated with an unexpected error."
        )
        raise

    print_pipeline_result(result)

    if not result.get("pass", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()