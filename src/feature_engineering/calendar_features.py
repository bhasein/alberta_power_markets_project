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
data/features/calendar/calendar_features_hourly.parquet
data/features/calendar/calendar_features_hourly.csv          optional
data/audits/calendar_features_audit_checks.csv
data/audits/calendar_features_summary.csv
data/audits/calendar_holiday_dates.csv

Run
---
python src/feature_engineering/calendar_features.py

Overwrite existing output:

python src/feature_engineering/calendar_features.py --overwrite

Custom range:

python src/feature_engineering/calendar_features.py \
    --start "2015-01-01 00:00:00+00:00" \
    --end "2026-06-30 23:00:00+00:00" \
    --overwrite

Write CSV as well:

python src/feature_engineering/calendar_features.py \
    --overwrite \
    --write-csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

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
# Paths
# ============================================================================

PROJECT_ROOT = Path(
    "/Users/brodiehasein/alberta_power_markets_project"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/features/calendar"
)

OUTPUT_PARQUET = (
    OUTPUT_DIR
    / "calendar_features_hourly.parquet"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "calendar_features_hourly.csv"
)

AUDIT_DIR = (
    PROJECT_ROOT
    / "data/audits"
)

AUDIT_FILE = (
    AUDIT_DIR
    / "calendar_features_audit_checks.csv"
)

SUMMARY_FILE = (
    AUDIT_DIR
    / "calendar_features_summary.csv"
)

HOLIDAY_DATES_FILE = (
    AUDIT_DIR
    / "calendar_holiday_dates.csv"
)


# ============================================================================
# Configuration
# ============================================================================

TIMEZONE = "America/Edmonton"

DEFAULT_START_UTC = "2015-01-01 00:00:00+00:00"
DEFAULT_END_UTC = "2026-06-30 23:00:00+00:00"

VALID_SEASONS = {
    "winter",
    "spring",
    "summer",
    "fall",
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


# ============================================================================
# General helpers
# ============================================================================

def add_check(
    rows: list[dict],
    check: str,
    passed: bool,
    observed=None,
    expected=None,
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
    """Parse a timezone-aware UTC timestamp."""

    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is None:
        raise ValueError(
            f"{field_name} must include timezone information. "
            f"Example: 2015-01-01 00:00:00+00:00"
        )

    return timestamp.tz_convert("UTC")


def validate_range(
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
) -> None:
    """Validate requested calendar coverage."""

    if end_utc < start_utc:
        raise ValueError(
            f"End timestamp must be on or after start timestamp. "
            f"Observed start={start_utc}, end={end_utc}"
        )

    if start_utc.minute != 0 or start_utc.second != 0:
        raise ValueError(
            f"Start timestamp must be aligned to the hour: {start_utc}"
        )

    if end_utc.minute != 0 or end_utc.second != 0:
        raise ValueError(
            f"End timestamp must be aligned to the hour: {end_utc}"
        )


def meteorological_season(
    month: pd.Series,
) -> pd.Series:
    """Map month numbers to meteorological seasons."""

    conditions = [
        month.isin([12, 1, 2]),
        month.isin([3, 4, 5]),
        month.isin([6, 7, 8]),
        month.isin([9, 10, 11]),
    ]

    choices = [
        "winter",
        "spring",
        "summer",
        "fall",
    ]

    return pd.Series(
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

    conditions = [
        hour.between(0, 5),
        hour.between(6, 9),
        hour.between(10, 15),
        hour.between(16, 19),
        hour.between(20, 23),
    ]

    choices = [
        "overnight",
        "morning_ramp",
        "daytime",
        "evening_peak",
        "late_evening",
    ]

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

    minimum_year = int(
        local_dates.dt.year.min()
    )

    maximum_year = int(
        local_dates.dt.year.max()
    )

    holiday_calendar = holidays.CA(
        subdiv="AB",
        years=range(
            minimum_year - 1,
            maximum_year + 2,
        ),
        observed=True,
    )

    rows = [
        {
            "local_date": pd.Timestamp(date_value),
            "holiday_name": str(name),
        }
        for date_value, name in sorted(
            holiday_calendar.items()
        )
    ]

    holiday_table = pd.DataFrame(rows)

    if holiday_table.empty:
        return pd.DataFrame(
            columns=[
                "local_date",
                "holiday_name",
            ]
        )

    holiday_table["local_date"] = pd.to_datetime(
        holiday_table["local_date"]
    )

    # A date can occasionally receive multiple holiday labels.
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
        .sort_values(
            "local_date"
        )
        .reset_index(
            drop=True
        )
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

    validate_range(
        start_utc,
        end_utc,
    )

    timestamp_utc = pd.date_range(
        start=start_utc,
        end=end_utc,
        freq="h",
        tz="UTC",
    )

    calendar = pd.DataFrame(
        {
            "timestamp_utc": timestamp_utc,
        }
    )

    timestamp_local = (
        calendar["timestamp_utc"]
        .dt.tz_convert(
            TIMEZONE
        )
    )

    calendar["timestamp_alberta"] = (
        timestamp_local
    )

    # ------------------------------------------------------------------------
    # Local-time primitives
    # ------------------------------------------------------------------------

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

    calendar["hour_alberta"] = (
        timestamp_local
        .dt.hour
        .astype("int8")
    )

    calendar["utc_offset_hours"] = (
        timestamp_local
        .map(
            lambda value: (
                value.utcoffset()
                .total_seconds()
                / 3600.0
            )
        )
        .astype("int8")
    )

    calendar["is_daylight_saving_time"] = (
        timestamp_local
        .map(
            lambda value: (
                value.dst()
                .total_seconds()
                != 0
            )
        )
        .astype("int8")
    )

    # ------------------------------------------------------------------------
    # Weekday and calendar-boundary flags
    # ------------------------------------------------------------------------

    calendar["is_weekday"] = (
        calendar[
            "day_of_week_alberta"
        ]
        .between(
            0,
            4,
        )
        .astype("int8")
    )

    calendar["is_weekend"] = (
        (
            1
            - calendar["is_weekday"]
        )
        .astype("int8")
    )

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

    holiday_table = (
        build_alberta_holiday_table(
            calendar["local_date"]
        )
    )

    holiday_name_map = (
        holiday_table
        .set_index(
            "local_date"
        )["holiday_name"]
        .to_dict()
    )

    holiday_dates = set(
        holiday_table[
            "local_date"
        ]
    )

    calendar["holiday_name"] = (
        calendar["local_date"]
        .map(
            holiday_name_map
        )
        .fillna("")
        .astype("string")
    )

    calendar["is_holiday"] = (
        calendar[
            "local_date"
        ]
        .isin(
            holiday_dates
        )
        .astype("int8")
    )

    calendar["is_day_before_holiday"] = (
        (
            calendar["local_date"]
            + pd.Timedelta(
                days=1
            )
        )
        .isin(
            holiday_dates
        )
        .astype("int8")
    )

    calendar["is_day_after_holiday"] = (
        (
            calendar["local_date"]
            - pd.Timedelta(
                days=1
            )
        )
        .isin(
            holiday_dates
        )
        .astype("int8")
    )

    calendar["is_business_day"] = (
        (
            calendar["is_weekday"].eq(1)
            & calendar["is_holiday"].eq(0)
        )
        .astype("int8")
    )

    # ------------------------------------------------------------------------
    # Season and operating-period features
    # ------------------------------------------------------------------------

    calendar["season"] = (
        meteorological_season(
            calendar[
                "month_alberta"
            ]
        )
    )

    calendar["is_heating_season"] = (
        calendar[
            "month_alberta"
        ]
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
        calendar[
            "month_alberta"
        ]
        .isin(
            [
                6,
                7,
                8,
            ]
        )
        .astype("int8")
    )

    hour = calendar[
        "hour_alberta"
    ]

    calendar["period_of_day"] = (
        period_of_day(
            hour
        )
    )

    calendar["is_overnight"] = (
        hour
        .between(
            0,
            5,
        )
        .astype("int8")
    )

    calendar["is_morning_ramp"] = (
        hour
        .between(
            6,
            9,
        )
        .astype("int8")
    )

    calendar["is_business_hour"] = (
        (
            calendar["is_business_day"].eq(1)
            & hour.between(
                8,
                16,
            )
        )
        .astype("int8")
    )

    calendar["is_afternoon"] = (
        hour
        .between(
            12,
            16,
        )
        .astype("int8")
    )

    calendar["is_evening_peak"] = (
        hour
        .between(
            17,
            20,
        )
        .astype("int8")
    )

    calendar["is_late_evening"] = (
        hour
        .between(
            21,
            23,
        )
        .astype("int8")
    )

    # ------------------------------------------------------------------------
    # Cyclical encodings
    # ------------------------------------------------------------------------

    calendar["hour_sin"] = np.sin(
        2.0
        * np.pi
        * calendar["hour_alberta"]
        / 24.0
    )

    calendar["hour_cos"] = np.cos(
        2.0
        * np.pi
        * calendar["hour_alberta"]
        / 24.0
    )

    calendar["day_of_week_sin"] = np.sin(
        2.0
        * np.pi
        * calendar["day_of_week_alberta"]
        / 7.0
    )

    calendar["day_of_week_cos"] = np.cos(
        2.0
        * np.pi
        * calendar["day_of_week_alberta"]
        / 7.0
    )

    calendar["month_sin"] = np.sin(
        2.0
        * np.pi
        * (
            calendar["month_alberta"]
            - 1
        )
        / 12.0
    )

    calendar["month_cos"] = np.cos(
        2.0
        * np.pi
        * (
            calendar["month_alberta"]
            - 1
        )
        / 12.0
    )

    # 366 preserves a consistent denominator across leap and non-leap years.
    calendar["day_of_year_sin"] = np.sin(
        2.0
        * np.pi
        * (
            calendar["day_of_year_alberta"]
            - 1
        )
        / 366.0
    )

    calendar["day_of_year_cos"] = np.cos(
        2.0
        * np.pi
        * (
            calendar["day_of_year_alberta"]
            - 1
        )
        / 366.0
    )

    return (
        calendar,
        holiday_table,
    )


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
    """Audit calendar coverage, values, and logical consistency."""

    rows: list[dict] = []

    expected_index = pd.date_range(
        start=start_utc,
        end=end_utc,
        freq="h",
        tz="UTC",
    )

    timestamp_index = pd.DatetimeIndex(
        pd.to_datetime(
            calendar[
                "timestamp_utc"
            ],
            utc=True,
        )
    )

    add_check(
        rows,
        "row_count_positive",
        len(calendar) > 0,
        len(calendar),
        "> 0",
    )

    add_check(
        rows,
        "expected_row_count",
        len(calendar)
        == len(
            expected_index
        ),
        len(calendar),
        len(
            expected_index
        ),
    )

    add_check(
        rows,
        "timestamp_start",
        timestamp_index.min()
        == start_utc,
        str(
            timestamp_index.min()
        ),
        str(
            start_utc
        ),
    )

    add_check(
        rows,
        "timestamp_end",
        timestamp_index.max()
        == end_utc,
        str(
            timestamp_index.max()
        ),
        str(
            end_utc
        ),
    )

    add_check(
        rows,
        "timestamps_unique",
        timestamp_index.is_unique,
        int(
            timestamp_index
            .duplicated()
            .sum()
        ),
        0,
    )

    add_check(
        rows,
        "timestamps_monotonic",
        timestamp_index.is_monotonic_increasing,
        timestamp_index.is_monotonic_increasing,
        True,
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
        len(
            missing_hours
        )
        == 0,
        len(
            missing_hours
        ),
        0,
    )

    add_check(
        rows,
        "extra_hours",
        len(
            extra_hours
        )
        == 0,
        len(
            extra_hours
        ),
        0,
    )

    if len(
        timestamp_index
    ) > 1:
        bad_spacing = (
            pd.Series(
                timestamp_index
            )
            .diff()
            .dropna()
            .ne(
                pd.Timedelta(
                    hours=1
                )
            )
            .sum()
        )

        add_check(
            rows,
            "hourly_spacing",
            bad_spacing == 0,
            int(
                bad_spacing
            ),
            0,
        )

    add_check(
        rows,
        "no_missing_values",
        not calendar.isna().any().any(),
        int(
            calendar
            .isna()
            .sum()
            .sum()
        ),
        0,
    )

    for column in BINARY_COLUMNS:
        if column not in calendar.columns:
            add_check(
                rows,
                f"binary_column_present__{column}",
                False,
                False,
                True,
            )
            continue

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
            invalid_count,
            0,
        )

    add_check(
        rows,
        "weekday_weekend_complement",
        (
            calendar[
                "is_weekday"
            ]
            + calendar[
                "is_weekend"
            ]
        )
        .eq(1)
        .all(),
        int(
            (
                (
                    calendar[
                        "is_weekday"
                    ]
                    + calendar[
                        "is_weekend"
                    ]
                )
                != 1
            )
            .sum()
        ),
        0,
    )

    add_check(
        rows,
        "business_day_logic",
        (
            calendar[
                "is_business_day"
            ]
            == (
                calendar[
                    "is_weekday"
                ].eq(1)
                & calendar[
                    "is_holiday"
                ].eq(0)
            ).astype(
                "int8"
            )
        )
        .all(),
        int(
            (
                calendar[
                    "is_business_day"
                ]
                != (
                    calendar[
                        "is_weekday"
                    ].eq(1)
                    & calendar[
                        "is_holiday"
                    ].eq(0)
                ).astype(
                    "int8"
                )
            )
            .sum()
        ),
        0,
    )

    add_check(
        rows,
        "valid_hour_range",
        calendar[
            "hour_alberta"
        ]
        .between(
            0,
            23,
        )
        .all(),
        (
            f"min={calendar['hour_alberta'].min()}, "
            f"max={calendar['hour_alberta'].max()}"
        ),
        "[0, 23]",
    )

    add_check(
        rows,
        "valid_day_of_week_range",
        calendar[
            "day_of_week_alberta"
        ]
        .between(
            0,
            6,
        )
        .all(),
        (
            f"min={calendar['day_of_week_alberta'].min()}, "
            f"max={calendar['day_of_week_alberta'].max()}"
        ),
        "[0, 6]",
    )

    add_check(
        rows,
        "valid_month_range",
        calendar[
            "month_alberta"
        ]
        .between(
            1,
            12,
        )
        .all(),
        (
            f"min={calendar['month_alberta'].min()}, "
            f"max={calendar['month_alberta'].max()}"
        ),
        "[1, 12]",
    )

    observed_seasons = set(
        calendar[
            "season"
        ]
        .unique()
    )

    add_check(
        rows,
        "valid_season_values",
        observed_seasons.issubset(
            VALID_SEASONS
        ),
        "; ".join(
            sorted(
                observed_seasons
            )
        ),
        "; ".join(
            sorted(
                VALID_SEASONS
            )
        ),
    )

    for column in CYCLICAL_COLUMNS:
        if column not in calendar.columns:
            continue

        column_min = float(
            calendar[column]
            .min()
        )

        column_max = float(
            calendar[column]
            .max()
        )

        add_check(
            rows,
            f"cyclical_range__{column}",
            (
                column_min
                >= -1.000001
                and column_max
                <= 1.000001
            ),
            (
                f"min={column_min:.6g}, "
                f"max={column_max:.6g}"
            ),
            "[-1, 1]",
        )

    add_check(
        rows,
        "holiday_names_present_for_holiday_rows",
        (
            calendar.loc[
                calendar[
                    "is_holiday"
                ].eq(1),
                "holiday_name",
            ]
            .str.len()
            .gt(0)
            .all()
        ),
        int(
            (
                calendar.loc[
                    calendar[
                        "is_holiday"
                    ].eq(1),
                    "holiday_name",
                ]
                .str.len()
                .eq(0)
            )
            .sum()
        ),
        0,
    )

    period_categories = {
        "overnight",
        "morning_ramp",
        "daytime",
        "evening_peak",
        "late_evening",
    }

    observed_periods = set(
        calendar[
            "period_of_day"
        ]
        .unique()
    )

    add_check(
        rows,
        "valid_period_of_day_values",
        observed_periods.issubset(
            period_categories
        ),
        "; ".join(
            sorted(
                observed_periods
            )
        ),
        "; ".join(
            sorted(
                period_categories
            )
        ),
    )

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

    summary = (
        calendar[
            numeric_columns
        ]
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

    summary[
        "missing_count"
    ] = (
        calendar[
            numeric_columns
        ]
        .isna()
        .sum()
        .values
    )

    summary[
        "missing_pct"
    ] = (
        summary[
            "missing_count"
        ]
        / len(
            calendar
        )
        * 100.0
    )

    summary[
        "dtype"
    ] = [
        str(
            calendar[column]
            .dtype
        )
        for column
        in numeric_columns
    ]

    audit = pd.DataFrame(
        rows
    )

    error_checks = audit.loc[
        audit[
            "severity"
        ].eq(
            "error"
        ),
        "pass",
    ]

    audit_pass = (
        bool(
            error_checks.all()
        )
        if not error_checks.empty
        else True
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
        audit[
            "severity"
        ].eq(
            "error"
        ),
        "pass",
    ]

    audit_pass = (
        bool(
            error_checks.all()
        )
        if not error_checks.empty
        else True
    )

    failed = audit.loc[
        ~audit[
            "pass"
        ]
    ]

    holiday_dates = (
        calendar.loc[
            calendar[
                "is_holiday"
            ].eq(1),
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
        f"Start UTC       : {start_utc}"
    )

    print(
        f"End UTC         : {end_utc}"
    )

    print(
        f"Start Alberta   : "
        f"{calendar['timestamp_alberta'].min()}"
    )

    print(
        f"End Alberta     : "
        f"{calendar['timestamp_alberta'].max()}"
    )

    print(
        f"Holiday dates   : {holiday_dates:,}"
    )

    print(
        f"DST hours       : "
        f"{int(calendar['is_daylight_saving_time'].sum()):,}"
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


# ============================================================================
# Pipeline
# ============================================================================

def process_calendar_features(
    start: str = DEFAULT_START_UTC,
    end: str = DEFAULT_END_UTC,
    overwrite: bool = False,
    write_csv: bool = False,
) -> dict:
    """Build, audit, and save calendar features."""

    started = time.perf_counter()

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

    if (
        OUTPUT_PARQUET.exists()
        and not overwrite
    ):
        return {
            "dataset": "calendar_features",
            "status": "skipped_existing",
            "pass": True,
            "parquet_file": str(
                OUTPUT_PARQUET
            ),
            "csv_file": (
                str(
                    OUTPUT_CSV
                )
                if OUTPUT_CSV.exists()
                else "not written"
            ),
        }

    calendar, holiday_table = (
        build_calendar_features(
            start_utc,
            end_utc,
        )
    )

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

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit.to_csv(
        AUDIT_FILE,
        index=False,
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    holiday_table.to_csv(
        HOLIDAY_DATES_FILE,
        index=False,
    )

    if not audit_pass:
        return {
            "dataset": "calendar_features",
            "status": "audit_failed",
            "pass": False,
            "audit_file": str(
                AUDIT_FILE
            ),
            "summary_file": str(
                SUMMARY_FILE
            ),
            "holiday_dates_file": str(
                HOLIDAY_DATES_FILE
            ),
            "processing_seconds": round(
                time.perf_counter()
                - started,
                3,
            ),
        }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    calendar.to_parquet(
        OUTPUT_PARQUET,
        index=False,
    )

    if write_csv:
        calendar.to_csv(
            OUTPUT_CSV,
            index=False,
        )

    return {
        "dataset": "calendar_features",
        "status": "saved",
        "pass": True,
        "rows": len(
            calendar
        ),
        "columns": len(
            calendar.columns
        ),
        "start_utc": str(
            calendar[
                "timestamp_utc"
            ].min()
        ),
        "end_utc": str(
            calendar[
                "timestamp_utc"
            ].max()
        ),
        "start_alberta": str(
            calendar[
                "timestamp_alberta"
            ].min()
        ),
        "end_alberta": str(
            calendar[
                "timestamp_alberta"
            ].max()
        ),
        "holiday_dates": int(
            calendar.loc[
                calendar[
                    "is_holiday"
                ].eq(1),
                "local_date",
            ]
            .nunique()
        ),
        "parquet_file": str(
            OUTPUT_PARQUET
        ),
        "csv_file": (
            str(
                OUTPUT_CSV
            )
            if write_csv
            else "not written"
        ),
        "audit_file": str(
            AUDIT_FILE
        ),
        "summary_file": str(
            SUMMARY_FILE
        ),
        "holiday_dates_file": str(
            HOLIDAY_DATES_FILE
        ),
        "processing_seconds": round(
            time.perf_counter()
            - started,
            3,
        ),
    }


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
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
            "Overwrite an existing calendar-feature Parquet file."
        ),
    )

    parser.add_argument(
        "--write-csv",
        action="store_true",
        help=(
            "Also write the full calendar table to CSV."
        ),
    )

    args = parser.parse_args()

    result = process_calendar_features(
        start=args.start,
        end=args.end,
        overwrite=args.overwrite,
        write_csv=args.write_csv,
    )

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


if __name__ == "__main__":
    main()
