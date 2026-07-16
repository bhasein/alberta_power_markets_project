"""
================================================================================
PURPOSE:
    Preprocess and audit AESO hourly area-and-region load data.

WHY THIS FILE IS USEFUL:
    AESO area-load history arrives across multiple file formats (Excel, CSV,
    TSV, plain text) and multiple timestamp layouts (a newer DT_MST column,
    and an older DATE + HOUR ENDING pair). This file automatically discovers
    which raw files actually contain area-load data, standardizes each one
    into a common schema, reconciles any overlapping/duplicate hours across
    files, fills in isolated single-hour gaps in the timeline, and records
    which source file every row came from. The result is one continuous,
    trustworthy hourly load series suitable for downstream modeling.

PIPELINE OVERVIEW:
    raw AESO area-load files (mixed formats/timestamp layouts)
        --> discover_raw_files()          scans the raw folder for compatible files
        --> read_raw_file()               reads each file in its native format
        --> clean_area_load_file()        standardizes one file's schema/timestamps
        --> combine_area_load_files()     merges all files, resolves overlaps
        --> fill_missing_hourly_records() completes the timeline, flags fills
        --> audit_area_load()             validates the combined dataset
        --> process_area_load()           orchestrates the steps, writes outputs
        --> main()                        exposes the pipeline as a CLI script
================================================================================
"""

# Imports 
from pathlib import Path
import argparse
import time

import numpy as np
import pandas as pd


# Path to project root
PROJECT_ROOT = Path("/Users/brodiehasein/alberta_power_markets_project")

# Paths to data folders
RAW_DATA_DIR = PROJECT_ROOT / "data/raw"
PREPROCESSING_DIR = PROJECT_ROOT / "data/preprocessing"
AUDIT_DIR = PROJECT_ROOT / "data/audits"

# Paths to outputs
OUTPUT_CSV = PREPROCESSING_DIR / "area_load_preprocessed.csv"
OUTPUT_PARQUET = PREPROCESSING_DIR / "area_load_preprocessed.parquet"

# Paths to audit/summary outputs
AUDIT_FILE = AUDIT_DIR / "area_load_audit_checks.csv"
SUMMARY_FILE = AUDIT_DIR / "area_load_feature_summary.csv"
FILE_SUMMARY_FILE = AUDIT_DIR / "area_load_file_summary.csv"


# a list literal of the six AESO planning-region names as they appear in raw files.
REGION_COLUMNS = [
    "CALGARY",
    "CENTRAL",
    "EDMONTON",
    "NORTHEAST",
    "NORTHWEST",
    "SOUTH",
]

# This dictionary connects each raw AESO region name to the standardized
# snake_case column name that will be used in the cleaned dataset.
REGION_CLEAN_MAP = {
    "CALGARY": "calgary_load_mw",
    "CENTRAL": "central_load_mw",
    "EDMONTON": "edmonton_load_mw",
    "NORTHEAST": "northeast_load_mw",
    "NORTHWEST": "northwest_load_mw",
    "SOUTH": "south_load_mw",
}


# a dict literal mapping each granular AESO "AREA" code (key) to its parent region (value).
AREA_TO_REGION = {
    "AREA6": "CALGARY",
    "AREA57": "CALGARY",

    "AREA4": "CENTRAL",
    "AREA13": "CENTRAL",
    "AREA28": "CENTRAL",
    "AREA29": "CENTRAL",
    "AREA30": "CENTRAL",
    "AREA32": "CENTRAL",
    "AREA34": "CENTRAL",
    "AREA35": "CENTRAL",
    "AREA36": "CENTRAL",
    "AREA37": "CENTRAL",
    "AREA38": "CENTRAL",
    "AREA39": "CENTRAL",
    "AREA42": "CENTRAL",
    "AREA43": "CENTRAL",
    "AREA44": "CENTRAL",
    "AREA45": "CENTRAL",
    "AREA46": "CENTRAL",
    "AREA47": "CENTRAL",
    "AREA48": "CENTRAL",
    "AREA49": "CENTRAL",
    "AREA52": "CENTRAL",
    "AREA53": "CENTRAL",
    "AREA54": "CENTRAL",
    "AREA55": "CENTRAL",
    "AREA56": "CENTRAL",

    "AREA31": "EDMONTON",
    "AREA40": "EDMONTON",
    "AREA60": "EDMONTON",

    "AREA25": "NORTHEAST",
    "AREA27": "NORTHEAST",
    "AREA33": "NORTHEAST",

    "AREA17": "NORTHWEST",
    "AREA18": "NORTHWEST",
    "AREA19": "NORTHWEST",
    "AREA20": "NORTHWEST",
    "AREA21": "NORTHWEST",
    "AREA22": "NORTHWEST",
    "AREA23": "NORTHWEST",
    "AREA24": "NORTHWEST",
    "AREA26": "NORTHWEST",
}

# sorted(dict) returns a sorted list of a dict's keys.
EXPECTED_AREA_COLUMNS = sorted(AREA_TO_REGION)

# A dictionary comprehension creates one mapping for every expected AREA code.
# Each raw name becomes a standardized column name such as AREA6 -> area6_load_mw.
AREA_CLEAN_MAP = {
    area: f"{area.lower()}_load_mw"
    for area in EXPECTED_AREA_COLUMNS
}

# This list comprehension looks up each standardized AREA column name while
# preserving the ordering established by EXPECTED_AREA_COLUMNS.
AREA_COLUMNS_CLEAN = [
    AREA_CLEAN_MAP[area]
    for area in EXPECTED_AREA_COLUMNS
]

REGION_COLUMNS_CLEAN = [
    REGION_CLEAN_MAP[region]
    for region in REGION_COLUMNS
]

# The starred expressions unpack all AREA and region column names into this set.
# A set is useful here because the audit only needs to test which names are
# present, rather than preserve their order.
EXPECTED_OUTPUT_COLUMNS = {
    "timestamp_utc",
    *AREA_COLUMNS_CLEAN,
    *REGION_COLUMNS_CLEAN,
    "total_area_load_mw",
    "total_region_load_mw",
    "area_load_imputed",
}

# The two dictionary comprehensions assign an acceptable numeric range to
# every AREA and region column. The ** operators unpack those generated
# mappings into one dictionary alongside the ranges for totals and flags.
RANGE_EXPECTATIONS = {
    **{
        column: (-5000, 20000)
        for column in AREA_COLUMNS_CLEAN
    },
    **{
        column: (-5000, 40000)
        for column in REGION_COLUMNS_CLEAN
    },
    "total_area_load_mw": (-5000, 50000),
    "total_region_load_mw": (-5000, 50000),
    "area_load_imputed": (0, 1),
}


def normalize_header_value(value) -> str:
    """
    General purpose:
        Normalize a single raw header cell into a comparable string: strip
        whitespace, remove byte-order-mark characters, collapse newlines,
        and uppercase everything.

    Role in the pipeline:
        Used by read_excel_area_load() and discover_raw_files() to compare
        inconsistent Excel header cells without being tripped up by case,
        stray whitespace, or hidden characters.
    """
    # The input is intentionally left unrestricted because spreadsheet header
    # cells may contain strings, numbers, or other values. The -> str annotation
    # documents that the function always returns a normalized string.
    return (
        str(value)
        .strip()
        .replace("\ufeff", "")
        .replace("\n", " ")
        .upper()
    )


def read_excel_area_load(
    path: Path,
    nrows: int | None = None,
) -> pd.DataFrame:
    """
    General purpose:
        Search every sheet and the first several rows of an Excel workbook
        to locate the row that actually contains the AESO area-load header,
        supporting either the newer (DT_MST) or older (DATE + HOUR ENDING)
        timestamp layout.

    Role in the pipeline:
        Called by read_raw_file() whenever the file being read is an Excel
        workbook. It hands back a properly-headered DataFrame so the rest
        of the pipeline doesn't need to know Excel's sheet/row quirks.
    """
    # nrows may be an integer limiting how many data rows are loaded, or None
    # to read the complete table. This lets the same function support both
    # lightweight file discovery and full preprocessing.
    workbook = pd.ExcelFile(path)
    required_regions = set(REGION_COLUMNS)

    for sheet_name in workbook.sheet_names:
        preview = pd.read_excel(
            path,
            sheet_name=sheet_name,
            header=None,
            nrows=40,
        )

        for row_number in range(len(preview)):
            row_values = {
                normalize_header_value(value)
                for value in preview.iloc[row_number].dropna()
            }

            has_regions = required_regions.issubset(row_values)
            has_new_timestamp = "DT_MST" in row_values
            has_old_timestamp = {
                "DATE",
                "HOUR ENDING",
            }.issubset(row_values)

            if has_regions and (
                has_new_timestamp
                or has_old_timestamp
            ):
                return pd.read_excel(
                    path,
                    sheet_name=sheet_name,
                    header=row_number,
                    nrows=nrows,
                )

    raise ValueError(
        f"Could not locate the AESO area-load table in {path.name}"
    )


def read_raw_file(
    path: Path,
    nrows: int | None = None,
) -> pd.DataFrame:
    """
    General purpose:
        Read one raw area-load file, regardless of whether it's an Excel
        workbook, CSV, TSV, or plain text file, trying multiple delimiter
        and encoding combinations until one works.

    Role in the pipeline:
        This gives both discover_raw_files() and combine_area_load_files()
        a single, uniform way to read any supported raw file, hiding all
        the format/encoding guesswork behind one function call.
    """
    suffix = path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        return read_excel_area_load(
            path,
            nrows=nrows,
        )

    # Each dictionary contains one delimiter-and-encoding combination to try.
    # Keeping the options in a list allows the same read operation to be
    # repeated systematically until the file is parsed into multiple columns.
    attempts = [
        {"sep": ",", "encoding": "utf-8-sig"},
        {"sep": "\t", "encoding": "utf-8-sig"},
        {"sep": ",", "encoding": "utf-16"},
        {"sep": "\t", "encoding": "utf-16"},
    ]

    errors = []

    for options in attempts:
        try:
            # **options passes the dictionary entries to pd.read_csv as named
            # arguments, equivalent to writing sep=... and encoding=... directly.
            frame = pd.read_csv(
                path,
                nrows=nrows,
                low_memory=False,
                **options,
            )

            if len(frame.columns) > 1:
                return frame

        except Exception as exc:
            errors.append(
                f"{options}: {repr(exc)}"
            )

    raise ValueError(
        f"Could not read raw area-load file {path}. "
        f"Attempts: {errors}"
    )


def discover_raw_files(
    raw_dir: Path = RAW_DATA_DIR,
) -> list[Path]:
    """
    General purpose:
        Scan the raw data directory for every CSV/TXT/TSV/XLSX/XLS file,
        peek at each one's header row, and keep only the files whose
        columns actually match the expected area-load schema.

    Role in the pipeline:
        This is the discovery stage — it supplies combine_area_load_files()
        with a complete, verified list of usable source files, so the
        pipeline doesn't depend on a hardcoded list of filenames.
    """
    # The return annotation list[Path] documents that the result will be a
    # collection of filesystem Path objects rather than plain filename strings.
    candidates = sorted(
        [
            *raw_dir.rglob("*.csv"),
            *raw_dir.rglob("*.txt"),
            *raw_dir.rglob("*.tsv"),
            *raw_dir.rglob("*.xlsx"),
            *raw_dir.rglob("*.xls"),
        ]
    )

    matched = []
    required_regions = set(REGION_COLUMNS)

    for path in candidates:
        try:
            header = read_raw_file(
                path,
                nrows=5,
            )

            normalized_columns = {
                normalize_header_value(column)
                for column in header.columns
            }

            has_regions = required_regions.issubset(
                normalized_columns
            )

            has_timestamp = (
                "DT_MST" in normalized_columns
                or {
                    "DATE",
                    "HOUR ENDING",
                }.issubset(normalized_columns)
            )

            if has_regions and has_timestamp:
                matched.append(path)

        except Exception:
            continue

    if not matched:
        candidate_names = "\n".join(
            f"  - {path}"
            for path in candidates
        )

        raise FileNotFoundError(
            "No AESO area-load files were identified under "
            f"{raw_dir}.\n"
            "Files inspected:\n"
            f"{candidate_names or '  None'}"
        )

    return matched

def standardize_raw_columns(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    General purpose:
        Normalize a raw DataFrame's column labels: strip whitespace, remove
        byte-order marks, collapse repeated whitespace, and uppercase.

    Role in the pipeline:
        Used by clean_area_load_file() so that every downstream schema
        check (e.g. "does this have a DT_MST column?") works regardless of
        superficial formatting differences between raw source files.
    """
    raw = raw.copy()

    raw.columns = (
        pd.Index(raw.columns)
        .astype(str)
        .str.strip()
        .str.replace(
            "\ufeff",
            "",
            regex=False,
        )
        # The regular expression r"\s+" matches any run of one or more
        # whitespace characters. Replacing each run with one space makes
        # headers consistent even when they contain tabs or repeated spaces.
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
        .str.upper()
    )

    return raw


def parse_dt_mst(
    values: pd.Series,
) -> pd.Series:
    """
    General purpose:
        Parse a Series of newer-format DT_MST timestamp strings and convert
        them from Alberta local time (fixed MST, no daylight saving) to UTC.

    Role in the pipeline:
        Used by clean_area_load_file() whenever a raw file uses the DT_MST
        timestamp column, so its timestamps end up on the same UTC basis as
        every other source file in the combined dataset.
    """
    parsed = pd.to_datetime(
        values,
        errors="coerce",
    )

    # tz_localize assigns the fixed MST timezone to timestamps that currently
    # have no timezone information. tz_convert then represents those same
    # moments in UTC without changing the actual points in time.
    return (
        parsed
        .dt.tz_localize("Etc/GMT+7")
        .dt.tz_convert("UTC")
    )


def parse_date_hour_ending(
    date_values: pd.Series,
    hour_ending_values: pd.Series,
) -> pd.Series:
    """
    General purpose:
        Parse the older-format DATE + HOUR ENDING (1-24) fields into UTC
        timestamps, converting "hour ending" (which labels the end of an
        hour) into an hour-beginning timestamp before localizing.

    Role in the pipeline:
        Used by clean_area_load_file() whenever a raw file uses the legacy
        DATE + HOUR ENDING schema, so older records align on the same
        hour-beginning UTC basis as newer DT_MST records.
    """
    dates = pd.to_datetime(
        date_values,
        errors="coerce",
    )

    hour_ending = pd.to_numeric(
        hour_ending_values,
        errors="coerce",
    )

    # between(1, 24) creates a boolean mask identifying valid hour-ending
    # values. Combining it with dates.notna() ensures both timestamp parts
    # are usable before the code attempts to construct a complete timestamp.
    valid = (
        dates.notna()
        & hour_ending.between(1, 24)
    )

    # Hour-ending values label the end of each hourly interval, so subtracting
    # one converts them to hour-beginning offsets: HE 1 becomes 00:00,
    # HE 2 becomes 01:00, and HE 24 becomes 23:00.
    local_hour_beginning = (
        dates
        + pd.to_timedelta(
            hour_ending - 1,
            unit="h",
        )
    ).where(valid)

    return (
        local_hour_beginning
        .dt.tz_localize("Etc/GMT+7")
        .dt.tz_convert("UTC")
    )


def numeric_series(
    values: pd.Series,
) -> pd.Series:
    """
    General purpose:
        Convert a Series of raw text values into numbers, first stripping
        thousands-separator commas and surrounding whitespace.

    Role in the pipeline:
        Used throughout clean_area_load_file() to convert every raw load
        column into numeric form, turning any unparseable cell into a
        missing value (NaN) that the audit stage can later detect.
    """
    return pd.to_numeric(
        values
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def clean_area_load_file(
    raw: pd.DataFrame,
    source_file: Path,
) -> pd.DataFrame:
    """
    General purpose:
        Standardize a single raw area-load file end-to-end: validate it has
        a usable schema, parse its timestamps into UTC, convert its area and
        region load columns to numbers, compute totals, and attach source
        provenance columns.

    Role in the pipeline:
        This is the per-file cleaning stage, called once for every file
        that combine_area_load_files() processes. Its consistent output
        schema is what allows many different raw files to be concatenated
        together safely.
    """
    raw = standardize_raw_columns(raw)

    has_dt_mst = "DT_MST" in raw.columns

    has_date_hour_ending = {
        "DATE",
        "HOUR ENDING",
    }.issubset(raw.columns)

    if not (
        has_dt_mst
        or has_date_hour_ending
    ):
        raise ValueError(
            f"{source_file.name} contains neither DT_MST nor "
            "DATE + HOUR ENDING."
        )

    missing_regions = (
        set(REGION_COLUMNS)
        - set(raw.columns)
    )

    if missing_regions:
        raise ValueError(
            f"{source_file.name} is missing regional columns: "
            f"{sorted(missing_regions)}"
        )

    available_area_columns = sorted(
        column
        for column in raw.columns
        if column.startswith("AREA")
    )

    if not available_area_columns:
        raise ValueError(
            f"{source_file.name} contains no AREA columns."
        )

    out = pd.DataFrame()

    if has_dt_mst:
        out["timestamp_utc"] = parse_dt_mst(
            raw["DT_MST"]
        )
        timestamp_schema = "DT_MST"

    else:
        out["timestamp_utc"] = parse_date_hour_ending(
            raw["DATE"],
            raw["HOUR ENDING"],
        )
        timestamp_schema = "DATE_PLUS_HOUR_ENDING"

    for raw_column in available_area_columns:
        clean_column = (
            f"{raw_column.lower()}_load_mw"
        )

        out[clean_column] = numeric_series(
            raw[raw_column]
        )

    for (
        raw_column,
        clean_column,
    ) in REGION_CLEAN_MAP.items():
        out[clean_column] = numeric_series(
            raw[raw_column]
        )

    observed_area_columns = [
        column
        for column in out.columns
        if (
            column.startswith("area")
            and column.endswith("_load_mw")
        )
    ]

    # axis=1 tells pandas to sum horizontally across the selected columns
    # for each row. min_count=1 preserves NaN only when every selected AREA
    # value in that row is missing, rather than incorrectly returning zero.
    out["total_area_load_mw"] = (
        out[observed_area_columns]
        .sum(
            axis=1,
            min_count=1,
        )
    )

    out["total_region_load_mw"] = (
        out[REGION_COLUMNS_CLEAN]
        .sum(
            axis=1,
            min_count=1,
        )
    )

    out["source_file"] = source_file.name
    out["timestamp_schema"] = timestamp_schema

    out = (
        out
        .dropna(
            subset=["timestamp_utc"]
        )
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )

    return out

def combine_area_load_files(
    files: list[Path],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    General purpose:
        Read and clean every discovered raw file, concatenate them into one
        table, detect and reject any files that disagree on the same
        timestamp (true conflicts), and otherwise keep the most recent
        duplicate observation for each hour.

    Role in the pipeline:
        This is the combination stage. It takes the list of files found by
        discover_raw_files() and produces the single unified (but not yet
        gap-filled) dataset that fill_missing_hourly_records() will refine.
    """
    # The return annotation documents that this function produces exactly
    # two DataFrames: the combined hourly dataset and a per-file summary table.
    cleaned_frames = []
    file_summaries = []

    for path in files:
        raw = read_raw_file(path)

        clean = clean_area_load_file(
            raw,
            path,
        )

        for column in AREA_COLUMNS_CLEAN:
            if column not in clean.columns:
                clean[column] = np.nan

        file_summaries.append({
            "source_file": path.name,
            "source_path": str(path),
            "timestamp_schema": clean[
                "timestamp_schema"
            ].iloc[0],
            "raw_rows": len(raw),
            "clean_rows": len(clean),
            "start_utc": str(
                clean["timestamp_utc"].min()
            ),
            "end_utc": str(
                clean["timestamp_utc"].max()
            ),
            "duplicate_timestamps_within_file": int(
                clean["timestamp_utc"]
                .duplicated()
                .sum()
            ),
            "area_columns_present": ";".join(
                sorted(
                    column
                    for column in clean.columns
                    if (
                        column.startswith("area")
                        and column.endswith("_load_mw")
                    )
                )
            ),
        })

        cleaned_frames.append(
            clean[
                [
                    "timestamp_utc",
                    *AREA_COLUMNS_CLEAN,
                    *REGION_COLUMNS_CLEAN,
                    "total_area_load_mw",
                    "total_region_load_mw",
                    "source_file",
                    "timestamp_schema",
                ]
            ]
        )

    combined = pd.concat(
        cleaned_frames,
        ignore_index=True,
    )

    # keep=False marks every row involved in a duplicated timestamp, including
    # the first occurrence. This makes it possible to compare all competing
    # records rather than seeing only the later duplicates.
    duplicate_rows = combined.loc[
        combined["timestamp_utc"].duplicated(
            keep=False
        )
    ].copy()

    if not duplicate_rows.empty:
        compare_columns = [
            *AREA_COLUMNS_CLEAN,
            *REGION_COLUMNS_CLEAN,
            "total_area_load_mw",
            "total_region_load_mw",
        ]

        conflicting_timestamps = []

        for (
            timestamp,
            group,
        ) in duplicate_rows.groupby(
            "timestamp_utc"
        ):
            # nunique counts the distinct values in each comparison column.
            # Including missing values means NaN versus a recorded value is
            # treated as a disagreement. Taking max() asks whether any column
            # contains more than one distinct value within this timestamp group.
            if (
                group[compare_columns]
                .nunique(
                    dropna=False
                )
                .max()
                > 1
            ):
                conflicting_timestamps.append(
                    str(timestamp)
                )

                if len(
                    conflicting_timestamps
                ) >= 20:
                    break

        if conflicting_timestamps:
            raise ValueError(
                "Conflicting duplicate timestamps across raw files: "
                f"{conflicting_timestamps}"
            )

    combined = (
        combined
        .drop_duplicates(
            subset=["timestamp_utc"],
            keep="last",
        )
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )

    final_columns = [
        "timestamp_utc",
        *AREA_COLUMNS_CLEAN,
        *REGION_COLUMNS_CLEAN,
        "total_area_load_mw",
        "total_region_load_mw",
    ]

    return (
        combined[final_columns],
        pd.DataFrame(file_summaries),
    )


def fill_missing_hourly_records(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    General purpose:
        Reindex the combined dataset onto a complete, gap-free hourly UTC
        timeline, linearly interpolate isolated single-hour gaps, flag which
        rows were inserted/interpolated, and recompute the total columns.

    Role in the pipeline:
        This is the gap-filling stage, applied after combine_area_load_files()
        produces a merged but potentially incomplete timeline. Its output
        feeds directly into audit_area_load() as the final candidate dataset.
    """
    out = df.copy()

    out["timestamp_utc"] = pd.to_datetime(
        out["timestamp_utc"],
        utc=True,
    )

    out = (
        out
        .sort_values("timestamp_utc")
        .drop_duplicates(
            subset=["timestamp_utc"],
            keep="last",
        )
        .set_index("timestamp_utc")
    )

    # Build a complete hourly UTC index from the first observed timestamp to
    # the last. Any hour absent from the original data will appear in this
    # index and can then be inserted through reindexing.
    complete_index = pd.date_range(
        start=out.index.min(),
        end=out.index.max(),
        freq="h",
        tz="UTC",
    )

    original_index = out.index

    # Reindexing aligns the DataFrame to the complete timeline. Existing rows
    # retain their values, while previously missing timestamps are inserted
    # as new rows containing NaN values.
    out = out.reindex(
        complete_index
    )

    # isin identifies timestamps that existed before reindexing. The ~ operator
    # reverses the boolean result, so newly inserted hours become True, and
    # astype("int8") stores those flags efficiently as 1 and 0.
    out["area_load_imputed"] = (
        ~out.index.isin(original_index)
    ).astype("int8")

    load_columns = [
        column
        for column in out.columns
        if column.endswith("_load_mw")
    ]

    # Time-based interpolation estimates missing values according to their
    # position between surrounding timestamps. limit=1 permits only isolated
    # one-hour gaps, while limit_area="inside" prevents filling missing values
    # before the first observation or after the last one.
    out[load_columns] = (
        out[load_columns]
        .interpolate(
            method="time",
            limit=1,
            limit_area="inside",
        )
    )

    unresolved = (
        out[load_columns]
        .isna()
        .any(axis=1)
    )

    if unresolved.any():
        unresolved_times = (
            out.index[unresolved]
            .astype(str)
            .tolist()[:20]
        )

        raise ValueError(
            "Unresolved missing area-load values after interpolation: "
            f"{unresolved_times}"
        )

    out["total_area_load_mw"] = (
        out[AREA_COLUMNS_CLEAN]
        .sum(
            axis=1,
            min_count=1,
        )
    )

    out["total_region_load_mw"] = (
        out[REGION_COLUMNS_CLEAN]
        .sum(
            axis=1,
            min_count=1,
        )
    )

    out.index.name = "timestamp_utc"

    return out.reset_index()


def load_and_clean_area_load(
    raw_dir: Path = RAW_DATA_DIR,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    list[Path],
]:
    """
    General purpose:
        Run the full discovery-through-gap-filling sequence in one call:
        find raw files, clean and combine them, then fill timeline gaps.

    Role in the pipeline:
        This is a convenience orchestrator used by process_area_load(). It
        packages three separate stages into a single function call so the
        top-level workflow doesn't need to sequence them itself.
    """
    # The return annotation shows the order and type of all three outputs:
    # the cleaned dataset, the per-file summary DataFrame, and the list of
    # source-file Path objects that were discovered and processed.
    files = discover_raw_files(
        raw_dir
    )

    clean, file_summary = (
        combine_area_load_files(
            files
        )
    )

    clean = fill_missing_hourly_records(
        clean
    )

    return (
        clean,
        file_summary,
        files,
    )

def audit_area_load(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    bool,
]:
    """
    General purpose:
        Run a full suite of validation checks against the cleaned, gap-filled
        area-load data: timeline integrity, schema, missingness, value
        ranges, imputation counts, and whether area-level and region-level
        totals are mutually consistent.

    Role in the pipeline:
        This is the quality-control gate between cleaning and saving. Its
        boolean pass/fail result determines whether process_area_load() is
        allowed to write the cleaned data out as an approved product.
    """
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
            Append one standardized audit result (as a dict) to the shared
            `rows` list, so every check is recorded in the same shape.

        Role in the pipeline:
            This is a local helper used only within audit_area_load(). It
            keeps every individual check consistent so they can all be
            converted into a single audit_df table at the end.
        """
        # These optional parameters provide sensible defaults for an audit
        # result. A caller only needs to supply them when a check has a known
        # expectation, a non-error severity, or additional explanatory notes.
        rows.append({
            "check": check,
            "pass": bool(passed),
            "severity": severity,
            "observed": observed,
            "expected": expected,
            "notes": notes,
        })

    add(
        "timestamp_column_exists",
        "timestamp_utc" in df.columns,
        "timestamp_utc" in df.columns,
        True,
    )

    if "timestamp_utc" not in df.columns:
        audit_df = pd.DataFrame(rows)
        return (
            audit_df,
            pd.DataFrame(),
            False,
        )

    ts = pd.DatetimeIndex(
        pd.to_datetime(
            df["timestamp_utc"],
            utc=True,
        )
    )

    feature_cols = [
        column
        for column in df.columns
        if column != "timestamp_utc"
    ]

    observed_start = ts.min()
    observed_end = ts.max()

    expected_index = pd.date_range(
        observed_start,
        observed_end,
        freq="h",
        tz="UTC",
    )

    missing_hours = (
        expected_index
        .difference(ts)
    )

    extra_hours = (
        ts.difference(
            expected_index
        )
    )

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
        severity="info",
    )

    add(
        "observed_period_end",
        True,
        str(observed_end),
        "recorded",
        severity="info",
    )

    add(
        "expected_hours_from_start_to_end",
        len(df) == len(expected_index),
        len(df),
        len(expected_index),
    )

    add(
        "timestamps_monotonic",
        ts.is_monotonic_increasing,
        ts.is_monotonic_increasing,
        True,
    )

    add(
        "duplicate_timestamps",
        not ts.has_duplicates,
        int(
            ts.duplicated().sum()
        ),
        0,
    )

    if len(ts) > 1:
        diffs = (
            pd.Series(ts)
            .diff()
            .dropna()
        )

        bad_diffs = diffs[
            diffs
            != pd.Timedelta(hours=1)
        ]

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

    if "area_load_imputed" in df.columns:
        imputed_count = int(
            df["area_load_imputed"].sum()
        )

        invalid_flags = int(
            (
                ~df["area_load_imputed"]
                .isin([0, 1])
            ).sum()
        )

        add(
            "imputed_flag_binary",
            invalid_flags == 0,
            invalid_flags,
            0,
        )

        add(
            "imputed_hour_count",
            imputed_count == 10,
            imputed_count,
            10,
            severity="warning",
            notes=(
                "Ten isolated hourly rows were inserted and linearly "
                "interpolated, one in each year from 2011 through 2020."
            ),
        )

    actual_columns = set(
        df.columns
    )

    missing_columns = (
        EXPECTED_OUTPUT_COLUMNS
        - actual_columns
    )

    extra_columns = (
        actual_columns
        - EXPECTED_OUTPUT_COLUMNS
    )

    add(
        "expected_columns_present",
        len(missing_columns) == 0,
        "; ".join(
            sorted(missing_columns)
        ),
        "no missing expected columns",
    )

    add(
        "no_unexpected_extra_columns",
        len(extra_columns) == 0,
        "; ".join(
            sorted(extra_columns)
        ),
        "no extra columns",
        severity="warning",
    )

    non_numeric = [
        column
        for column in feature_cols
        if not pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]

    add(
        "all_features_numeric",
        len(non_numeric) == 0,
        "; ".join(non_numeric),
        "all numeric",
    )

    all_null = [
        column
        for column in feature_cols
        if df[column].isna().all()
    ]

    any_null = [
        column
        for column in feature_cols
        if df[column].isna().any()
    ]

    add(
        "no_all_null_feature_columns",
        len(all_null) == 0,
        "; ".join(all_null),
        "",
    )

    add(
        "no_partial_null_feature_columns",
        len(any_null) == 0,
        "; ".join(any_null),
        "",
        severity="warning",
        notes=(
            "Some planning-area values may be unavailable in "
            "parts of the historical series."
        ),
    )

    for (
        column,
        (
            low,
            high,
        ),
    ) in RANGE_EXPECTATIONS.items():
        if column not in df.columns:
            continue

        column_min = df[column].min(
            skipna=True
        )

        column_max = df[column].max(
            skipna=True
        )

        add(
            f"range_check__{column}",
            (
                column_min >= low
                and column_max <= high
            ),
            observed=(
                f"min={column_min:.6g}, "
                f"max={column_max:.6g}"
            ),
            expected=f"[{low}, {high}]",
            severity="warning",
        )

    for column in feature_cols:
        add(
            f"domain_negative_count__{column}",
            True,
            observed=int(
                (df[column] < 0).sum()
            ),
            expected="recorded",
            severity="info",
        )

        add(
            f"domain_zero_count__{column}",
            True,
            observed=int(
                (df[column] == 0).sum()
            ),
            expected="recorded",
            severity="info",
        )

        add(
            f"domain_missing_count__{column}",
            True,
            observed=int(
                df[column].isna().sum()
            ),
            expected="recorded",
            severity="info",
        )

    implied_total_area = (
        df[AREA_COLUMNS_CLEAN]
        .sum(
            axis=1,
            min_count=1,
        )
    )

    area_diff = (
        df["total_area_load_mw"]
        - implied_total_area
    ).abs()

    add(
        "domain_total_area_matches_sum",
        area_diff.max(
            skipna=True
        ) < 1e-6,
        observed=(
            f"max_abs_diff="
            f"{area_diff.max(skipna=True):.6g}"
        ),
        expected="0",
    )

    implied_total_region = (
        df[REGION_COLUMNS_CLEAN]
        .sum(
            axis=1,
            min_count=1,
        )
    )

    region_diff = (
        df["total_region_load_mw"]
        - implied_total_region
    ).abs()

    add(
        "domain_total_region_matches_sum",
        region_diff.max(
            skipna=True
        ) < 1e-6,
        observed=(
            f"max_abs_diff="
            f"{region_diff.max(skipna=True):.6g}"
        ),
        expected="0",
    )

    total_difference = (
        df["total_area_load_mw"]
        - df["total_region_load_mw"]
    ).abs()

    add(
        "domain_area_total_matches_region_total",
        total_difference.max(
            skipna=True
        ) < 1.0,
        observed=(
            f"max_abs_diff="
            f"{total_difference.max(skipna=True):.6g}"
        ),
        expected="< 1 MW",
        severity="warning",
        notes=(
            "Differences may reflect AESO rounding or historical "
            "planning-area aggregation changes."
        ),
    )

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

    summary = summary.rename(
        columns={
            "index": "feature",
            "1%": "p01",
            "50%": "median",
            "99%": "p99",
        }
    )

    summary["missing_count"] = (
        df[feature_cols]
        .isna()
        .sum()
        .values
    )

    summary["missing_pct"] = (
        summary["missing_count"]
        / len(df)
        * 100
    )

    summary["dtype"] = [
        str(df[column].dtype)
        for column in feature_cols
    ]

    audit_df = pd.DataFrame(rows)

    error_checks = audit_df.loc[
        audit_df["severity"].eq("error"),
        "pass",
    ]

    audit_pass = (
        bool(error_checks.all())
        if not error_checks.empty
        else True
    )

    return (
        audit_df,
        summary,
        audit_pass,
    )


def print_audit_report(
    audit_df: pd.DataFrame,
    feature_summary: pd.DataFrame,
    clean: pd.DataFrame,
    files: list[Path],
) -> None:
    """
    General purpose:
        Print a formatted, human-readable summary of the audit results,
        feature statistics, and the list of raw files that were combined.

    Role in the pipeline:
        This is the presentation stage. It does not alter any data or the
        pass/fail decision — it only surfaces what audit_area_load() found
        so a person can quickly review the health of the cleaned dataset.
    """
    # The -> None annotation documents that this function does not return a
    # value. Its purpose is to produce terminal output through print calls.
    error_checks = audit_df.loc[
        audit_df["severity"].eq("error"),
        "pass",
    ]

    audit_pass = (
        bool(error_checks.all())
        if not error_checks.empty
        else True
    )

    failed = audit_df.loc[
        ~audit_df["pass"]
    ]

    ts = pd.DatetimeIndex(
        pd.to_datetime(
            clean["timestamp_utc"],
            utc=True,
        )
    )

    expected_hours = len(
        pd.date_range(
            ts.min(),
            ts.max(),
            freq="h",
            tz="UTC",
        )
    )

    imputed_hours = (
        int(
            clean["area_load_imputed"].sum()
        )
        if "area_load_imputed" in clean.columns
        else 0
    )

    print("\n" + "=" * 80)
    print("AREA LOAD AUDIT")
    print("=" * 80)
    print(f"Overall pass  : {audit_pass}")
    print(f"Raw files     : {len(files)}")
    print(f"Rows          : {len(clean):,}")
    print(f"Features      : {len(clean.columns) - 1}")
    print(f"Start UTC     : {ts.min()}")
    print(f"End UTC       : {ts.max()}")
    print(f"Observed hrs  : {len(clean):,}")
    print(f"Expected hrs  : {expected_hours:,}")
    print(
        f"Coverage      : "
        f"{len(clean) / expected_hours:.2%}"
    )
    print(f"Imputed hrs   : {imputed_hours:,}")

    print("\nRaw files:")
    for path in files:
        print(f"  - {path}")

    print("\nFailed checks:")
    if failed.empty:
        print("  None")
    else:
        for _, row in failed.iterrows():
            print(
                f"  - {row['check']} "
                f"[{row['severity']}] "
                f"observed={row['observed']} "
                f"expected={row['expected']}"
            )

    print("\nFeature statistics:")

    compact = feature_summary[
        [
            "feature",
            "missing_count",
            "mean",
            "median",
            "p01",
            "p99",
            "min",
            "max",
        ]
    ].copy()

    compact = compact.rename(
        columns={
            "missing_count": "missing",
        }
    )

    for column in [
        "mean",
        "median",
        "p01",
        "p99",
        "min",
        "max",
    ]:
        compact[column] = (
            compact[column]
            .round(2)
        )

    print(
        compact.to_string(
            index=False
        )
    )

    print("=" * 80)


def process_area_load(
    overwrite: bool = False,
) -> dict:
    """
    General purpose:
        Run the full area-load pipeline end-to-end: discover raw files,
        clean and combine them, fill timeline gaps, audit the result, and
        conditionally save the cleaned data and audit evidence.

    Role in the pipeline:
        This is the top-level orchestrator that main() calls. It decides
        whether to skip work (if outputs already exist), and only writes
        the final cleaned files if the audit's error-level checks pass.
    """
    # overwrite defaults to False, so existing outputs are normally preserved.
    # The function returns a dictionary containing the pipeline's status,
    # output paths, summary metrics, or error information.
    start_time = time.perf_counter()

    if (
        OUTPUT_CSV.exists()
        and OUTPUT_PARQUET.exists()
        and not overwrite
    ):
        return {
            "dataset": "area_load",
            "status": "skipped_existing",
            "pass": True,
            "csv_file": str(OUTPUT_CSV),
            "parquet_file": str(
                OUTPUT_PARQUET
            ),
        }

    try:
        (
            clean,
            file_summary,
            files,
        ) = load_and_clean_area_load()

        (
            audit_df,
            summary_df,
            audit_pass,
        ) = audit_area_load(clean)

        print_audit_report(
            audit_df,
            summary_df,
            clean,
            files,
        )

        AUDIT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        audit_df.to_csv(
            AUDIT_FILE,
            index=False,
        )

        summary_df.to_csv(
            SUMMARY_FILE,
            index=False,
        )

        file_summary.to_csv(
            FILE_SUMMARY_FILE,
            index=False,
        )

        if not audit_pass:
            return {
                "dataset": "area_load",
                "status": "audit_failed",
                "pass": False,
                "raw_files": len(files),
                "audit_file": str(
                    AUDIT_FILE
                ),
                "summary_file": str(
                    SUMMARY_FILE
                ),
                "file_summary": str(
                    FILE_SUMMARY_FILE
                ),
                "processing_seconds": round(
                    time.perf_counter()
                    - start_time,
                    3,
                ),
            }

        PREPROCESSING_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        clean.to_csv(
            OUTPUT_CSV,
            index=False,
        )

        clean.to_parquet(
            OUTPUT_PARQUET,
            index=False,
        )

        return {
            "dataset": "area_load",
            "status": "saved",
            "pass": True,
            "rows": len(clean),
            "features": (
                len(clean.columns) - 1
            ),
            "start": str(
                clean["timestamp_utc"].min()
            ),
            "end": str(
                clean["timestamp_utc"].max()
            ),
            "raw_files": len(files),
            "imputed_hours": int(
                clean["area_load_imputed"].sum()
            ),
            "csv_file": str(
                OUTPUT_CSV
            ),
            "parquet_file": str(
                OUTPUT_PARQUET
            ),
            "audit_file": str(
                AUDIT_FILE
            ),
            "summary_file": str(
                SUMMARY_FILE
            ),
            "file_summary": str(
                FILE_SUMMARY_FILE
            ),
            "processing_seconds": round(
                time.perf_counter()
                - start_time,
                3,
            ),
        }

    except Exception as exc:
        return {
            "dataset": "area_load",
            "status": "error",
            "pass": False,
            "error": repr(exc),
            "processing_seconds": round(
                time.perf_counter()
                - start_time,
                3,
            ),
        }


def main() -> None:
    """
    General purpose:
        Provide a command-line entry point for running this file's
        preprocessing pipeline directly from a terminal.

    Role in the pipeline:
        This is the outermost wrapper. It parses the --overwrite flag,
        calls process_area_load(), and prints the resulting status
        dictionary so a user running this script sees what happened.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess AESO hourly load "
            "by planning area and region."
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
    print("AREA LOAD PREPROCESSING RESULT")
    print("=" * 80)

    for key, value in result.items():
        print(f"{key}: {value}")

    print("=" * 80)


# Python assigns the special variable __name__ the value "__main__" when
# this file is executed directly. When another file imports it, __name__
# instead contains the module name, so main() is not run automatically.
if __name__ == "__main__":
    main()