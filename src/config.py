"""
================================================================================
Central path configuration for the Alberta power markets project.

All paths are built from PROJECT_ROOT so the project can be moved or cloned
without touching any other file. Import the constants you need rather than
reconstructing paths elsewhere.
================================================================================
"""

from pathlib import Path

# ============================================================================
# Project root
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# ============================================================================
# Top-level data directories
# ============================================================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PREPROCESSING_DIR = PROCESSED_DIR / "preprocessing"
FEATURES_DIR = PROCESSED_DIR / "feature_engineering"

AUDITS_DIR = DATA_DIR / "audits"
PREPROCESSING_AUDITS_DIR = AUDITS_DIR / "preprocessing"
FEATURE_ENGINEERING_AUDITS_DIR = AUDITS_DIR / "feature_engineering"

# ============================================================================
# Raw data
# ============================================================================

WEATHER_RAW_DIR = RAW_DIR / "weather"
ERA5_RAW_DIR = WEATHER_RAW_DIR / "era5"
ERA5_SINGLE_LEVEL_DIR = ERA5_RAW_DIR / "single_levels"
ERA5_PRESSURE_LEVEL_DIR = ERA5_RAW_DIR / "pressure_levels"

# ============================================================================
# Preprocessing: weather (ERA5)
# ============================================================================

WEATHER_PREPROCESSING_DIR = PREPROCESSING_DIR / "weather"
ERA5_PREPROCESSING_DIR = WEATHER_PREPROCESSING_DIR / "era5"
ERA5_MONTHLY_STANDARDIZED_DIR = ERA5_PREPROCESSING_DIR / "monthly_standardized"

# ============================================================================
# Preprocessing: market/system datasets (CSV)
# ============================================================================

AREA_LOAD_CSV = PREPROCESSING_DIR / "area_load_preprocessed.csv"
PA_TABLE_CSV = PREPROCESSING_DIR / "pa_hourly_preprocessed.csv"
INTERTIES_HOUR_AHEAD_CSV = PREPROCESSING_DIR / "interties_hour_ahead.csv"
INTERTIE_CAPABILITY_CSV = PREPROCESSING_DIR / "intertie_capability.csv"
GENERATION_CSV = PREPROCESSING_DIR / "generation_by_fuel.csv"
OUTAGES_CSV = PREPROCESSING_DIR / "outages_preprocessed.csv"
WIND_PROJECTS_CSV = PREPROCESSING_DIR / "wind_projects_preprocessed.csv"
SOLAR_PROJECTS_CSV = PREPROCESSING_DIR / "solar_projects_preprocessed.csv"

# ============================================================================
# Preprocessing: market/system datasets (Parquet)
# ============================================================================

AREA_LOAD_PARQUET = PREPROCESSING_DIR / "area_load_preprocessed.parquet"
PA_TABLE_PARQUET = PREPROCESSING_DIR / "pa_hourly_preprocessed.parquet"
INTERTIES_HOUR_AHEAD_PARQUET = PREPROCESSING_DIR / "interties_hour_ahead.parquet"
INTERTIE_CAPABILITY_PARQUET = PREPROCESSING_DIR / "intertie_capability.parquet"
GENERATION_PARQUET = PREPROCESSING_DIR / "generation_by_fuel.parquet"
OUTAGES_PARQUET = PREPROCESSING_DIR / "outages_preprocessed.parquet"

# ============================================================================
# Master analytical dataset
# ============================================================================

MASTER_PREPROCESSING_DIR = PROCESSED_DIR / "master"
MASTER_PARQUET = MASTER_PREPROCESSING_DIR / "master_hourly.parquet"
MASTER_CSV = MASTER_PREPROCESSING_DIR / "master_hourly.csv"

# ============================================================================
# Feature engineering outputs
# ============================================================================

CALENDAR_FEATURES = (
    FEATURES_DIR / "calendar" / "calendar_features_hourly.parquet"
)
LOAD_WEATHER_FEATURES = (
    FEATURES_DIR / "weather" / "load_weather_features_hourly.parquet"
)
RENEWABLE_WEATHER_FEATURES = (
    FEATURES_DIR / "weather" / "renewable_weather_features_hourly.parquet"
)
MARKET_FEATURES = FEATURES_DIR / "market" / "market_features_hourly.parquet"
GENERATION_FEATURES = (
    FEATURES_DIR / "generation" / "generation_features_hourly.parquet"
)

# ============================================================================
# Reproducibility
# ============================================================================

# Used for sampling, train/test splits, random simulations, and bootstrapping.
RANDOM_SEED = 42

# ============================================================================
# Canonical pipeline coverage
# ============================================================================

PIPELINE_START_YEAR = 2015
PIPELINE_END_YEAR = 2026
PIPELINE_END_MONTH = 6
PIPELINE_START_UTC = "2015-01-01 00:00:00+00:00"
PIPELINE_END_UTC = "2026-06-30 23:00:00+00:00"

# CDS API area order is north, west, south, east.
ERA5_ALBERTA_AREA = [60.0, -120.5, 48.5, -109.0]
