from pathlib import Path

PROJECT_ROOT = Path('/Users/brodiehasein/alberta_power_markets_project')

# Data Directories
DATA_DIR = PROJECT_ROOT / 'data'

RAW_DIR = DATA_DIR / 'raw'
PREPROCESSING_DIR = DATA_DIR / 'preprocessing'
AUDITS_DIR = DATA_DIR / 'audits'
FEATURES_DIR = DATA_DIR / 'features'

# Raw Data
WEATHER_RAW_DIR = RAW_DIR / 'weather'
ERA5_RAW_DIR = WEATHER_RAW_DIR / 'era5'
ERA5_SINGLE_LEVEL_DIR = ERA5_RAW_DIR / "single_levels"
ERA5_PRESSURE_LEVEL_DIR = ERA5_RAW_DIR / "pressure_levels"


# Preprocessed Data CSVs
AREA_LOAD_CSV = PREPROCESSING_DIR / "area_load_preprocessed.csv"
PA_TABLE_CSV = PREPROCESSING_DIR / "pa_hourly_preprocessed.csv"
INTERTIES_HOUR_AHEAD_CSV = PREPROCESSING_DIR / "interties_hour_ahead_preprocessed.csv"
INTERTIE_CAPABILITY_CSV = PREPROCESSING_DIR / "intertie_capability_preprocessed.csv"
GENERATION_CSV = PREPROCESSING_DIR / "generation_by_fuel_preprocessed.csv"
OUTAGES_CSV = PREPROCESSING_DIR / "outages_preprocessed_preprocessed.csv"
WIND_PROJECTS_CSV = PREPROCESSING_DIR / "wind_projects_preprocessed.csv"
SOLAR_PROJECTS_CSV = PREPROCESSING_DIR / "solar_projects_preprocessed.csv"

# Preprocessed Data Parquets
AREA_LOAD_PARQUET = PREPROCESSING_DIR / "area_load_preprocessed.parquet"
PA_TABLE_PARQUET = PREPROCESSING_DIR / "pa_hourly_preprocessed.parquet"
INTERTIES_HOUR_AHEAD_PARQUET = PREPROCESSING_DIR / "interties_hour_ahead_preprocessed.parquet"
INTERTIE_CAPABILITY_PARQUET = PREPROCESSING_DIR / "intertie_capability_preprocessed.parquet"
GENERATION_PARQUET = PREPROCESSING_DIR / "generation_by_fuel_preprocessed.parquet"
OUTAGES_PARQUET = PREPROCESSING_DIR / "outages_preprocessed.parquet"

# Weather Data ncs
WEATHER_PREPROCESSING_DIR = PREPROCESSING_DIR / "weather"
ERA5_PREPROCESSING_DIR = WEATHER_PREPROCESSING_DIR / "era5"

ERA5_MONTHLY_STANDARDIZED_DIR = (ERA5_PREPROCESSING_DIR / "monthly_standardized")

# Notebooks
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Random Seed
# Random seed is useful for sampling, train/test splits, random simulations, bootstrapping
RANDOM_SEED = 42