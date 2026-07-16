from pathlib import Path

PROJECT_ROOT = Path('/Users/brodiehasein/alberta_power_markets_project')

# Data Directories
DATA_DIR = PROJECT_ROOT / 'data'

RAW_DIR = DATA_DIR / 'raw'
PREPROCESSING_DIR = DATA_DIR / 'preprocessing'
PROCESSED_DIR = DATA_DIR / 'processed'
DICTIONARIES_DIR = DATA_DIR / 'dictionaries'

# Raw Data
WEATHER_RAW_DIR = RAW_DIR / 'weather'
ERA5_RAW_DIR = WEATHER_RAW_DIR / 'era5'
ERA5_SINGLE_LEVEL_DIR = ERA5_RAW_DIR / "single_levels"
ERA5_PRESSURE_LEVEL_DIR = ERA5_RAW_DIR / "pressure_levels"

# Preprocessed Data
WEATHER_PREPROCESSING_DIR = PREPROCESSING_DIR / "weather"
PA_TABLE = PREPROCESSING_DIR / "P&A_Table_preprocessed.csv"
INTERTIES = PREPROCESSING_DIR / "interties_hour_ahead.csv"
INTERTIE_CAPABILITY = PREPROCESSING_DIR / "intertie_capability_preprocessed.csv"
GENERATION = PREPROCESSING_DIR / "generation_by_fuel_preprocessed.csv"
OUTAGES = PREPROCESSING_DIR / "outages_preprocessed.csv"
WIND_PROJECTS = PREPROCESSING_DIR / "wind_projects_preprocessed.csv"
SOLAR_PROJECTS = PREPROCESSING_DIR / "solar_projects_preprocessed.csv"
ERA5_PREPROCESSING = WEATHER_PREPROCESSING_DIR

# Notebooks
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Random Seed
RANDOM_SEED = 42