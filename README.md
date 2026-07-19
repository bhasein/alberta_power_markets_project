# Alberta Power Markets Project

Hourly data pipeline and feature-engineering workspace for Alberta electricity-market analysis.

The project standardizes raw AESO market data and ERA5 weather data into UTC-indexed hourly datasets, audits each preprocessing step, and builds model-ready feature tables for downstream price, load, outage, renewable, and intertie analysis.

## Project Goal

The core goal is to build a clean, defensible Alberta power-market dataset where every major source is:

- represented at hourly frequency
- indexed by `timestamp_utc`
- saved in a predictable preprocessing location
- audited before being used downstream
- ready to merge into feature tables for modeling and analysis

This structure keeps raw data, cleaned preprocessing outputs, audit artifacts, and engineered features separate.

## Repository Layout

```text
.
├── data/
│   ├── raw/                 # Original source files from AESO, ERA5, and project databases
│   ├── preprocessing/       # Clean hourly source-level datasets
│   ├── features/            # Engineered feature datasets
│   └── audits/              # Audit reports and feature summaries
├── notebooks/               # Exploratory notebooks
├── src/
│   ├── preprocessing/       # Source-specific raw-to-clean pipelines
│   ├── feature_engineering/ # Feature builders using preprocessed data
│   ├── era5/                # ERA5 download and preprocessing scripts
│   └── config.py            # Central path configuration
└── README.md
```

## Data Pipeline

The intended flow is:

```text
raw data
  -> source-specific preprocessing
  -> audit checks
  -> cleaned hourly preprocessing tables
  -> feature engineering
  -> model-ready feature tables
```

Preprocessing scripts generally follow this pattern:

```text
load raw file
  -> clean schema and timestamps
  -> convert values to numeric columns
  -> audit completeness, ranges, and domain logic
  -> write CSV/parquet outputs
  -> write audit reports
```

The canonical merge key is `timestamp_utc`.

## Main Preprocessing Scripts

Run these from the project root.

```bash
python3 src/preprocessing/pa_preprocessing.py --overwrite
python3 src/preprocessing/area_load_preprocessing.py --overwrite
python3 src/preprocessing/generation_preprocessing.py --overwrite
python3 src/preprocessing/outages_preprocessing.py --overwrite
python3 src/preprocessing/intertie_capability_preprocessing.py --overwrite
python3 src/preprocessing/interties_hour_ahead_preprocessing.py --overwrite
```

Current source-level outputs include:

- `data/preprocessing/pa_hourly_preprocessed.csv`
- `data/preprocessing/area_load_preprocessed.csv`
- `data/preprocessing/generation_by_fuel_preprocessed.csv`
- `data/preprocessing/outages_preprocessed.csv`
- `data/preprocessing/intertie_capability_preprocessed.csv`
- `data/preprocessing/interties_hour_ahead_preprocessed.csv`
- corresponding `.parquet` files where supported

Audit outputs are written to `data/audits/`.

## ERA5 Weather Pipeline

ERA5 scripts live in `src/era5/`.

```bash
python3 src/era5/era5_downloader.py
python3 src/era5/era5_preprocessing.py --overwrite
```

ERA5 raw files are expected under:

```text
data/raw/weather/era5/
```

Standardized monthly ERA5 outputs are written under:

```text
data/preprocessing/weather/era5/
```

## Feature Engineering

Feature builders live in `src/feature_engineering/`.

```bash
python3 src/feature_engineering/calendar_features.py --overwrite
python3 src/feature_engineering/market_features.py --overwrite
python3 src/feature_engineering/renewable_weather_features.py
python3 src/feature_engineering/load_weather_features.py
```

Current feature outputs include:

- `data/features/calendar/calendar_features_hourly.parquet`
- `data/features/market/market_features_hourly.parquet`
- `data/features/weather/renewable_weather_features_hourly.parquet`
- `data/features/weather/load_weather_features_hourly.parquet`

## Configuration

Project paths are centralized in:

```text
src/config.py
```

At the moment, `PROJECT_ROOT` is set as an absolute path:

```python
PROJECT_ROOT = Path('/Users/brodiehasein/alberta_power_markets_project')
```

If the project is moved to another machine or folder, update this value before running scripts.

## Python Environment

This repository does not currently include a dependency lockfile or requirements file. The code imports these main packages:

- `pandas`
- `numpy`
- `xarray`
- `cdsapi`
- `holidays`
- parquet support such as `pyarrow` or `fastparquet`
- Excel support such as `openpyxl`

A typical setup is:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy xarray cdsapi holidays pyarrow openpyxl netcdf4
```

## Notes

- Raw source files should remain unchanged in `data/raw/`.
- Preprocessing outputs should be reproducible from raw files.
- Audit files in `data/audits/` are part of the workflow, not throwaway logs.
- Use `timestamp_utc` as the primary key when merging hourly datasets.
- Keep source-specific preprocessing separate from feature engineering so pipeline failures are easier to isolate.
