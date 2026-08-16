"""Focused regression tests for shared feature-engineering contracts."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from feature_engineering import calendar_features
from feature_engineering import load_weather_features
from feature_engineering import renewable_weather_features
from feature_engineering.shared import (
    add_changes_through_current_hour,
    add_prior_rolling_statistics,
    build_manifest,
    classify_feature_timing,
    manifest_path,
    output_is_current,
    validate_continuous_hourly_frame,
    write_manifest,
)


class HourlyFeatureContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "timestamp_utc": pd.date_range(
                    "2024-01-01", periods=5, freq="h", tz="UTC"
                ),
                "value": [1.0, 2.0, 3.0, 4.0, 100.0],
            }
        )

    def test_prior_rolling_excludes_current_hour(self) -> None:
        result = add_prior_rolling_statistics(
            self.frame,
            "value",
            [3],
            minimum_observations=1,
        )
        self.assertEqual(result.loc[4, "value_mean_prior_3h"], 3.0)

    def test_changes_are_explicitly_current_minus_prior(self) -> None:
        result = add_changes_through_current_hour(self.frame, "value", [1])
        self.assertEqual(result.loc[4, "value_change_1h"], 96.0)

    def test_row_based_features_reject_hourly_gaps(self) -> None:
        irregular = self.frame.drop(index=2).reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, "uninterrupted hourly"):
            validate_continuous_hourly_frame(irregular)


class CalendarFeatureTests(unittest.TestCase):
    def test_non_leap_year_uses_365_day_cycle(self) -> None:
        calendar, _ = calendar_features.build_calendar_features(
            pd.Timestamp("2019-12-31 07:00", tz="UTC"),
            pd.Timestamp("2019-12-31 07:00", tz="UTC"),
        )
        expected = np.sin(2.0 * np.pi * 364.0 / 365.0)
        self.assertAlmostEqual(calendar.loc[0, "day_of_year_sin"], expected)


class RenewableProjectTimingTests(unittest.TestCase):
    def test_exact_dates_are_preserved_and_year_fallback_is_flagged(self) -> None:
        projects = pd.DataFrame(
            {
                "project_name": ["Example", "Example"],
                "capacity_mw": [10.0, 5.0],
                "commissioning_date": ["2022-10-15", None],
                "commissioning_year": [2022, 2024],
                "latitude": [51.0, 51.0],
                "longitude": [-114.0, -114.0],
            }
        )
        result = renewable_weather_features.normalize_project_columns(
            projects,
            "solar",
        )
        self.assertEqual(
            result.loc[0, "commissioning_date"],
            pd.Timestamp("2022-10-15", tz="UTC"),
        )
        self.assertEqual(result.loc[0, "commissioning_date_precision"], "exact_date")
        self.assertEqual(result.loc[1, "commissioning_date_precision"], "year_estimate")
        self.assertEqual(result["project_id"].nunique(), 2)
        self.assertTrue(result["project_id"].str.contains("__PHASE_").all())

    def test_capacity_activates_on_exact_commissioning_hour(self) -> None:
        projects = pd.DataFrame(
            {
                "commissioning_date": [pd.Timestamp("2024-06-01 12:00", tz="UTC")],
                "retirement_date": [pd.NaT],
                "capacity_mw": [10.0],
            }
        )
        timestamps = pd.date_range(
            "2024-06-01 11:00", periods=2, freq="h", tz="UTC"
        )
        capacity = renewable_weather_features.active_capacity_matrix(
            timestamps,
            projects,
        )
        np.testing.assert_array_equal(capacity[:, 0], [0.0, 10.0])


class WeatherSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = xr.Dataset(
            {
                "temperature_2m": (
                    ("timestamp", "lat", "lon"),
                    np.ones((1, 1, 1)),
                )
            },
            coords={
                "timestamp": pd.date_range("2024-01-01", periods=1),
                "lat": [51.0],
                "lon": [-114.0],
            },
        )

    def test_load_weather_rejects_partial_era5_schema(self) -> None:
        mapping = pd.DataFrame(
            {
                "weather_site_id": [0],
                "weather_lat": [51.0],
                "weather_lon": [-114.0],
            }
        )
        with self.assertRaisesRegex(ValueError, "missing required load-weather"):
            load_weather_features.extract_load_region_sites(self.dataset, mapping)

    def test_renewable_weather_rejects_partial_era5_schema(self) -> None:
        mapping = pd.DataFrame(
            {
                "weather_site_id": [0],
                "weather_lat": [51.0],
                "weather_lon": [-114.0],
            }
        )
        with self.assertRaisesRegex(ValueError, "missing required renewable-weather"):
            renewable_weather_features.extract_project_sites(self.dataset, mapping)

    def test_missing_temperature_does_not_become_non_extreme(self) -> None:
        frame = pd.DataFrame({"load_weighted_temperature_c": [np.nan, -31.0]})
        result = load_weather_features.add_extreme_temperature_features(frame)
        self.assertTrue(pd.isna(result.loc[0, "extreme_cold_below_minus_30c"]))
        self.assertEqual(result.loc[1, "extreme_cold_below_minus_30c"], 1)


class ProvenanceAndTimingTests(unittest.TestCase):
    def test_manifest_invalidates_changed_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            code = root / "code.py"
            output = root / "features.parquet"
            source.write_text("one", encoding="utf-8")
            code.write_text("x = 1\n", encoding="utf-8")
            output.write_text("placeholder", encoding="utf-8")
            manifest = build_manifest("test", [source], [code], {"mode": "test"})
            write_manifest(output, manifest)
            self.assertTrue(output_is_current(output, manifest))

            output.write_text("tampered", encoding="utf-8")
            self.assertFalse(output_is_current(output, manifest))
            output.write_text("placeholder", encoding="utf-8")
            write_manifest(output, manifest)

            source.write_text("changed", encoding="utf-8")
            changed = build_manifest("test", [source], [code], {"mode": "test"})
            self.assertFalse(output_is_current(output, changed))
            json.loads(manifest_path(output).read_text(encoding="utf-8"))

    def test_historical_columns_override_target_derived_prefixes(self) -> None:
        timing = classify_feature_timing(
            "pool_price_lag_24h",
            target_columns={"pool_price"},
            target_derived_prefixes={"pool_price"},
        )
        self.assertEqual(timing, "historical")


if __name__ == "__main__":
    unittest.main()
