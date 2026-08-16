"""Regression tests for shared preprocessing contracts."""

from __future__ import annotations

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

import config
from preprocessing import area_load_preprocessing
from preprocessing import era5_preprocessing
from preprocessing import generation_preprocessing
from preprocessing import intertie_capability_preprocessing
from preprocessing import interties_hour_ahead_preprocessing
from preprocessing import master_preprocessing
from preprocessing import outages_preprocessing
from preprocessing import pa_preprocessing
from preprocessing.shared import (
    DuplicateConflictError,
    build_manifest,
    deduplicate_or_raise,
    duplicate_failure_audit,
    outputs_are_current,
    preprocessing_code_paths,
    write_manifests,
)


class PathContractTests(unittest.TestCase):
    def test_config_is_portable(self) -> None:
        self.assertEqual(config.PROJECT_ROOT, PROJECT_ROOT)

    def test_preprocessing_outputs_use_config_contract(self) -> None:
        expected = {
            area_load_preprocessing.OUTPUT_CSV: config.AREA_LOAD_CSV,
            area_load_preprocessing.OUTPUT_PARQUET: config.AREA_LOAD_PARQUET,
            pa_preprocessing.OUTPUT_CSV: config.PA_TABLE_CSV,
            pa_preprocessing.OUTPUT_PARQUET: config.PA_TABLE_PARQUET,
            interties_hour_ahead_preprocessing.OUTPUT_CSV:
                config.INTERTIES_HOUR_AHEAD_CSV,
            interties_hour_ahead_preprocessing.OUTPUT_PARQUET:
                config.INTERTIES_HOUR_AHEAD_PARQUET,
            intertie_capability_preprocessing.OUTPUT_CSV:
                config.INTERTIE_CAPABILITY_CSV,
            intertie_capability_preprocessing.OUTPUT_PARQUET:
                config.INTERTIE_CAPABILITY_PARQUET,
            generation_preprocessing.OUTPUT_CSV: config.GENERATION_CSV,
            generation_preprocessing.OUTPUT_PARQUET: config.GENERATION_PARQUET,
            outages_preprocessing.OUTPUT_CSV: config.OUTAGES_CSV,
            outages_preprocessing.OUTPUT_PARQUET: config.OUTAGES_PARQUET,
            master_preprocessing.OUTPUT_CSV: config.MASTER_CSV,
            master_preprocessing.OUTPUT_PARQUET: config.MASTER_PARQUET,
        }
        for observed, configured in expected.items():
            self.assertEqual(observed, configured)

    def test_master_consumes_engineered_generation_features(self) -> None:
        self.assertEqual(
            master_preprocessing.SOURCE_FILES["generation"],
            config.GENERATION_FEATURES,
        )


class DuplicateContractTests(unittest.TestCase):
    def test_exact_duplicates_are_collapsed(self) -> None:
        frame = pd.DataFrame(
            {"timestamp_utc": ["2024-01-01", "2024-01-01"], "value": [1, 1]}
        )
        result, count = deduplicate_or_raise(
            frame,
            ["timestamp_utc"],
            dataset_name="test",
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(count, 1)

    def test_conflicting_duplicates_are_rejected(self) -> None:
        frame = pd.DataFrame(
            {"timestamp_utc": ["2024-01-01", "2024-01-01"], "value": [1, 2]}
        )
        with self.assertRaises(DuplicateConflictError) as context:
            deduplicate_or_raise(
                frame,
                ["timestamp_utc"],
                dataset_name="test",
            )
        self.assertEqual(context.exception.conflicting_keys, 1)
        audit = duplicate_failure_audit(context.exception)
        conflict = audit.loc[
            audit["check"].eq("conflicting_duplicate_source_keys")
        ].iloc[0]
        self.assertFalse(conflict["pass"])
        self.assertEqual(conflict["observed"], 1)

    def test_duplicate_statistics_are_recorded_in_normal_audit(self) -> None:
        raw = pd.DataFrame(
            {
                "Date (MST)": ["2024-01-01", "2024-01-01"],
                "AIL": [10_000, 10_000],
                "Gas Price": [2.0, 2.0],
                "Price": [50.0, 50.0],
                "Spark Spread": [30.0, 30.0],
            }
        )
        clean = pa_preprocessing.clean_pa(raw)
        audit, _, passed = pa_preprocessing.audit_pa(clean)
        collapsed = audit.loc[
            audit["check"].eq("exact_duplicate_source_rows_collapsed")
        ].iloc[0]
        conflicts = audit.loc[
            audit["check"].eq("conflicting_duplicate_source_keys")
        ].iloc[0]
        self.assertTrue(passed)
        self.assertEqual(collapsed["observed"], 1)
        self.assertEqual(conflicts["observed"], 0)


class TransformationContractTests(unittest.TestCase):
    def test_pa_uses_fixed_mst(self) -> None:
        raw = pd.DataFrame(
            {
                "Date (MST)": ["2024-01-01 00:00:00"],
                "AIL": [10_000],
                "Gas Price": [2.0],
                "Price": [50.0],
                "Spark Spread": [30.0],
            }
        )
        result = pa_preprocessing.clean_pa(raw)
        self.assertEqual(
            result.loc[0, "timestamp_utc"],
            pd.Timestamp("2024-01-01 07:00:00", tz="UTC"),
        )

    def test_area_load_extension_remains_explicitly_frozen(self) -> None:
        timestamp = pd.Timestamp("2024-12-31 07:00:00", tz="UTC")
        values = {
            column: float(index + 1)
            for index, column in enumerate(
                [
                    *area_load_preprocessing.AREA_COLUMNS_CLEAN,
                    *area_load_preprocessing.REGION_COLUMNS_CLEAN,
                ]
            )
        }
        frame = pd.DataFrame(
            [{"timestamp_utc": timestamp, "area_load_imputed": 0, **values}]
        )
        result = area_load_preprocessing.extend_with_frozen_distribution(
            frame,
            extend_to_utc=timestamp + pd.Timedelta(hours=2),
        )
        self.assertEqual(result["area_load_frozen"].tolist(), [0, 1, 1])
        self.assertEqual(result["area_load_imputed"].tolist(), [0, 1, 1])
        for column in values:
            self.assertEqual(result[column].nunique(), 1)

    def test_generation_reshape_and_totals(self) -> None:
        rows = []
        for fuel, generation in [("Coal", 100.0), ("Wind", 25.0)]:
            rows.append(
                {
                    "Date - MST": "01/01/2024 12:00:00 AM",
                    "Fuel Type": fuel,
                    "System Generation": generation,
                    "Total Generation": generation + 1,
                    "System Available": generation + 2,
                    "System Capacity": generation + 3,
                    "Maximum Capacity": generation + 4,
                }
            )
        result = generation_preprocessing.clean_generation(pd.DataFrame(rows))
        self.assertEqual(result.loc[0, "coal_system_generation"], 100.0)
        self.assertEqual(result.loc[0, "wind_system_generation"], 25.0)
        self.assertEqual(result.loc[0, "total_system_generation"], 125.0)
        self.assertEqual(result.loc[0, "total_total_generation"], 127.0)

    def test_signed_interties_move_negative_flow_to_opposite_direction(self) -> None:
        frame = pd.DataFrame(
            {
                "import_bc": [0.0],
                "import_mt": [-14.0],
                "import_sk": [5.0],
                "export_bc": [0.0],
                "export_mt": [0.0],
                "export_sk": [-3.0],
            }
        )
        result = interties_hour_ahead_preprocessing.clean_signed_intertie_flows(
            frame
        )
        self.assertEqual(result.loc[0, "import_mt"], 0.0)
        self.assertEqual(result.loc[0, "export_mt"], 14.0)
        self.assertEqual(result.loc[0, "import_sk"], 8.0)
        self.assertEqual(result.loc[0, "export_sk"], 0.0)
        self.assertEqual(result.loc[0, "import_mt_raw"], -14.0)

    def test_area_load_rejects_conflicting_file_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            base = {
                "DT_MST": ["2024-01-01 00:00:00"],
                "AREA6": [100.0],
                **{
                    region: [100.0]
                    for region in area_load_preprocessing.REGION_COLUMNS
                },
            }
            first = root / "first.csv"
            second = root / "second.csv"
            pd.DataFrame(base).to_csv(first, index=False)
            changed = dict(base)
            changed["AREA6"] = [101.0]
            pd.DataFrame(changed).to_csv(second, index=False)
            with self.assertRaises(DuplicateConflictError):
                area_load_preprocessing.combine_area_load_files([first, second])


class Era5ContractTests(unittest.TestCase):
    def test_grid_and_time_audits_accept_canonical_month(self) -> None:
        timestamps = pd.date_range("2024-02-01", periods=696, freq="h")
        latitudes = np.arange(48.5, 60.0 + 0.25, 0.25)
        longitudes = np.arange(-120.5, -109.0 + 0.25, 0.25)
        dataset = xr.Dataset(
            coords={
                "timestamp": timestamps,
                "lat": latitudes,
                "lon": longitudes,
            }
        )
        grid = era5_preprocessing.audit_grid(dataset, 2024, 2)
        timeline = era5_preprocessing.audit_time(dataset, 2024, 2)
        self.assertTrue(grid.loc[grid["severity"].eq("error"), "pass"].all())
        self.assertTrue(timeline.loc[timeline["severity"].eq("error"), "pass"].all())


class MasterContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.timestamps = pd.date_range(
            "2024-01-01", periods=2, freq="h", tz="UTC"
        )

    def test_identical_overlap_is_reconciled(self) -> None:
        datasets = {
            "calendar": pd.DataFrame(
                {"timestamp_utc": self.timestamps, "shared": [1.0, 2.0]}
            ),
            "incoming": pd.DataFrame(
                {
                    "timestamp_utc": self.timestamps,
                    "shared": [1.0, 2.0],
                    "new": [3.0, 4.0],
                }
            ),
        }
        master, reconciled, _ = master_preprocessing.merge_master_sources(
            datasets
        )
        self.assertEqual(master.columns.tolist(), ["timestamp_utc", "shared", "new"])
        self.assertEqual(reconciled.loc[0, "conflicting_rows"], 0)

    def test_conflicting_overlap_is_rejected(self) -> None:
        datasets = {
            "calendar": pd.DataFrame(
                {"timestamp_utc": self.timestamps, "shared": [1.0, 2.0]}
            ),
            "incoming": pd.DataFrame(
                {"timestamp_utc": self.timestamps, "shared": [1.0, 9.0]}
            ),
        }
        with self.assertRaisesRegex(ValueError, "conflicts"):
            master_preprocessing.merge_master_sources(datasets)


class ProvenanceContractTests(unittest.TestCase):
    def test_preprocessing_manifest_hashes_central_configuration(self) -> None:
        code_paths = preprocessing_code_paths(
            Path(pa_preprocessing.__file__)
        )
        self.assertIn(Path(config.__file__).resolve(), code_paths)

    def test_source_change_invalidates_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.csv"
            code = root / "pipeline.py"
            output = root / "output.parquet"
            source.write_text("one\n", encoding="utf-8")
            code.write_text("pass\n", encoding="utf-8")
            output.write_text("data\n", encoding="utf-8")

            manifest = build_manifest("test", [source], [code])
            write_manifests([output], manifest)
            self.assertTrue(outputs_are_current([output], manifest))

            source.write_text("two\n", encoding="utf-8")
            refreshed = build_manifest("test", [source], [code])
            self.assertFalse(outputs_are_current([output], refreshed))

    def test_era5_quick_audits_do_not_replace_full_audits(self) -> None:
        self.assertNotEqual(
            era5_preprocessing.AUDIT_FILE,
            era5_preprocessing.QUICK_AUDIT_FILE,
        )
        self.assertNotEqual(
            era5_preprocessing.FEATURE_SUMMARY_FILE,
            era5_preprocessing.QUICK_FEATURE_SUMMARY_FILE,
        )

    def test_missing_or_changed_audit_invalidates_output_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.csv"
            code = root / "pipeline.py"
            output = root / "output.parquet"
            audit = root / "audit.csv"
            source.write_text("source\n", encoding="utf-8")
            code.write_text("pass\n", encoding="utf-8")
            output.write_text("data\n", encoding="utf-8")
            audit.write_text("audit\n", encoding="utf-8")
            manifest = build_manifest("test", [source], [code])
            write_manifests([output, audit], manifest)
            self.assertTrue(outputs_are_current([output, audit], manifest))
            audit.write_text("changed audit\n", encoding="utf-8")
            self.assertFalse(outputs_are_current([output, audit], manifest))


if __name__ == "__main__":
    unittest.main()
