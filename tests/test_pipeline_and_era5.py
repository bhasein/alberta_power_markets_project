"""Regression tests for orchestration and ERA5 acquisition contracts."""

from __future__ import annotations

import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import run_pipeline
from era5 import era5_download_progress, era5_downloader


class PipelineDependencyTests(unittest.TestCase):
    """Verify that selective runs preserve the declared dependency graph."""

    def test_master_expands_to_every_required_stage(self) -> None:
        self.assertEqual(
            run_pipeline.expand_dependencies({"master"}),
            {name for name, _ in run_pipeline.STAGES},
        )

    def test_only_runs_prerequisites_in_pipeline_order(self) -> None:
        calls: list[str] = []

        def stage(name: str):
            def execute(**_: object) -> dict[str, object]:
                calls.append(name)
                return {"stage": name, "status": "saved", "pass": True}

            return execute

        names = ["pa", "outages", "interties_hour_ahead", "market_features"]
        stages = [(name, stage(name)) for name in names]
        with (
            patch.object(run_pipeline, "STAGES", stages),
            redirect_stdout(StringIO()),
        ):
            results = run_pipeline.run_pipeline(only={"market_features"})

        self.assertEqual(calls, names)
        self.assertTrue(all(result["pass"] for result in results))

    def test_empty_selection_is_rejected(self) -> None:
        all_names = {name for name, _ in run_pipeline.STAGES}
        with self.assertRaisesRegex(ValueError, "No pipeline stages"):
            run_pipeline.run_pipeline(skip=all_names)


class Era5DownloadTests(unittest.TestCase):
    """Verify content validation and batch failure propagation."""

    def write_month(self, path: Path, year: int, month: int) -> None:
        dataset = xr.Dataset(
            coords={
                "valid_time": era5_downloader.expected_timestamps(
                    year,
                    month,
                ).tz_localize(None)
            }
        )
        dataset.to_netcdf(path, engine="netcdf4")

    def test_valid_nc_rejects_a_different_month_with_same_hour_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "month.nc"
            self.write_month(path, 2024, 4)
            self.assertTrue(era5_downloader.valid_nc(path, 2024, 4))
            self.assertFalse(era5_downloader.valid_nc(path, 2024, 6))

    def test_month_attempts_every_pressure_request_and_collects_failures(self) -> None:
        with (
            patch.object(
                era5_downloader,
                "download_single_levels",
                side_effect=RuntimeError("single"),
            ),
            patch.object(
                era5_downloader,
                "download_pressure_file",
                side_effect=RuntimeError("pressure"),
            ) as pressure_download,
        ):
            with redirect_stdout(StringIO()):
                failures = era5_downloader.download_month(Mock(), 2024, 1)

        self.assertEqual(
            len(failures),
            1 + len(era5_downloader.PRESSURE_REQUESTS),
        )
        self.assertEqual(
            pressure_download.call_count,
            len(era5_downloader.PRESSURE_REQUESTS),
        )

    def test_batch_raises_when_a_month_has_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(era5_downloader, "SINGLE_DIR", root / "single"),
                patch.object(era5_downloader, "PRESSURE_DIR", root / "pressure"),
                patch.object(era5_downloader.cdsapi, "Client", return_value=Mock()),
                patch.object(
                    era5_downloader,
                    "download_month",
                    return_value=["2024-01 failed"],
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "1 failures"):
                    era5_downloader.download_range(2024, 2024, 1)

    def test_progress_requires_valid_files_not_just_present_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            era5_root = root / "data" / "raw" / "weather" / "era5"
            single = era5_root / "single_levels" / "era5_single_levels_alberta_2024_01"
            pressure = era5_root / "pressure_levels"
            single.mkdir(parents=True)
            pressure.mkdir(parents=True)
            for filename in era5_downloader.SINGLE_LEVEL_FILENAMES:
                self.write_month(single / filename, 2024, 2)
            for request in era5_downloader.PRESSURE_REQUESTS:
                path = pressure / era5_downloader.pressure_output_path(
                    2024,
                    1,
                    request,
                ).name
                self.write_month(path, 2024, 2)

            audit = era5_download_progress.audit_era5_downloads(
                root,
                2024,
                2024,
                1,
            )

        self.assertEqual(audit.loc[0, "single_files_present"], 3)
        self.assertEqual(audit.loc[0, "pressure_files_present"], 3)
        self.assertFalse(audit.loc[0, "month_complete"])


if __name__ == "__main__":
    unittest.main()
