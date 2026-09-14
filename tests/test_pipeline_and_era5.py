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
from tigge import tigge_downloader


class PipelineDependencyTests(unittest.TestCase):
    """Verify that selective runs preserve the declared dependency graph."""

    def test_master_expands_to_every_required_stage(self) -> None:
        self.assertEqual(
            run_pipeline.expand_dependencies({"master"}),
            {name for name, _ in run_pipeline.STAGES},
        )

    def test_aeso_api_is_required_by_master(self) -> None:
        self.assertIn(
            "aeso_api",
            run_pipeline.expand_dependencies({"master"}),
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


class TiggeDownloadTests(unittest.TestCase):
    """Verify the TIGGE request, vintage, and resumability contracts."""

    def test_request_uses_alberta_domain_and_preserves_vintages(self) -> None:
        request = tigge_downloader.single_level_request(
            2024,
            2,
            tigge_downloader.SINGLE_LEVEL_REQUESTS[0],
        )

        self.assertEqual(request["area"], "60.0/-120.5/48.5/-109.0")
        self.assertEqual(request["origin"], "ecmf")
        self.assertEqual(request["type"], "cf")
        self.assertEqual(request["time"], "00:00:00/12:00:00")
        self.assertEqual(
            request["step"],
            "6/12/18/24/30/36/42/48/54/60/66/72",
        )
        self.assertEqual(request["date"], "2024-02-01/2024-02-29")
        self.assertEqual(request["param"].split("/")[:4], [
            "167", "168", "165", "166"
        ])

    def test_default_archive_begins_in_2020(self) -> None:
        args = tigge_downloader.build_argument_parser().parse_args([])
        self.assertEqual(args.start_year, 2020)

    def test_valid_download_requires_matching_request_metadata(self) -> None:
        request = tigge_downloader.base_request(2024, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.grib"
            path.write_bytes(b"GRIB" + b"forecast-data" + b"7777")
            tigge_downloader.write_metadata(path, request)

            self.assertTrue(tigge_downloader.valid_download(path, request))
            changed = {**request, "leadtime_hour": ["6"]}
            self.assertFalse(tigge_downloader.valid_download(path, changed))

    def test_dry_run_does_not_require_api_credentials(self) -> None:
        with (
            patch.object(tigge_downloader.cdsapi, "Client") as client,
            patch.object(tigge_downloader, "SINGLE_DIR", Path("single")),
            patch.object(tigge_downloader, "PRESSURE_DIR", Path("pressure")),
            redirect_stdout(StringIO()),
        ):
            tigge_downloader.download_range(2024, 1, 2024, 1, dry_run=True)

        client.assert_not_called()

    def test_surface_only_attempts_two_monthly_requests(self) -> None:
        with (
            patch.object(tigge_downloader, "ensure_download") as ensure,
            redirect_stdout(StringIO()),
        ):
            failures = tigge_downloader.download_month(
                None,
                2024,
                1,
                dry_run=True,
                include_surface=True,
                include_pressure=False,
            )

        self.assertEqual(failures, [])
        self.assertEqual(ensure.call_count, 2)

    def test_pressure_only_attempts_three_monthly_requests(self) -> None:
        with (
            patch.object(tigge_downloader, "ensure_download") as ensure,
            redirect_stdout(StringIO()),
        ):
            failures = tigge_downloader.download_month(
                None,
                2024,
                1,
                dry_run=True,
                include_surface=False,
                include_pressure=True,
            )

        self.assertEqual(failures, [])
        self.assertEqual(ensure.call_count, 3)


if __name__ == "__main__":
    unittest.main()
