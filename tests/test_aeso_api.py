"""Regression tests for AESO API chunking and raw persistence contracts."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aeso_api.catalog import (
    ACTUAL_FORECAST,
    ENERGY_MERIT_ORDER,
    GENERATION_CAPACITY,
    REFERENCE_ENDPOINTS,
    UNIT_COMMITMENT,
)
from aeso_api.client import (
    AESOClient,
    AESORequestError,
    read_project_env_value,
    validate_aeso_payload,
)
from aeso_api.download import (
    download_snapshot,
    iter_date_chunks,
    validate_endpoint_dates,
)
from preprocessing.aeso_api_preprocessing import (
    normalize_actual_forecast,
    normalize_smp,
)


class AESODownloadTests(unittest.TestCase):
    def test_aeso_envelope_requires_success_and_return_value(self) -> None:
        validate_aeso_payload({"responseCode": 200, "return": {}})
        with self.assertRaisesRegex(AESORequestError, "non-success"):
            validate_aeso_payload({"responseCode": 400, "return": {}})
        with self.assertRaisesRegex(AESORequestError, "expected 'return'"):
            validate_aeso_payload({"responseCode": 200})

    def test_project_env_parser_accepts_spaces_and_quotes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text('AESO_API_KEY = "secret"\n', encoding="utf-8")
            with patch("aeso_api.client.PROJECT_ROOT", Path(directory)):
                self.assertEqual(read_project_env_value("AESO_API_KEY"), "secret")

    def test_chunks_are_inclusive_and_respect_limit(self) -> None:
        chunks = list(
            iter_date_chunks(
                date(2024, 1, 1),
                date(2024, 12, 31),
                31,
            )
        )
        self.assertEqual(chunks[0], (date(2024, 1, 1), date(2024, 1, 31)))
        self.assertEqual(chunks[-1][1], date(2024, 12, 31))
        self.assertTrue(
            all((end - start).days + 1 <= 31 for start, end in chunks)
        )

    def test_generation_endpoint_records_exclusive_end_behavior(self) -> None:
        self.assertFalse(GENERATION_CAPACITY.end_date_inclusive)
        self.assertEqual(GENERATION_CAPACITY.maximum_days, 30)

    def test_documented_earliest_date_is_enforced(self) -> None:
        with self.assertRaisesRegex(ValueError, "begins on"):
            validate_endpoint_dates(
                ACTUAL_FORECAST,
                date(1999, 12, 31),
                date(2000, 1, 1),
            )

    def test_delayed_endpoint_rejects_recent_dates(self) -> None:
        with self.assertRaisesRegex(ValueError, "delayed 60 days"):
            validate_endpoint_dates(
                ENERGY_MERIT_ORDER,
                date.today(),
                date.today(),
            )

    def test_unit_commitment_has_no_undocumented_disclosure_delay(self) -> None:
        self.assertEqual(UNIT_COMMITMENT.delayed_days, 0)
        validate_endpoint_dates(
            UNIT_COMMITMENT,
            date(2024, 7, 1),
            date.today(),
        )

    def test_reference_catalog_includes_both_requested_snapshots(self) -> None:
        self.assertEqual(
            set(REFERENCE_ENDPOINTS),
            {"asset-list", "pool-participants"},
        )

    def test_snapshot_reuses_valid_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot_dir = Path(directory) / "asset_list"
            snapshot_dir.mkdir()
            snapshot = snapshot_dir / "snapshot_20260101T000000Z.json"
            snapshot.write_text('{"assets": []}', encoding="utf-8")

            with patch("aeso_api.download.AESOClient") as client:
                result = download_snapshot(
                    "/assetlist-api/v1/assetlist",
                    "asset_list",
                    output_root=Path(directory),
                )

            self.assertEqual(result.status, "already_exists")
            self.assertEqual(result.path, snapshot)
            client.assert_not_called()

    def test_download_writes_raw_and_metadata_without_key(self) -> None:
        body = json.dumps({"rows": [{"value": 1}]}).encode("utf-8")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "response.json"
            client = AESOClient(api_key="secret")
            with patch.object(
                client,
                "_request",
                return_value=(body, "application/json"),
            ):
                result = client.download_json(
                    "/example",
                    {"startDate": "2025-01-01"},
                    output,
                )

            self.assertEqual(result.status, "downloaded")
            self.assertEqual(json.loads(output.read_text()), {"rows": [{"value": 1}]})
            metadata = output.with_suffix(".metadata.json").read_text()
            self.assertNotIn("secret", metadata)
            self.assertIn("sha256", metadata)

    def test_actual_forecast_normalization_calculates_signed_error(self) -> None:
        raw = pd.DataFrame(
            {
                "begin_datetime_utc": ["2025-01-01 07:00"],
                "alberta_internal_load": ["10,100"],
                "forecast_alberta_internal_load": ["10,000"],
            }
        )
        with patch(
            "preprocessing.aeso_api_preprocessing._records",
            return_value=raw,
        ):
            result = normalize_actual_forecast()
        self.assertEqual(result.loc[0, "load_forecast_error_mw"], 100)

    def test_smp_hourly_average_is_duration_weighted(self) -> None:
        raw = pd.DataFrame(
            {
                "begin_datetime_utc": [
                    "2025-01-01 07:00",
                    "2025-01-01 07:15",
                ],
                "end_datetime_utc": [
                    "2025-01-01 07:15",
                    "2025-01-01 08:00",
                ],
                "system_marginal_price": ["100", "20"],
                "volume": ["", "50"],
            }
        )
        with patch(
            "preprocessing.aeso_api_preprocessing._records",
            return_value=raw,
        ):
            _, hourly = normalize_smp()
        self.assertEqual(hourly.loc[0, "smp_observed_minutes"], 60)
        self.assertAlmostEqual(
            hourly.loc[0, "smp_time_weighted_mean_cad_mwh"],
            40,
        )


if __name__ == "__main__":
    unittest.main()
