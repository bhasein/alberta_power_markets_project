"""Reusable date-range and snapshot download orchestration."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from aeso_api.catalog import DatedEndpoint
from aeso_api.client import (
    AESOClient,
    AESORequestError,
    DownloadResult,
    validate_aeso_payload,
)
from config import AESO_API_RAW_DIR


def parse_date(value: str) -> date:
    """Parse an ISO calendar date for command-line arguments."""

    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Invalid date {value!r}; expected YYYY-MM-DD."
        ) from error


def iter_date_chunks(start: date, end: date, maximum_days: int):
    """Yield inclusive date chunks no longer than the API limit."""

    if end < start:
        raise ValueError("end_date must be on or after start_date")
    current = start
    while current <= end:
        chunk_end = min(end, current + timedelta(days=maximum_days - 1))
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


def validate_endpoint_dates(endpoint: DatedEndpoint, start: date, end: date) -> None:
    """Reject dates outside documented historical or disclosure bounds."""

    if endpoint.earliest_date and start < date.fromisoformat(endpoint.earliest_date):
        raise ValueError(
            f"{endpoint.name} begins on {endpoint.earliest_date}; "
            f"received {start.isoformat()}."
        )
    latest_disclosed = date.today() - timedelta(days=endpoint.delayed_days)
    if endpoint.delayed_days and end > latest_disclosed:
        raise ValueError(
            f"{endpoint.name} is delayed {endpoint.delayed_days} days. "
            f"Latest currently requestable date is {latest_disclosed.isoformat()}."
        )


def download_date_range(
    endpoint: DatedEndpoint,
    start: date,
    end: date,
    *,
    extra_parameters: dict[str, str] | None = None,
    output_root: Path = AESO_API_RAW_DIR,
    overwrite: bool = False,
) -> list[DownloadResult]:
    """Download one endpoint over an inclusive range using safe chunks."""

    validate_endpoint_dates(endpoint, start, end)
    client = AESOClient()
    results: list[DownloadResult] = []

    for chunk_start, chunk_end in iter_date_chunks(
        start,
        end,
        endpoint.maximum_days,
    ):
        params = {
            endpoint.start_parameter: chunk_start.strftime(endpoint.date_format),
        }
        if endpoint.end_parameter is not None:
            request_end = chunk_end
            if not endpoint.end_date_inclusive:
                request_end += timedelta(days=1)
            params[endpoint.end_parameter] = request_end.strftime(
                endpoint.date_format
            )
        if extra_parameters:
            params.update(extra_parameters)

        if endpoint.maximum_days == 1:
            filename = f"{chunk_start.isoformat()}.json"
        else:
            filename = (
                f"{chunk_start.isoformat()}_to_{chunk_end.isoformat()}.json"
            )
        output_path = output_root / endpoint.output_directory / filename
        result = client.download_json(
            endpoint.path,
            params,
            output_path,
            overwrite=overwrite,
        )
        print(
            f"{endpoint.output_directory}: {result.status}: "
            f"{result.path.name} ({result.bytes_written:,} bytes)"
        )
        results.append(result)

    return results


def download_snapshot(
    path: str,
    output_directory: str,
    *,
    parameters: dict[str, str] | None = None,
    output_root: Path = AESO_API_RAW_DIR,
    overwrite: bool = False,
) -> DownloadResult:
    """Download a latest-only endpoint, reusing an existing valid snapshot."""

    output_directory_path = output_root / output_directory
    existing_snapshots = sorted(
        output_directory_path.glob("snapshot_*.json"),
        reverse=True,
    )
    if existing_snapshots and not overwrite:
        existing = existing_snapshots[0]
        try:
            payload = json.loads(existing.read_text(encoding="utf-8"))
            validate_aeso_payload(payload)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AESORequestError(
                "Existing snapshot is invalid; inspect it or rerun with "
                f"--overwrite: {existing}"
            ) from error
        result = DownloadResult(
            existing,
            "already_exists",
            existing.stat().st_size,
        )
        print(
            f"{output_directory}: {result.status}: {result.path.name} "
            f"({result.bytes_written:,} bytes)"
        )
        return result

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_directory_path / f"snapshot_{timestamp}.json"
    result = AESOClient().download_json(
        path,
        parameters or {},
        output_path,
        overwrite=overwrite,
    )
    print(
        f"{output_directory}: {result.status}: {result.path.name} "
        f"({result.bytes_written:,} bytes)"
    )
    return result


def add_range_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared date-range and overwrite arguments."""

    parser.add_argument("--start-date", required=True, type=parse_date)
    parser.add_argument("--end-date", required=True, type=parse_date)
    parser.add_argument("--overwrite", action="store_true")
