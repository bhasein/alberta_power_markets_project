"""Unified command-line entry point for AESO API downloads."""

from __future__ import annotations

import argparse

from aeso_api.catalog import ENDPOINTS, REFERENCE_ENDPOINTS
from aeso_api.download import download_date_range, download_snapshot, parse_date


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    range_parser = subparsers.add_parser(
        "range",
        help="Download one date-based API in documented safe chunks.",
    )
    range_parser.add_argument("dataset", choices=sorted(ENDPOINTS))
    range_parser.add_argument("--start-date", required=True, type=parse_date)
    range_parser.add_argument("--end-date", required=True, type=parse_date)
    range_parser.add_argument("--overwrite", action="store_true")

    snapshot_parser = subparsers.add_parser(
        "snapshot",
        help="Download one latest-only reference dataset.",
    )
    snapshot_parser.add_argument(
        "dataset",
        choices=sorted(REFERENCE_ENDPOINTS),
    )
    snapshot_parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    if args.command == "snapshot":
        path, output_directory = REFERENCE_ENDPOINTS[args.dataset]
        download_snapshot(
            path,
            output_directory,
            overwrite=args.overwrite,
        )
        return

    download_date_range(
        ENDPOINTS[args.dataset],
        args.start_date,
        args.end_date,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
