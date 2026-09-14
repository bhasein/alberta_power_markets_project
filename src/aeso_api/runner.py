"""Small command-line runner shared by endpoint-specific modules."""

from __future__ import annotations

import argparse

from aeso_api.catalog import DatedEndpoint
from aeso_api.download import add_range_arguments, download_date_range


def run_standard_range(endpoint: DatedEndpoint) -> None:
    """Parse standard arguments and download a configured endpoint."""

    parser = argparse.ArgumentParser(description=f"Download {endpoint.name}.")
    add_range_arguments(parser)
    args = parser.parse_args()
    download_date_range(
        endpoint,
        args.start_date,
        args.end_date,
        overwrite=args.overwrite,
    )
