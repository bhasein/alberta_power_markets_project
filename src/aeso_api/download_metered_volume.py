"""Optionally download settled hourly metered volume by asset."""

import argparse

from aeso_api.catalog import METERED_VOLUME
from aeso_api.download import add_range_arguments, download_date_range


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_range_arguments(parser)
    filters = parser.add_mutually_exclusive_group()
    filters.add_argument("--asset-ids", help="Up to 20 comma-separated asset IDs.")
    filters.add_argument(
        "--participant-ids",
        help="Up to 20 comma-separated pool participant IDs.",
    )
    args = parser.parse_args()
    extra = None
    if args.asset_ids:
        extra = {"asset_ID": args.asset_ids}
    elif args.participant_ids:
        extra = {"pool_participant_ID": args.participant_ids}
    download_date_range(
        METERED_VOLUME,
        args.start_date,
        args.end_date,
        extra_parameters=extra,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
