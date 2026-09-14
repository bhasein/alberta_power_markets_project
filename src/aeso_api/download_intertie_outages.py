"""Download outages affecting AESO interties and flowgates."""

import argparse

from aeso_api.catalog import INTERTIE_OUTAGES
from aeso_api.download import add_range_arguments, download_date_range


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_range_arguments(parser)
    parser.add_argument(
        "--affected",
        choices=["BC", "MATL", "SK", "BC_MATL"],
        help="Optional affected intertie or flowgate filter.",
    )
    args = parser.parse_args()
    extra = (
        {"affectedIntertieOrFlowgate": args.affected}
        if args.affected
        else None
    )
    download_date_range(
        INTERTIE_OUTAGES,
        args.start_date,
        args.end_date,
        extra_parameters=extra,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
