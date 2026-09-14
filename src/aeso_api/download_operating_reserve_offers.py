"""Optionally download 60-day-delayed operating-reserve offer control data."""

from aeso_api.catalog import OPERATING_RESERVE_OFFER_CONTROL
from aeso_api.runner import run_standard_range


def main() -> None:
    run_standard_range(OPERATING_RESERVE_OFFER_CONTROL)


if __name__ == "__main__":
    main()
