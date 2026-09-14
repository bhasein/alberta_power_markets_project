"""Optionally download 60-day-delayed Energy Merit Order snapshots."""

from aeso_api.catalog import ENERGY_MERIT_ORDER
from aeso_api.runner import run_standard_range


def main() -> None:
    run_standard_range(ENERGY_MERIT_ORDER)


if __name__ == "__main__":
    main()
