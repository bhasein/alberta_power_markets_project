"""Optionally download 60-day-delayed unit-commitment directives."""

from aeso_api.catalog import UNIT_COMMITMENT
from aeso_api.runner import run_standard_range


def main() -> None:
    run_standard_range(UNIT_COMMITMENT)


if __name__ == "__main__":
    main()
