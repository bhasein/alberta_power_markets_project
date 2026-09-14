"""Download AIES capacity, availability, and outage data by fuel type."""

from aeso_api.catalog import GENERATION_CAPACITY
from aeso_api.runner import run_standard_range


def main() -> None:
    run_standard_range(GENERATION_CAPACITY)


if __name__ == "__main__":
    main()
