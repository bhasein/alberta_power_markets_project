"""Download hourly AESO pool price and published forecast price."""

from aeso_api.catalog import POOL_PRICE
from aeso_api.runner import run_standard_range


def main() -> None:
    run_standard_range(POOL_PRICE)


if __name__ == "__main__":
    main()
