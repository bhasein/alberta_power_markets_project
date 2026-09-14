"""Download historical AESO actual and forecast Alberta Internal Load."""

from aeso_api.catalog import ACTUAL_FORECAST
from aeso_api.runner import run_standard_range


def main() -> None:
    run_standard_range(ACTUAL_FORECAST)


if __name__ == "__main__":
    main()
