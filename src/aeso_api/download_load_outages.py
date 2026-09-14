"""Download aggregate hourly load-outage forecasts."""

from aeso_api.catalog import LOAD_OUTAGE_FORECAST
from aeso_api.runner import run_standard_range


def main() -> None:
    run_standard_range(LOAD_OUTAGE_FORECAST)


if __name__ == "__main__":
    main()
