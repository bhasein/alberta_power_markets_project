"""Download historical System Marginal Price intervals."""

from aeso_api.catalog import SYSTEM_MARGINAL_PRICE
from aeso_api.runner import run_standard_range


def main() -> None:
    run_standard_range(SYSTEM_MARGINAL_PRICE)


if __name__ == "__main__":
    main()
