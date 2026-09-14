"""Download a timestamped snapshot of the latest AESO pool participants."""

from __future__ import annotations

import argparse

from aeso_api.download import download_snapshot


def download_pool_participants(*, overwrite: bool = False):
    """Download or reuse the latest locally stored participant snapshot."""

    return download_snapshot(
        "/PoolParticipant-api/v1/poolparticipantlist",
        "pool_participants",
        overwrite=overwrite,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    download_pool_participants(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
