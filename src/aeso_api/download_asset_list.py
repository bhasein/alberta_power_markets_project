"""Download a timestamped snapshot of the latest AESO asset list."""

import argparse

from aeso_api.download import download_snapshot


def download_asset_list(*, overwrite: bool = False):
    """Download or reuse the latest locally stored asset-list snapshot."""

    return download_snapshot(
        "/assetlist-api/v1/assetlist",
        "asset_list",
        overwrite=overwrite,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    download_asset_list(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
