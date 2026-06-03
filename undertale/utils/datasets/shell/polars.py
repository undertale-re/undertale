"""Load a dataset using ``polars`` (lazy)."""

import polars

from . import main as shell_main


def load(path: str) -> None:
    return polars.scan_parquet(path)


def main():
    shell_main(load, "polars")


if __name__ == "__main__":
    main()
