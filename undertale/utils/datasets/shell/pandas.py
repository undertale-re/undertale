"""Load a dataset using ``pandas``."""

import pandas

from . import main as shell_main


def load(path: str) -> None:
    return pandas.read_parquet(path)


def main():
    shell_main(load, "pandas")


if __name__ == "__main__":
    main()
