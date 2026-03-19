"""Generate a timestamp in the standard format."""

from argparse import ArgumentParser

from . import timestamp

if __name__ == "__main__":
    parser = ArgumentParser(description="generate a timestamp in the standard format")

    arguments = parser.parse_args()

    print(timestamp())
