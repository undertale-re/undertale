"""Compute the hash of one or more files."""

from argparse import ArgumentParser

from . import hash

if __name__ == "__main__":
    parser = ArgumentParser(description="compute the hash of one or more files")

    parser.add_argument("input", nargs="+", help="path to the file(s) to hash")

    arguments = parser.parse_args()

    for path in arguments.input:
        with open(path, "rb") as f:
            print(f"{hash(f.read())}  {path}")
