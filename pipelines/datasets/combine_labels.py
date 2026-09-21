"""Merge `label_functions.py` output back into the parquet shards it came from.

`export_functions.py --split` numbers every function it exports with a running
index taken in shard-then-row order over `sorted(glob(inputs))`, and writes it as
the filename prefix: `0002977_ropADD_AL_imm.json` is row 0 of `part.1.parquet`
when `part.0.parquet` holds 2977 rows. `label_functions.py` keys its output rows
by that filename, so the prefix is the join key back to the original rows -- no
column in the shards themselves identifies a function uniquely.

This script rebuilds that numbering from the shards' own row counts, then writes
a copy of every shard with three columns appended:

    labels       list<string>, most specific first, null when labeling failed
    flags        list<string>, the taxonomy's orthogonal flags, null on failure
    label_error  string, why labeling failed, null when it succeeded

Usage (needs pyarrow -- `module load anaconda/Python-ML-2025a`):

    python combine_labels.py \
        $SHARE/datasets/nixpkgs-legacy_10percent/split-testing/ \
        --labels $SHARE/people/pa27879/data/function_labels \
        --destination $SHARE/people/pa27879/data/split-testing-labeled

Shards are written one at a time and an existing output file is left alone, so a
killed run resumes by rerunning the same command. `--shard`/`--shards` split the
work across the tasks of an array job.

To go from the original shards to labeled ones in one step -- no JSON corpus in
between -- use `label_parquet.py` instead.
"""

import argparse
import glob
import json
import os
import pathlib
import re
import sys
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq

# Filenames `export_functions.py --split` writes: a zero-padded index, then the
# sanitized symbol name.
INDEX_RE = re.compile(r"^(\d+)_")

# Appended to every shard, in this order.
LABEL_FIELDS = [
    pa.field("labels", pa.list_(pa.large_string())),
    pa.field("flags", pa.list_(pa.large_string())),
    pa.field("label_error", pa.large_string()),
]


def resolve(patterns: list[str]) -> list[str]:
    """Expand input patterns into shard paths, in `export_functions.py` order.

    The ordering has to match exactly: it is what assigns each function its
    index, and so what the labels are keyed by.

    Args:
        patterns: Parquet files, directories, or globs.

    Returns:
        The matched paths, sorted the way `export_functions.py` sorts them.
    """
    paths = []
    for pattern in patterns:
        if os.path.isdir(pattern):
            matches = sorted(glob.glob(os.path.join(pattern, "*.parquet")))
        elif any(character in pattern for character in "*?["):
            matches = sorted(glob.glob(pattern))
        else:
            matches = [pattern]
        if not matches:
            sys.exit(f"no parquet files matched: {pattern}")
        paths.extend(matches)

    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        sys.exit("not a file: " + ", ".join(missing))
    return paths


def offsets(paths: list[str]) -> list[int]:
    """Read each shard's row count and turn it into a starting index.

    Args:
        paths: Shard paths in export order.

    Returns:
        One starting index per shard, plus a final entry holding the total.
    """
    starts = [0]
    for path in paths:
        starts.append(starts[-1] + pq.ParquetFile(path).metadata.num_rows)
    return starts


def read(directory: pathlib.Path, total: int) -> tuple[list, list, list]:
    """Load every shard of labeling output, indexed by function index.

    Indices are only meaningful against the whole export, so a row naming a
    function past the end means the shards given are not the ones the labels came
    from -- including the case of passing only some of them, where the join would
    otherwise silently attach each shard the wrong labels. Restrict the work with
    `--shard`/`--shards`, which keeps the full input list, rather than by passing
    fewer inputs.

    Args:
        directory: Directory holding `labels-*.jsonl`.
        total: Number of functions across every shard.

    Returns:
        A tuple of (labels, flags, errors), each a list of length `total` whose
        entries are None where nothing was recorded.
    """
    paths = sorted(directory.glob("labels-*.jsonl"))
    if not paths:
        sys.exit(f"no labels-*.jsonl in {directory}")

    labels: list[Optional[list[str]]] = [None] * total
    flags: list[Optional[list[str]]] = [None] * total
    errors: list[Optional[str]] = [None] * total

    # The same handful of label sets recur across millions of functions; sharing
    # one list per distinct set keeps the whole join in a few hundred megabytes.
    cache: dict[tuple[str, ...], list[str]] = {}

    def shared(names: list[str]) -> list[str]:
        key = tuple(names)
        if key not in cache:
            cache[key] = names
        return cache[key]

    seen = 0
    malformed = 0
    for path in paths:
        with path.open() as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1  # A partial final line from a killed run.
                    continue

                match = INDEX_RE.match(os.path.basename(row["file"]))
                if not match:
                    malformed += 1
                    continue
                index = int(match.group(1))
                if not 0 <= index < total:
                    sys.exit(
                        f"{path}: function index {index} is past the {total} rows "
                        "of the shards given. Pass every shard of the original "
                        "export and use --shard/--shards to split the work; the "
                        "index is assigned across the whole export, so a partial "
                        "input list joins the wrong labels to every shard."
                    )

                seen += 1
                if "error" in row:
                    errors[index] = row["error"]
                else:
                    labels[index] = shared(row["labels"])
                    flags[index] = shared(row["flags"])

    labeled = sum(1 for entry in labels if entry is not None)
    failed = sum(1 for entry in errors if entry is not None)
    print(
        f"{seen} label rows from {len(paths)} file(s): {labeled} labeled, "
        f"{failed} failed, {total - labeled - failed} missing"
        + (f", {malformed} unparseable" if malformed else "")
    )
    return labels, flags, errors


def compression(path: str) -> str:
    """Report the codec a shard was written with, so the copy matches it.

    Args:
        path: Shard path.

    Returns:
        A codec name pyarrow accepts, e.g. `none` or `snappy`.
    """
    metadata = pq.ParquetFile(path).metadata
    if not metadata.num_row_groups or not metadata.row_group(0).num_columns:
        return "none"
    codec = metadata.row_group(0).column(0).compression.lower()
    return "none" if codec == "uncompressed" else codec


def write(
    source: str,
    destination: pathlib.Path,
    start: int,
    labels: list,
    flags: list,
    errors: list,
    arguments: argparse.Namespace,
) -> tuple[int, int]:
    """Copy one shard with its label columns appended.

    The copy is written to a temporary name and moved into place at the end, so
    an interrupted run never leaves a short file that a later run would mistake
    for finished work.

    Args:
        source: Shard path.
        destination: Output path for the labeled copy.
        start: Function index of this shard's first row.
        labels: Labels for every function, indexed globally.
        flags: Flags for every function, indexed globally.
        errors: Errors for every function, indexed globally.
        arguments: Parsed command-line arguments.

    Returns:
        A tuple of (rows written, rows carrying labels).
    """
    parquet = pq.ParquetFile(source)
    codec = arguments.compression or compression(source)
    schema = pa.schema(list(parquet.schema_arrow) + LABEL_FIELDS)
    partial = destination.parent / (destination.name + ".partial")

    written = 0
    labeled = 0
    with pq.ParquetWriter(partial, schema, compression=codec) as writer:
        for batch in parquet.iter_batches(batch_size=arguments.batch_size):
            first = start + written
            last = first + batch.num_rows
            rows = slice(first, last)

            columns = list(batch.columns)
            for field, values in zip(
                LABEL_FIELDS, (labels[rows], flags[rows], errors[rows])
            ):
                columns.append(pa.array(values, type=field.type))

            table = pa.Table.from_arrays(columns, schema=schema)
            if arguments.drop_unlabeled:
                table = table.filter(table["labels"].is_valid())

            writer.write_table(table)
            written += batch.num_rows
            labeled += sum(1 for entry in labels[rows] if entry is not None)

    partial.replace(destination)
    return written, labeled


def run(arguments: argparse.Namespace) -> int:
    """Merge labels into every shard this worker is responsible for.

    Args:
        arguments: Parsed command-line arguments.

    Returns:
        A process exit code.
    """
    paths = resolve(arguments.inputs)
    starts = offsets(paths)
    total = starts[-1]
    print(f"{len(paths)} shard(s), {total} functions")

    labels, flags, errors = read(arguments.labels, total)

    arguments.destination.mkdir(parents=True, exist_ok=True)
    assigned = range(arguments.shard, len(paths), arguments.shards)

    written = 0
    labeled = 0
    skipped = 0
    for position in assigned:
        source = paths[position]
        destination = arguments.destination / os.path.basename(source)
        if destination.exists() and not arguments.overwrite:
            skipped += 1
            continue

        rows, found = write(
            source, destination, starts[position], labels, flags, errors, arguments
        )
        written += rows
        labeled += found
        print(f"  {os.path.basename(source)}: {rows} rows, {found} labeled", flush=True)

    print(
        f"shard {arguments.shard}/{arguments.shards}: wrote {written} rows "
        f"({labeled} labeled) across {len(assigned) - skipped} file(s), "
        f"{skipped} already present -> {arguments.destination}"
    )
    return 0


def parse_arguments(arguments: Optional[list[str]] = None) -> argparse.Namespace:
    """Build the command-line interface.

    Args:
        arguments: Argument list, defaulting to `sys.argv`.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("inputs", nargs="+", help="parquet file(s), directory, or glob")
    parser.add_argument(
        "--labels",
        type=pathlib.Path,
        required=True,
        help="directory of labels-*.jsonl written by label_functions.py",
    )
    parser.add_argument(
        "--destination",
        type=pathlib.Path,
        required=True,
        help="directory for the labeled shards",
    )
    parser.add_argument(
        "--drop-unlabeled",
        action="store_true",
        help="omit rows that have no labels instead of writing them with nulls",
    )
    parser.add_argument(
        "--compression", help="output codec; defaults to whatever each input shard used"
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="rows held in memory at a time"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="rewrite shards already present"
    )
    parser.add_argument("--shard", type=int, default=0, help="this worker's index")
    parser.add_argument("--shards", type=int, default=1, help="number of workers")
    return parser.parse_args(arguments)


def main() -> int:
    """Entry point.

    Returns:
        A process exit code.
    """
    return run(parse_arguments())


if __name__ == "__main__":
    raise SystemExit(main())
