"""Label parquet shards against the function taxonomy.

Three columns are appended to each shard:

    labels       list<string>, most specific first, null when labeling failed
    flags        list<string>, the taxonomy's orthogonal flags, null on failure
    label_error  string, why labeling failed, null when it succeeded

Usage (needs pyarrow -- `module load anaconda/Python-ML-2025a`, and a vLLM server
already running elsewhere; see `label_functions.slurm` for how that is started):

    python label_parquet.py \
        $SHARE/datasets/nixpkgs-legacy_10percent/split-testing/ \
        --destination $SHARE/datasets/nixpkgs-legacy_10percent/ \
        --endpoint http://node-name:node-port

Work is split by shard, not by row, so each task writes whole files and a killed
run resumes by rerunning the same command -- shards already present are left
alone. `--shard`/`--shards` spread the shards across an array job's tasks.
"""

import argparse
import json
import os
import pathlib
import sys
import types
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import combine_labels
import export_functions
import label_functions
import pyarrow as pa
import pyarrow.parquet as pq

HOME = pathlib.Path(__file__).resolve().parent
DATA = HOME / "data"

# How `export_functions.build_record` is asked to shape each record: the model
# only sees source file paths, but `describe` reads them out of the parsed blocks,
# so the bodies have to be carried. They are dropped again after the prompt is
# built, one batch at a time.
RECORD_OPTIONS = types.SimpleNamespace(
    source="files",
    include_disassembly=False,
    include_decompilation=False,
    include_bytes=False,
)


def normalize(endpoint: str) -> str:
    """Accept a server address with or without the OpenAI-compatible `/v1` path.

    Args:
        endpoint: Base URL as given on the command line.

    Returns:
        The URL with exactly one trailing `/v1`.
    """
    endpoint = endpoint.rstrip("/")
    return endpoint if endpoint.endswith("/v1") else f"{endpoint}/v1"


def serving(endpoint: str) -> str:
    """Ask the server which model it is serving.

    Defaulting to this rather than to a name means the model recorded in the
    output cannot drift away from the one that actually answered.

    Args:
        endpoint: Base URL of the OpenAI-compatible API.

    Returns:
        The model name the server reports.
    """
    call = f"{endpoint}/models"
    try:
        with label_functions.OPENER.open(call, timeout=30) as response:
            body = json.loads(response.read())
        return body["data"][0]["id"]
    except (urllib.error.URLError, OSError, KeyError, IndexError, ValueError) as error:
        sys.exit(f"no vLLM server reachable at {call}: {error}")


def label(
    record: dict[str, Any], taxonomy: str, arguments: argparse.Namespace
) -> dict[str, Any]:
    """Label one record, retrying transient failures and unusable replies.

    This is `label_functions.label` without the file: it takes the record that
    `export_functions.build_record` just produced instead of reading it back off
    disk.

    Args:
        record: A record as `export_functions.build_record` returns it.
        taxonomy: The rendered taxonomy description.
        arguments: Parsed command-line arguments, for the endpoint and model.

    Returns:
        A dict with `labels` and `flags`, or with `error` when every retry failed.
    """
    prompt = label_functions.describe(record)
    failure = "no valid reply"
    for _ in range(label_functions.RETRIES):
        try:
            content = label_functions.request(
                arguments.endpoint, arguments.model, prompt, taxonomy, arguments.timeout
            )
        except (urllib.error.URLError, OSError, KeyError, ValueError) as error:
            failure = str(error)
            continue

        parsed = label_functions.parse(content, arguments.classes, arguments.flags)
        if parsed:
            return parsed

    return {"error": failure}


def results(
    batch: pa.RecordBatch,
    taxonomy: str,
    pool: ThreadPoolExecutor,
    arguments: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Label every row of one batch, in row order.

    Args:
        batch: Rows read from the shard.
        taxonomy: The rendered taxonomy description.
        pool: Worker pool shared across batches.
        arguments: Parsed command-line arguments.

    Returns:
        One result dict per row, aligned with the batch.
    """
    records = [
        export_functions.build_record(index, row, RECORD_OPTIONS)
        for index, row in enumerate(batch.to_pylist())
    ]
    return list(pool.map(lambda record: label(record, taxonomy, arguments), records))


def write(
    source: str,
    destination: pathlib.Path,
    taxonomy: str,
    pool: ThreadPoolExecutor,
    arguments: argparse.Namespace,
) -> tuple[int, int]:
    """Copy one shard with labels for every row appended.

    Written to a temporary name and moved into place at the end, so an
    interrupted run never leaves a short file that a later run would mistake for
    finished work.

    Args:
        source: Shard path.
        destination: Output path for the labeled copy.
        taxonomy: The rendered taxonomy description.
        pool: Worker pool shared across batches.
        arguments: Parsed command-line arguments.

    Returns:
        A tuple of (rows written, rows carrying labels).
    """
    parquet = pq.ParquetFile(source)
    codec = arguments.compression or combine_labels.compression(source)
    schema = pa.schema(list(parquet.schema_arrow) + combine_labels.LABEL_FIELDS)
    partial = destination.parent / (destination.name + ".partial")

    written = 0
    labeled = 0
    with pq.ParquetWriter(partial, schema, compression=codec) as writer:
        for batch in parquet.iter_batches(batch_size=arguments.batch_size):
            rows = results(batch, taxonomy, pool, arguments)

            columns = list(batch.columns)
            for field, key in zip(
                combine_labels.LABEL_FIELDS, ("labels", "flags", "error")
            ):
                values = [row.get(key) for row in rows]
                columns.append(pa.array(values, type=field.type))

            table = pa.Table.from_arrays(columns, schema=schema)
            if arguments.drop_unlabeled:
                table = table.filter(table["labels"].is_valid())

            writer.write_table(table)
            written += batch.num_rows
            labeled += sum(1 for row in rows if "labels" in row)

    partial.replace(destination)
    return written, labeled


def run(arguments: argparse.Namespace) -> int:
    """Label every shard this worker is responsible for.

    Args:
        arguments: Parsed command-line arguments.

    Returns:
        A process exit code.
    """
    arguments.endpoint = normalize(arguments.endpoint)
    arguments.model = arguments.model or serving(arguments.endpoint)
    arguments.classes, arguments.flags, taxonomy = label_functions.load_taxonomy(
        arguments.taxonomy
    )

    paths = combine_labels.resolve(arguments.inputs)
    assigned = list(range(arguments.shard, len(paths), arguments.shards))
    if arguments.limit:
        assigned = assigned[: arguments.limit]

    arguments.destination.mkdir(parents=True, exist_ok=True)
    print(
        f"shard {arguments.shard}/{arguments.shards}: {len(assigned)} of "
        f"{len(paths)} file(s) -> {arguments.endpoint} ({arguments.model})",
        flush=True,
    )

    written = 0
    labeled = 0
    skipped = 0
    with ThreadPoolExecutor(max_workers=arguments.concurrency) as pool:
        for position in assigned:
            source = paths[position]
            destination = arguments.destination / os.path.basename(source)
            if destination.exists() and not arguments.overwrite:
                skipped += 1
                continue

            rows, found = write(source, destination, taxonomy, pool, arguments)
            written += rows
            labeled += found
            print(
                f"  {os.path.basename(source)}: {rows} rows, {found} labeled",
                flush=True,
            )

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
        "--destination",
        type=pathlib.Path,
        required=True,
        help="directory for the labeled shards",
    )
    parser.add_argument(
        "--taxonomy", type=pathlib.Path, default=DATA / "function_taxonomy_v2.json"
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:8000/v1",
        help="base URL of an already-running OpenAI-compatible vLLM server",
    )
    parser.add_argument(
        "--model", help="defaults to whatever the server reports at /v1/models"
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
        "--concurrency", type=int, default=32, help="requests in flight at once"
    )
    parser.add_argument(
        "--timeout", type=int, default=120, help="per-request socket timeout"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="rewrite shards already present"
    )
    parser.add_argument("--limit", type=int, help="stop after this many shards")
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
