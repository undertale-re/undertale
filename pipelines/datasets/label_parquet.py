"""Label parquet shards against the function taxonomy.

Three columns are appended to each shard:

    labels       list<string>, most specific first, null when labeling failed
    flags        list<string>, the taxonomy's orthogonal flags, null on failure
    label_error  string, why labeling failed, null when it succeeded

    python label_parquet.py \
        $SHARE/datasets/nixpkgs-legacy_10percent/split-testing/ \
        --destination $SHARE/datasets/nixpkgs-legacy_10percent/ \
        --endpoint http://node-name:node-port

Work is split by shard, not by row, so each task writes whole files and a killed
run resumes by rerunning the same command -- shards already present are left
alone. `--shard`/`--shards` spread the shards across an array job's tasks.

This file is self-contained. What it needs of `export_functions.py` (carving a
function's definition out of the `source` column), of `label_functions.py`
(prompting the server, validating its replies) and of `combine_labels.py`
(finding shards, matching their compression) is inlined below, pruned to the
paths this pipeline actually takes. Those three scripts are unchanged and still
stand on their own for the JSON-corpus route.
"""

import argparse
import glob
import json
import os
import pathlib
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import pyarrow as pa
import pyarrow.parquet as pq

HOME = pathlib.Path(__file__).resolve().parent
DATA = HOME / "data"

SOURCE_LIMIT = 1200
RETRIES = 3

# The cluster sets http_proxy, which urllib would otherwise apply to the vLLM
# endpoint; the server is always in-cluster, so go to it directly.
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

SYSTEM = """You classify binary-derived functions into a fixed taxonomy.

Rules:
- Assign one or more classes, most specific first. Assign the catch-all
  "Domain / Application Logic" only when no other class fits.
- Use ONLY class names from the taxonomy, copied exactly.
- The symbol name and its package are the primary evidence. The listed source files
  are a retrieval guess and are frequently WRONG: treat them as weak context only,
  and ignore them when they disagree with the symbol name.
- Also report orthogonal flags that apply.
- Reply with JSON only: {"labels": ["..."], "flags": ["..."]}
"""

# Appended to every shard, in this order.
LABEL_FIELDS = [
    pa.field("labels", pa.list_(pa.large_string())),
    pa.field("flags", pa.list_(pa.large_string())),
    pa.field("label_error", pa.large_string()),
]

# Columns carried out of a row and into the prompt.
META_COLUMNS = ["binary_path", "package", "language", "function_name"]

# Start of a fenced block: a path on its own line, then ```.
BLOCK_RE = re.compile(r"^([^\s:][^\n:]*):\n```\n", re.MULTILINE)

# Suffixes the compiler appends to a symbol; the C definition uses the base name.
SUFFIX_RE = re.compile(r"\.(cold|part|isra|constprop|localalias)(\.\d+)*$")
TRAILING_NUM_RE = re.compile(r"\.\d+$")


# Carving a function out of a shard's `source` column, from `export_functions.py`.


def split_source(source: Optional[str]) -> list[dict[str, Optional[str]]]:
    """Split a `source` cell into [{"path": ..., "text": ...}].

    Args:
        source: The cell's contents, as path-labelled fenced blocks.

    Returns:
        One entry per block, or a single unnamed block if the cell is not fenced
        as expected.
    """
    if not source:
        return []

    marks = list(BLOCK_RE.finditer(source))
    if not marks:
        return [{"path": None, "text": source.strip()}]

    blocks = []
    for position, mark in enumerate(marks):
        start = mark.end()
        end = marks[position + 1].start() if position + 1 < len(marks) else len(source)
        text = source[start:end].rstrip()
        if text.endswith("```"):
            text = text[: -len("```")].rstrip()
        blocks.append({"path": mark.group(1), "text": text})
    return blocks


def base_name(name: Optional[str]) -> Optional[str]:
    """Strip compiler-added suffixes: `remove_node.cold.1` -> `remove_node`.

    Args:
        name: The symbol name as the shard records it.

    Returns:
        The name the C definition is expected to use.
    """
    if not name:
        return name
    return TRAILING_NUM_RE.sub("", SUFFIX_RE.sub("", name))


def match_delimiter(
    text: str, position: int, opening: str, closing: str
) -> Optional[int]:
    """Find the delimiter closing the one at `position`.

    String and character literals and both comment styles are skipped, so braces
    inside them do not unbalance the count.

    Args:
        text: The file body being scanned.
        position: Index of the opening delimiter.
        opening: The opening delimiter character.
        closing: The closing delimiter character.

    Returns:
        Index of the matching delimiter, or None when there is none.
    """
    if position >= len(text) or text[position] != opening:
        return None

    depth = 0
    end = len(text)
    while position < end:
        character = text[position]
        if character == "\\" and position + 1 < end:
            position += 2
            continue

        if character in "\"'":
            quote = character
            position += 1
            while position < end:
                if text[position] == "\\":
                    position += 2
                    continue
                if text[position] == quote:
                    break
                position += 1
        elif text.startswith("//", position):
            newline = text.find("\n", position)
            position = end if newline == -1 else newline
        elif text.startswith("/*", position):
            close = text.find("*/", position + 2)
            position = end if close == -1 else close + 1
        elif character == opening:
            depth += 1
        elif character == closing:
            depth -= 1
            if depth == 0:
                return position
        position += 1
    return None


def skip_to_body(text: str, position: int) -> Optional[int]:
    """Find the `{` opening a function body at or after `position`.

    Whitespace and whole preprocessor lines may sit between the parameter list
    and the body -- box64's interpreter files guard the declarator itself:

        #ifdef TEST_INTERPRETER
        uintptr_t Test66F30F(x64test_t* test, rex_t rex, uintptr_t addr)
        #else
        uintptr_t Run66F30F(x64emu_t* emu, rex_t rex, uintptr_t addr)
        #endif
        {

    A prototype ends in `;` rather than `{`, so it is rejected here.

    Args:
        text: The file body being scanned.
        position: Index just past the parameter list.

    Returns:
        Index of the opening brace, or None when this is not a definition.
    """
    end = len(text)
    while position < end:
        character = text[position]
        if character in " \t\r\n":
            position += 1
        elif character == "#":
            newline = text.find("\n", position)
            if newline == -1:
                return None
            position = newline + 1
        elif character == "{":
            return position
        else:
            return None
    return None


def line_start(text: str, position: int) -> int:
    """Walk back to the start of the declaration (return type, storage class).

    Args:
        text: The file body being scanned.
        position: Index of the symbol name.

    Returns:
        Index the extracted definition should start at.
    """
    start = text.rfind("\n", 0, position) + 1

    # Pull in a preceding line when the return type sits on its own line.
    previous_end = start - 1
    if previous_end > 0:
        previous_start = text.rfind("\n", 0, previous_end) + 1
        previous = text[previous_start:previous_end].strip()
        if (
            previous
            and not previous.endswith((";", "}", "{", ")", ":", "/"))
            and not previous.startswith("#")
        ):
            return previous_start
    return start


def extract_definition(text: str, name: Optional[str]) -> Optional[str]:
    """Best-effort carve-out of `name`'s definition from a C file body.

    Looks for `name (` at the start of a declarator, checks that a `{` follows
    the parameter list (i.e. it is a definition and not a call or prototype),
    then brace-matches to the end.

    Args:
        text: The file body to search.
        name: The function's base symbol name.

    Returns:
        The definition, or None when nothing matches. Macro-generated functions
        (box64's `my_<callback>_<n>` bridges) legitimately have no textual
        definition.
    """
    if not name:
        return None

    for match in re.finditer(r"(?<![\w.$])" + re.escape(name) + r"\s*\(", text):
        parameters_end = match_delimiter(text, match.end() - 1, "(", ")")
        if parameters_end is None:
            continue
        body_start = skip_to_body(text, parameters_end + 1)
        if body_start is None:
            continue
        body_end = match_delimiter(text, body_start, "{", "}")
        if body_end is None:
            continue
        return text[line_start(text, match.start()) : body_end + 1]
    return None


def build_record(row: dict[str, Any]) -> dict[str, Any]:
    """Turn one parquet row into the record `describe` reads.

    The model is only shown source file paths, but `describe` reads them out of
    the parsed blocks, so the whole file bodies are carried here. They are
    dropped again once the batch's prompts are built.

    Args:
        row: One row of a shard, as a dict.

    Returns:
        The record, carrying the row's metadata, its parsed source blocks, and
        the function's own definition where one could be carved out.
    """
    record = {column: row[column] for column in META_COLUMNS if column in row}

    blocks = split_source(row.get("source"))
    name = base_name(row.get("function_name"))

    definition = None
    for block in blocks:
        if block["text"] is None:
            definition = None
        else:
            definition = extract_definition(block["text"], name)
        if definition is not None:
            break

    record["function_source"] = definition
    record["source_files"] = blocks
    return record


# Prompting the server and reading its replies, from `label_functions.py`.


def load_taxonomy(path: pathlib.Path) -> tuple[list[str], list[str], str]:
    """Load the taxonomy file.

    Args:
        path: Path to the taxonomy JSON.

    Returns:
        A tuple of (class names in priority order, flag names, prompt text
        describing both).
    """
    taxonomy = json.loads(path.read_text())
    classes = [entry["name"] for entry in taxonomy["classes"]]
    flags = [entry["name"] for entry in taxonomy["orthogonal_flags"]]

    lines = ["Classes, ordered most specific to most generic:"]
    for entry in taxonomy["classes"]:
        lines.append(f"- {entry['name']}: {entry['description']}")
    lines.append("")
    lines.append("Orthogonal flags (independent of class):")
    for entry in taxonomy["orthogonal_flags"]:
        lines.append(f"- {entry['name']}: {entry['description']}")

    return classes, flags, "\n".join(lines)


def describe(record: dict[str, Any]) -> str:
    """Render one function record as the prompt's evidence block.

    Args:
        record: A record as `build_record` returns it.

    Returns:
        The evidence block, symbol name first.
    """
    paths = [entry.get("path") for entry in (record.get("source_files") or [])]
    paths = [path for path in paths if path][:3]

    lines = [
        f"symbol: {record.get('function_name')}",
        f"language: {record.get('language')}",
        f"package: {record.get('package')}",
        f"binary: {record.get('binary_path')}",
        f"source files (WEAK, often wrong): {', '.join(paths) or '-'}",
    ]

    source = record.get("function_source") or ""
    if source:
        lines.append("function source:")
        lines.append(source[:SOURCE_LIMIT])
    else:
        lines.append("function source: not recovered (label from the symbol name)")

    return "\n".join(lines)


def request(endpoint: str, model: str, prompt: str, taxonomy: str, timeout: int) -> str:
    """Send one completion request to the vLLM server.

    Args:
        endpoint: Base URL of the OpenAI-compatible API, e.g. `http://host:8000/v1`.
        model: Model name the server was started with.
        prompt: The evidence block for one function.
        taxonomy: The rendered taxonomy description.
        timeout: Socket timeout in seconds.

    Returns:
        The raw assistant message content.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{taxonomy}\n\nFunction:\n{prompt}"},
        ],
        "temperature": 0.0,
        "max_tokens": 200,
        "response_format": {"type": "json_object"},
    }
    call = urllib.request.Request(
        f"{endpoint.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with OPENER.open(call, timeout=timeout) as response:
        body = json.loads(response.read())
    return body["choices"][0]["message"]["content"]


def parse(
    content: str, classes: list[str], flags: list[str]
) -> Optional[dict[str, Any]]:
    """Parse and validate a model reply.

    Args:
        content: Raw assistant message content.
        classes: Valid class names.
        flags: Valid flag names.

    Returns:
        A dict with `labels` and `flags`, or None when the reply is unusable.
    """
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start < 0 or end < start:
            return None
        try:
            parsed = json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            return None

    labels = [name for name in parsed.get("labels", []) if name in classes]
    if not labels:
        return None

    # Preserve taxonomy order so the first label is always the most specific one.
    labels = sorted(set(labels), key=classes.index)
    return {
        "labels": labels,
        "flags": sorted({name for name in parsed.get("flags", []) if name in flags}),
    }


# Finding shards and matching how they were written, from `combine_labels.py`.


def resolve(patterns: list[str]) -> list[str]:
    """Expand input patterns into shard paths.

    Sorted, so that a resubmitted job hands each task the same shards it had
    before and the work already on disk is the work it skips.

    Args:
        patterns: Parquet files, directories, or globs.

    Returns:
        The matched paths.
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


# The pipeline itself.


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
        with OPENER.open(call, timeout=30) as response:
            body = json.loads(response.read())
        return body["data"][0]["id"]
    except (urllib.error.URLError, OSError, KeyError, IndexError, ValueError) as error:
        sys.exit(f"no vLLM server reachable at {call}: {error}")


def label(
    record: dict[str, Any], taxonomy: str, arguments: argparse.Namespace
) -> dict[str, Any]:
    """Label one record, retrying transient failures and unusable replies.

    Args:
        record: A record as `build_record` returns it.
        taxonomy: The rendered taxonomy description.
        arguments: Parsed command-line arguments, for the endpoint and model.

    Returns:
        A dict with `labels` and `flags`, or with `error` when every retry failed.
    """
    prompt = describe(record)
    failure = "no valid reply"
    for _ in range(RETRIES):
        try:
            content = request(
                arguments.endpoint, arguments.model, prompt, taxonomy, arguments.timeout
            )
        except (urllib.error.URLError, OSError, KeyError, ValueError) as error:
            failure = str(error)
            continue

        parsed = parse(content, arguments.classes, arguments.flags)
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
    records = [build_record(row) for row in batch.to_pylist()]
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
    codec = arguments.compression or compression(source)
    schema = pa.schema(list(parquet.schema_arrow) + LABEL_FIELDS)
    partial = destination.parent / (destination.name + ".partial")

    written = 0
    labeled = 0
    with pq.ParquetWriter(partial, schema, compression=codec) as writer:
        for batch in parquet.iter_batches(batch_size=arguments.batch_size):
            rows = results(batch, taxonomy, pool, arguments)

            columns = list(batch.columns)
            for field, key in zip(LABEL_FIELDS, ("labels", "flags", "error")):
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
    arguments.classes, arguments.flags, taxonomy = load_taxonomy(arguments.taxonomy)

    paths = resolve(arguments.inputs)
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
