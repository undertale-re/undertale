"""Label nixpkgs function records against the function taxonomy using a vLLM server.

The corpus (`data/test_functions`) holds ~3.4M JSON records, so labeling runs as
sharded, resumable batches against an OpenAI-compatible vLLM endpoint that is started
and managed separately from this script. Each shard appends JSON lines to its own
file; `--merge` assembles the shards into the single path-keyed JSON file that
downstream consumers expect.

Only the standard library is used so the script runs inside any environment that can
reach the server.

Typical use::

    # one shard (repeat with --shard 1..N, or use the array job in label_functions.slurm)
    python label_functions.py --listing listing.txt --shard 0 --shards 8 \
        --endpoint http://a-5-10:12644/v1 --model <name the server reports at /v1/models>

    # score a shard's output against hand labels before committing to a full run
    python label_functions.py --merge --gold function_labels_sample_v2.json

    # assemble every shard into the final file
    python label_functions.py --merge --output function_labels.json
"""

import argparse
import json
import pathlib
import sys
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable, Iterator, Optional

HOME = pathlib.Path.home() / "undertale_shared/people/pa27879/data"
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


def load_taxonomy(path: pathlib.Path) -> tuple[list[str], list[str], str]:
    """Load the taxonomy file.

    Args:
        path: Path to the taxonomy JSON.

    Returns:
        A tuple of (class names in priority order, flag names, prompt text describing
        both).
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
        record: A parsed record from the corpus.

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


def label(
    path: pathlib.Path,
    endpoint: str,
    model: str,
    taxonomy: str,
    classes: list[str],
    flags: list[str],
    timeout: int,
) -> dict[str, Any]:
    """Label one record file, retrying transient failures and unusable replies.

    Args:
        path: Path to the record's JSON file.
        endpoint: Base URL of the vLLM API.
        model: Model name.
        taxonomy: Rendered taxonomy description.
        classes: Valid class names.
        flags: Valid flag names.
        timeout: Socket timeout in seconds.

    Returns:
        A result row carrying the file path and either labels or an error.
    """
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return {"file": str(path), "error": f"unreadable: {error}"}

    prompt = describe(record)
    failure = "no valid reply"
    for _ in range(RETRIES):
        try:
            content = request(endpoint, model, prompt, taxonomy, timeout)
        except (urllib.error.URLError, OSError, KeyError, ValueError) as error:
            failure = str(error)
            continue
        parsed = parse(content, classes, flags)
        if parsed:
            return {"file": str(path), **parsed}

    return {"file": str(path), "error": failure}


def shard(paths: list[pathlib.Path], index: int, total: int) -> list[pathlib.Path]:
    """Select this worker's slice of the corpus.

    Args:
        paths: Every record path.
        index: This shard's index.
        total: Total number of shards.

    Returns:
        The paths belonging to this shard.
    """
    return paths[index::total]


def done(destination: pathlib.Path) -> set[str]:
    """Read the file paths already labeled in a shard output, for resuming.

    Args:
        destination: The shard's JSON-lines output file.

    Returns:
        The set of file paths already present.
    """
    if not destination.exists():
        return set()

    finished = set()
    with destination.open() as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # A partial final line from a killed run.
            if "error" not in row:
                finished.add(row["file"])
    return finished


def run(arguments: argparse.Namespace) -> int:
    """Label this shard of the corpus.

    Args:
        arguments: Parsed command-line arguments.

    Returns:
        A process exit code.
    """
    classes, flags, taxonomy = load_taxonomy(arguments.taxonomy)

    if arguments.listing:
        names = arguments.listing.read_text().split()
        paths = [arguments.functions / name for name in names]
    else:
        paths = sorted(arguments.functions.glob("*.json"))

    selected = shard(paths, arguments.shard, arguments.shards)
    if arguments.limit:
        selected = selected[: arguments.limit]

    destination = arguments.destination / f"labels-{arguments.shard:05d}.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    finished = done(destination)
    pending = [path for path in selected if str(path) not in finished]

    print(
        f"shard {arguments.shard}/{arguments.shards}: {len(selected)} assigned, "
        f"{len(finished)} done, {len(pending)} pending -> {destination}",
        flush=True,
    )

    lock = threading.Lock()
    written = 0
    with (
        destination.open("a") as handle,
        ThreadPoolExecutor(max_workers=arguments.concurrency) as pool,
    ):
        results = pool.map(
            lambda path: label(
                path,
                arguments.endpoint,
                arguments.model,
                taxonomy,
                classes,
                flags,
                arguments.timeout,
            ),
            pending,
        )
        for result in results:
            with lock:
                handle.write(json.dumps(result) + "\n")
                written += 1
                if written % 500 == 0:
                    handle.flush()
                    print(f"  {written}/{len(pending)}", flush=True)

    print(f"shard {arguments.shard}: wrote {written} rows", flush=True)
    return 0


def rows(directory: pathlib.Path) -> Iterator[dict[str, Any]]:
    """Read every shard output in a directory.

    Args:
        directory: Directory holding `labels-*.jsonl` files.

    Yields:
        Each parsed result row.
    """
    for path in sorted(directory.glob("labels-*.jsonl")):
        with path.open() as handle:
            for line in handle:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def score(labeled: dict[str, list[str]], gold: dict[str, list[str]]) -> None:
    """Report agreement between model labels and hand labels.

    Args:
        labeled: Model labels keyed by file path.
        gold: Hand labels keyed by file path.
    """
    shared = [key for key in gold if key in labeled]
    if not shared:
        print("no overlap between model output and gold labels", file=sys.stderr)
        return

    top = sum(1 for key in shared if labeled[key][0] == gold[key][0])
    overlap = sum(1 for key in shared if set(labeled[key]) & set(gold[key]))
    exact = sum(1 for key in shared if set(labeled[key]) == set(gold[key]))

    print(f"compared against gold: {len(shared)}")
    print(f"  top-1 agreement:   {top / len(shared):.1%}")
    print(f"  any-label overlap: {overlap / len(shared):.1%}")
    print(f"  exact set match:   {exact / len(shared):.1%}")


def merge(arguments: argparse.Namespace) -> int:
    """Assemble shard outputs into one path-keyed JSON file.

    Args:
        arguments: Parsed command-line arguments.

    Returns:
        A process exit code.
    """
    labeled: dict[str, list[str]] = {}
    failures = 0
    for row in rows(arguments.destination):
        if "error" in row:
            failures += 1
            continue
        labeled[row["file"]] = row["labels"]

    if arguments.gold:
        score(labeled, json.loads(arguments.gold.read_text()))

    if arguments.output:
        arguments.output.write_text(json.dumps(labeled, indent=1, sort_keys=True))
        print(f"wrote {len(labeled)} labels ({failures} failed) -> {arguments.output}")
    else:
        print(f"{len(labeled)} labels ({failures} failed); pass --output to write them")

    return 0


def parse_arguments(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Build the command-line interface.

    Args:
        argv: Argument list, defaulting to `sys.argv`.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--taxonomy", type=pathlib.Path, default=HOME / "function_taxonomy_v2.json"
    )
    parser.add_argument(
        "--functions", type=pathlib.Path, default=HOME / "test_functions"
    )
    parser.add_argument(
        "--listing",
        type=pathlib.Path,
        help="File of record filenames, one per line. Much faster than globbing 3.4M files.",
    )
    parser.add_argument(
        "--destination", type=pathlib.Path, default=HOME / "function_labels"
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:8000/v1",
        help="Base URL of an already-running OpenAI-compatible vLLM server.",
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--limit", type=int, help="Label at most this many records.")
    parser.add_argument(
        "--merge", action="store_true", help="Assemble shard outputs instead."
    )
    parser.add_argument("--output", type=pathlib.Path, help="Merged output file.")
    parser.add_argument(
        "--gold", type=pathlib.Path, help="Hand labels to score against."
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main() -> int:
    """Entry point.

    Returns:
        A process exit code.
    """
    arguments = parse_arguments()
    return merge(arguments) if arguments.merge else run(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
