"""Label parquet shards against the function taxonomy.

Four columns are appended to each shard:

    labels       list<string>, most specific first, null when labeling failed
    flags        list<string>, the taxonomy's orthogonal flags, null on failure
    label_error  string, why labeling failed, null when it succeeded
    label_model  string, the model that produced the row's labels, for
                 provenance; the same value on every row of a run

`tokenize-disassembly.py` also requires `id` and `name` columns, which the
nixpkgs shards lack, so each is appended when the shard does not have it:

    id    string, `<shard>:<row>`, the row's position in the input shard
    name  string, a copy of `function_name`

With `--drop-unlabeled`, the output can be tokenized as it is.

    python label_parquet.py \
        $SHARE/datasets/nixpkgs-legacy_10percent/split-testing/ \
        --destination $SHARE/datasets/nixpkgs-legacy_10percent/ \
        --endpoint http://node-name:node-port

Work is split by shard, not by row, so each task writes whole files and a killed
run resumes by rerunning the same command -- shards already present are left
alone. `--shard`/`--shards` spread the shards across an array job's tasks.

`--validate` scores this same labeling against a hand-labeled set instead of
writing shards. `labelling_validation.json` beside this file holds the sampled
functions; add a `labels` list (most specific first, and optionally a `flags`
list) to the entries you have checked by hand, and only those entries are
scored. The model labels each one through the very path the pipeline uses and
the predictions are reported with the same multi-label F1 the classifier itself
trains against.

    python label_parquet.py --validate --endpoint http://node-name:node-port

This file is self-contained, and `label_parquet.slurm` beside it runs it as an
array job.
"""

import argparse
import glob
import json
import os
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import pyarrow as pa
import pyarrow.parquet as pq

HOME = pathlib.Path(__file__).resolve().parent

SOURCE_LIMIT = 1200
RETRIES = 3

# Seconds to wait before the first retry; each further retry doubles it, so a
# momentarily overloaded or restarting server gets a chance to recover instead
# of being hammered by back-to-back attempts.
BACKOFF = 0.5

# Don't judge a shard's error rate until at least this many rows have been
# attempted, so a single early failure on a small shard can't abort it.
ERROR_SAMPLE = 64


class ShardAborted(Exception):
    """A shard's infrastructure-error rate crossed the abort threshold.

    Raised while labeling a shard when connection or HTTP errors -- a dead or
    stalled server, rather than replies the model simply got wrong -- make up
    too large a share of the rows seen so far. The partial output is discarded
    and never renamed into place, so a later run retries the whole shard instead
    of mistaking a near-empty file for finished work.
    """

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

# A provenance tag recording which model did the labeling, appended after the
# label fields. Its value is constant for a run, so it is filled per batch
# rather than coming out of each row's result.
LABEL_MODEL_FIELD = pa.field("label_model", pa.large_string())

# Columns carried out of a row and into the prompt.
META_COLUMNS = ["binary_path", "package", "language", "function_name"]

# Start of a fenced block: a path on its own line, then ```.
BLOCK_RE = re.compile(r"^([^\s:][^\n:]*):\n```\n", re.MULTILINE)

# Suffixes the compiler appends to a symbol; the C definition uses the base name.
SUFFIX_RE = re.compile(r"\.(cold|part|isra|constprop|localalias)(\.\d+)*$")
TRAILING_NUM_RE = re.compile(r"\.\d+$")


# Carving a function out of a shard's `source` column.


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

    This recognises only C-style declarators: anything between the parameter
    list and the body that is not whitespace or a preprocessor line makes it
    give up. That is deliberately narrow, and it is why non-C languages recover
    bodies poorly (see the taxonomy caveats). It notably rejects:

    - Rust return types, `fn f(x: i32) -> T {`;
    - Go return types and multiple results, `func f() (int, error) {`;
    - C++ trailing specifiers, `const`, `noexcept`, `override`, `-> T`.

    Broadening it would mean tolerating a bounded run of such tokens here before
    the brace; for now the limitation is documented rather than fixed.

    Args:
        text: The file body being scanned.
        position: Index just past the parameter list.

    Returns:
        Index of the opening brace, or None when this is not a C definition.
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
        definition = extract_definition(block["text"], name)
        if definition is not None:
            break

    record["function_source"] = definition
    record["source_files"] = blocks
    return record


# Prompting the server and reading its replies.


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


def request(
    endpoint: str,
    model: str,
    prompt: str,
    taxonomy: str,
    timeout: int,
    temperature: float = 0.0,
) -> str:
    """Send one completion request to the vLLM server.

    Args:
        endpoint: Base URL of the OpenAI-compatible API, e.g. `http://host:8000/v1`.
        model: Model name the server was started with.
        prompt: The evidence block for one function.
        taxonomy: The rendered taxonomy description.
        timeout: Socket timeout in seconds.
        temperature: Sampling temperature; raised above zero on retries so a
            reply that could not be parsed is not simply regenerated verbatim.

    Returns:
        The raw assistant message content.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{taxonomy}\n\nFunction:\n{prompt}"},
        ],
        "temperature": temperature,
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

    # A valid JSON reply need not be the object we asked for: a top-level array
    # has no `.get`, and `"labels": null` is not iterable. Treat anything that is
    # not a well-formed object as an unusable reply rather than letting it raise.
    if not isinstance(parsed, dict):
        return None

    labels = [name for name in (parsed.get("labels") or []) if name in classes]
    if not labels:
        return None

    # Preserve taxonomy order so the first label is always the most specific one.
    labels = sorted(set(labels), key=classes.index)
    return {
        "labels": labels,
        "flags": sorted(
            {name for name in (parsed.get("flags") or []) if name in flags}
        ),
    }


# Finding shards and matching how they were written.


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


# Scoring the labeling against a hand-labeled set.


def load_validation(
    path: pathlib.Path, classes: list[str]
) -> tuple[list[dict[str, Any]], set[str]]:
    """Read the hand-labeled validation entries.

    The file is the sampled-functions pool, either as a dict keyed by
    `<index>_<name>` or as a plain list of records. Only entries carrying a
    non-empty `labels` list count as ground truth; the rest are the unlabeled
    bulk and are skipped.

    Args:
        path: Path to the validation JSON.
        classes: Valid class names, for spotting typos in the gold labels.

    Returns:
        A tuple of (gold examples, class names used in the gold labels that are
        not in the taxonomy). Each example carries the record under `record`,
        its gold `labels`, and its gold `flags`.
    """
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        items: list[tuple[str, Any]] = list(data.items())
    else:
        items = [(str(record.get("index", i)), record) for i, record in enumerate(data)]

    examples = []
    unknown: set[str] = set()
    for key, record in items:
        if not isinstance(record, dict):
            continue
        gold = record.get("labels")
        if not gold:
            continue

        unknown.update(name for name in gold if name not in classes)
        examples.append(
            {
                "key": key,
                "record": record,
                "labels": list(gold),
                "flags": list(record.get("flags") or []),
            }
        )
    return examples, unknown


def counts(
    gold_sets: list[list[str]], predicted_sets: list[list[str]], names: list[str]
) -> tuple[list[int], list[int], list[int]]:
    """Tally per-name true positives, false positives and false negatives.

    Args:
        gold_sets: Ground-truth name lists, one per example.
        predicted_sets: Predicted name lists, aligned with `gold_sets`.
        names: The names (classes or flags) to score, defining the columns.

    Returns:
        A tuple of (true positives, false positives, false negatives), each a
        list aligned with `names`.
    """
    index = {name: position for position, name in enumerate(names)}
    true_positives = [0] * len(names)
    false_positives = [0] * len(names)
    false_negatives = [0] * len(names)

    for gold, predicted in zip(gold_sets, predicted_sets):
        want = {index[name] for name in gold if name in index}
        got = {index[name] for name in predicted if name in index}
        for position in got & want:
            true_positives[position] += 1
        for position in got - want:
            false_positives[position] += 1
        for position in want - got:
            false_negatives[position] += 1

    return true_positives, false_positives, false_negatives


def f1(true_positive: int, false_positive: int, false_negative: int) -> float:
    """F1 from one set of counts, zero when there is nothing to score.

    Args:
        true_positive: Correctly predicted positives.
        false_positive: Predicted positives that were wrong.
        false_negative: Missed positives.

    Returns:
        The F1 score.
    """
    denominator = 2 * true_positive + false_positive + false_negative
    return 2 * true_positive / denominator if denominator else 0.0


def score(gold_sets: list[list[str]], predicted_sets: list[list[str]], names: list[str]):
    """Score predicted name sets against gold name sets.

    Micro and macro F1 are defined exactly as the multi-label classifier
    computes them: micro pools every class's counts, macro averages per-class F1
    over the classes that appear in either the gold labels or the predictions.

    Args:
        gold_sets: Ground-truth name lists, one per example.
        predicted_sets: Predicted name lists, aligned with `gold_sets`.
        names: The names (classes or flags) to score.

    Returns:
        A dict of overall scores and a per-name breakdown.
    """
    true_positives, false_positives, false_negatives = counts(
        gold_sets, predicted_sets, names
    )

    micro = f1(sum(true_positives), sum(false_positives), sum(false_negatives))

    per_name = {}
    macro_scores = []
    for position, name in enumerate(names):
        tp = true_positives[position]
        fp = false_positives[position]
        fn = false_negatives[position]
        if tp + fp + fn == 0:
            continue
        name_f1 = f1(tp, fp, fn)
        per_name[name] = {"f1": name_f1, "tp": tp, "fp": fp, "fn": fn}
        macro_scores.append(name_f1)

    macro = sum(macro_scores) / len(macro_scores) if macro_scores else 0.0
    return {"micro_f1": micro, "macro_f1": macro, "per_name": per_name}


def validate(arguments: argparse.Namespace) -> int:
    """Score the labeling against the hand-labeled validation set.

    Args:
        arguments: Parsed command-line arguments.

    Returns:
        A process exit code: non-zero when no gold examples could be scored.
    """
    arguments.endpoint = normalize(arguments.endpoint)
    arguments.model = arguments.model or serving(arguments.endpoint)
    arguments.classes, arguments.flags, taxonomy = load_taxonomy(arguments.taxonomy)

    examples, unknown = load_validation(arguments.validation_set, arguments.classes)
    if unknown:
        print(
            "warning: gold labels use names absent from the taxonomy: "
            + ", ".join(sorted(unknown)),
            file=sys.stderr,
        )
    if not examples:
        print(
            f"no hand-labeled entries in {arguments.validation_set}: add a "
            "`labels` list to the entries you have checked by hand",
            file=sys.stderr,
        )
        return 1
    if arguments.limit:
        examples = examples[: arguments.limit]

    print(
        f"validating {len(examples)} hand-labeled function(s) -> "
        f"{arguments.endpoint} ({arguments.model})",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=arguments.concurrency) as pool:
        predictions = list(
            pool.map(
                lambda example: label(example["record"], taxonomy, arguments),
                examples,
            )
        )

    gold_labels = []
    gold_flags = []
    predicted_labels = []
    predicted_flags = []
    exact = 0
    primary = 0
    errored = 0
    mistakes = []
    for example, prediction in zip(examples, predictions):
        if "error" in prediction:
            errored += 1
            continue

        # Keep gold order (most specific first) only for the primary comparison;
        # set membership drives the F1 counts.
        gold = example["labels"]
        predicted = prediction["labels"]
        gold_labels.append(gold)
        predicted_labels.append(predicted)
        gold_flags.append(example["flags"])
        predicted_flags.append(prediction["flags"])

        if set(gold) == set(predicted):
            exact += 1
        if gold and predicted and gold[0] == predicted[0]:
            primary += 1
        else:
            mistakes.append((example, predicted))

    scored = len(gold_labels)
    if not scored:
        print("every validation request failed; is the server up?", file=sys.stderr)
        return 1

    labels_report = score(gold_labels, predicted_labels, arguments.classes)
    flags_report = score(gold_flags, predicted_flags, arguments.flags)

    print()
    print(f"scored {scored} of {len(examples)} example(s); {errored} failed")
    print(f"  exact-match (label set identical):  {exact / scored:.1%}")
    print(f"  primary (most-specific label) match: {primary / scored:.1%}")
    print(f"  label micro F1: {labels_report['micro_f1']:.3f}")
    print(f"  label macro F1: {labels_report['macro_f1']:.3f}")
    if arguments.flags:
        print(f"  flag micro F1:  {flags_report['micro_f1']:.3f}")
        print(f"  flag macro F1:  {flags_report['macro_f1']:.3f}")

    print()
    print("per-class F1 (classes present in gold or predictions):")
    ranked = sorted(
        labels_report["per_name"].items(), key=lambda item: item[1]["f1"]
    )
    for name, stats in ranked:
        print(
            f"  {stats['f1']:.3f}  {name}  "
            f"(tp={stats['tp']} fp={stats['fp']} fn={stats['fn']})"
        )

    if arguments.show_mistakes and mistakes:
        print()
        print("primary-label mismatches:")
        for example, predicted in mistakes:
            record = example["record"]
            print(
                f"  {record.get('function_name')!r} "
                f"({record.get('language')}/{record.get('package')}): "
                f"gold {example['labels']} vs predicted {predicted}"
            )

    if arguments.report:
        arguments.report.write_text(
            json.dumps(
                {
                    "model": arguments.model,
                    "scored": scored,
                    "errored": errored,
                    "exact_match": exact / scored,
                    "primary_match": primary / scored,
                    "labels": labels_report,
                    "flags": flags_report,
                },
                indent=2,
            )
        )
        print(f"\nwrote report to {arguments.report}")

    return 0


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

    Two kinds of failure are told apart. A connection or HTTP error means the
    server itself could not be reached or did not answer properly -- a fault in
    the infrastructure, not the data -- and the result is flagged `infra` so the
    caller can abort a shard whose server has died. A reply that arrives but
    cannot be parsed into valid labels is a fault in that one row and carries no
    such flag.

    Args:
        record: A record as `build_record` returns it.
        taxonomy: The rendered taxonomy description.
        arguments: Parsed command-line arguments, for the endpoint and model.

    Returns:
        A dict with `labels` and `flags`, or with `error` when every retry
        failed; the error dict also carries `infra` when the *last* failure was a
        connection or HTTP error rather than an unusable reply.
    """
    prompt = describe(record)
    failure = "no valid reply"
    infra = False
    for attempt in range(RETRIES):
        if attempt:
            # Back off before retrying so a momentarily overloaded or restarting
            # server can recover, growing the wait each time.
            time.sleep(BACKOFF * 2 ** (attempt - 1))

        # The first attempt is deterministic; later ones sample, since at
        # temperature 0 an unusable reply would just be regenerated verbatim.
        temperature = 0.0 if attempt == 0 else 0.7
        try:
            content = request(
                arguments.endpoint,
                arguments.model,
                prompt,
                taxonomy,
                arguments.timeout,
                temperature,
            )
        except urllib.error.HTTPError as error:
            failure, infra = f"HTTP {error.code}: {error.reason}", True
            continue
        except (urllib.error.URLError, OSError, KeyError, ValueError) as error:
            failure, infra = str(error), True
            continue

        parsed = parse(content, arguments.classes, arguments.flags)
        if parsed:
            return parsed
        # A reply arrived but did not parse; record that, so the reported error
        # reflects the last thing that actually happened rather than an earlier
        # network error that a later, answered attempt superseded.
        failure, infra = "no valid reply", False

    result: dict[str, Any] = {"error": failure}
    if infra:
        result["infra"] = True
    return result


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

    Raises:
        ShardAborted: When connection or HTTP errors cross `--max-error-rate`,
            which usually means the server has died; the partial output is
            removed so the shard is retried rather than silently truncated.
    """
    parquet = pq.ParquetFile(source)
    codec = arguments.compression or compression(source)
    inherited = parquet.schema_arrow
    stem = os.path.basename(source).removesuffix(".parquet")

    # The `id` and `name` columns `tokenize-disassembly.py` requires. Both the
    # prompt metadata and the appended `name` column are copied from
    # `function_name`; a shard without it (fine for nixpkgs, but not guaranteed
    # in general) cannot be labeled, so say so plainly rather than dying with a
    # bare KeyError deep in the batch loop.
    if "function_name" not in inherited.names:
        sys.exit(
            f"{source}: no `function_name` column (have: "
            f"{', '.join(inherited.names)}); this pipeline labels from the "
            "symbol name and cannot run without it"
        )

    # The `id` and `name` columns `tokenize-disassembly.py` requires.
    identify = "id" not in inherited.names
    rename = "name" not in inherited.names
    fields = list(inherited) + LABEL_FIELDS + [LABEL_MODEL_FIELD]
    if identify:
        fields.append(pa.field("id", pa.large_string()))
    if rename:
        fields.append(pa.field("name", inherited.field("function_name").type))

    schema = pa.schema(fields)
    partial = destination.parent / (destination.name + ".partial")

    written = 0
    labeled = 0
    infra = 0
    try:
        with pq.ParquetWriter(partial, schema, compression=codec) as writer:
            for batch in parquet.iter_batches(batch_size=arguments.batch_size):
                rows = results(batch, taxonomy, pool, arguments)

                columns = list(batch.columns)
                for field, key in zip(LABEL_FIELDS, ("labels", "flags", "error")):
                    values = [row.get(key) for row in rows]
                    columns.append(pa.array(values, type=field.type))

                # Stamp every row with the model that labeled it, so the shard
                # carries its own provenance rather than relying on the run log.
                columns.append(
                    pa.array(
                        [arguments.model] * batch.num_rows,
                        type=LABEL_MODEL_FIELD.type,
                    )
                )

                # Numbered before unlabeled rows are dropped, so an `id` names
                # the same row of the input shard whether or not its neighbours
                # survive.
                if identify:
                    positions = range(written, written + batch.num_rows)
                    identifiers = [f"{stem}:{position}" for position in positions]
                    columns.append(pa.array(identifiers, type=pa.large_string()))
                if rename:
                    columns.append(batch.column("function_name"))

                table = pa.Table.from_arrays(columns, schema=schema)
                if arguments.drop_unlabeled:
                    table = table.filter(table["labels"].is_valid())

                writer.write_table(table)
                written += batch.num_rows
                labeled += sum(1 for row in rows if "labels" in row)
                infra += sum(1 for row in rows if row.get("infra"))

                # Bail out of a shard whose server has stopped answering before
                # its near-empty output can be renamed into place and skipped.
                if (
                    written >= ERROR_SAMPLE
                    and infra / written > arguments.max_error_rate
                ):
                    raise ShardAborted(
                        f"{infra}/{written} rows failed with connection or HTTP "
                        f"errors (> {arguments.max_error_rate:.0%}); server down?"
                    )
    except ShardAborted:
        partial.unlink(missing_ok=True)
        raise

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
    aborted = 0
    with ThreadPoolExecutor(max_workers=arguments.concurrency) as pool:
        for position in assigned:
            source = paths[position]
            destination = arguments.destination / os.path.basename(source)
            if destination.exists() and not arguments.overwrite:
                skipped += 1
                continue

            try:
                rows, found = write(source, destination, taxonomy, pool, arguments)
            except ShardAborted as error:
                aborted += 1
                print(
                    f"  {os.path.basename(source)}: ABORTED, left unwritten "
                    f"for a later run ({error})",
                    flush=True,
                )
                continue

            written += rows
            labeled += found
            print(
                f"  {os.path.basename(source)}: {rows} rows, {found} labeled",
                flush=True,
            )

    print(
        f"shard {arguments.shard}/{arguments.shards}: wrote {written} rows "
        f"({labeled} labeled) across {len(assigned) - skipped - aborted} file(s), "
        f"{skipped} already present, {aborted} aborted -> {arguments.destination}"
    )
    return 1 if aborted else 0


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
    parser.add_argument(
        "inputs",
        nargs="*",
        help="parquet file(s), directory, or glob (not used with --validate)",
    )
    parser.add_argument(
        "--destination",
        type=pathlib.Path,
        help="directory for the labeled shards (required unless --validate)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="score the labeling against the hand-labeled validation set "
        "instead of writing shards",
    )
    parser.add_argument(
        "--validation-set",
        type=pathlib.Path,
        default=HOME / "labelling_validation.json",
        help="hand-labeled functions to score against with --validate",
    )
    parser.add_argument(
        "--report",
        type=pathlib.Path,
        help="with --validate, also write the scores to this JSON file",
    )
    parser.add_argument(
        "--show-mistakes",
        action="store_true",
        help="with --validate, list every primary-label mismatch",
    )
    parser.add_argument(
        "--taxonomy", type=pathlib.Path, default=HOME / "function_taxonomy.json"
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
        "--max-error-rate",
        type=float,
        default=0.25,
        help="abort a shard (leaving it unwritten) once this fraction of its "
        "rows fail with connection or HTTP errors, e.g. when the server dies",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="rewrite shards already present"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="stop after this many shards (or validation examples)",
    )
    parser.add_argument("--shard", type=int, default=0, help="this worker's index")
    parser.add_argument("--shards", type=int, default=1, help="number of workers")

    parsed = parser.parse_args(arguments)
    if not parsed.validate:
        if not parsed.inputs:
            parser.error("inputs are required unless --validate is given")
        if parsed.destination is None:
            parser.error("--destination is required unless --validate is given")
    return parsed


def main() -> int:
    """Entry point.

    Returns:
        A process exit code.
    """
    arguments = parse_arguments()
    if arguments.validate:
        return validate(arguments)
    return run(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
