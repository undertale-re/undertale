"""Export the per-function source code from an Undertale parquet shard to JSON.

Each row of a shard is one function from one binary. The `source` column holds
one or more path-labelled fenced blocks, e.g.

    src/wrapped/wrappedutil.c:
    ```
    #include <stdio.h>
    ...
    ```

so this script splits that column back into (path, text) pairs, tries to carve
out the definition of the row's own function, and writes one JSON record per
function.

Usage (needs pyarrow -- e.g. the `undertale` conda env):

    python export_functions.py ../datasets/part.53.parquet -o functions.json
    python export_functions.py ../datasets/*.parquet --jsonl -o functions.jsonl
    python export_functions.py ../datasets/part.53.parquet --split out/functions/
    python export_functions.py ../datasets/part.53.parquet --limit 100 --indent 2

The default `--source files` embeds each function's whole enclosing file, and
files repeat across rows: for part.53.parquet that is 1.5 GB of JSON against
4.6 MB of distinct source. Pass `--source paths` (or `none`) for bulk exports.

Records look like:

    {
      "index": 0,
      "package": "box64", "version": "0.2.6", "language": "C",
      "binary_path": "bin/box64", "flake": "...",
      "function_name": "my_forkpty",
      "source_files": [{"path": "src/wrapped/wrappedutil.c", "text": "..."}],
      "function_source": "EXPORT pid_t my_forkpty(...)\n{\n ... }",
      "function_source_path": "src/wrapped/wrappedutil.c"
    }
"""

import argparse
import base64
import glob
import json
import os
import re
import sys

import pyarrow.parquet as pq

# Columns always carried into the output, when the shard has them.
META_COLUMNS = [
    "binary_path",
    "flake",
    "package",
    "version",
    "language",
    "function_name",
]

# Start of a fenced block: a path on its own line, then ```.
BLOCK_RE = re.compile(r"^([^\s:][^\n:]*):\n```\n", re.MULTILINE)

# Suffixes the compiler appends to a symbol; the C definition uses the base name.
SUFFIX_RE = re.compile(r"\.(cold|part|isra|constprop|localalias)(\.\d+)*$")
TRAILING_NUM_RE = re.compile(r"\.\d+$")


def split_source(source):
    """Split a `source` cell into [{"path": ..., "text": ...}].

    Falls back to a single unnamed block if the cell is not fenced as expected.
    """
    if not source:
        return []
    marks = list(BLOCK_RE.finditer(source))
    if not marks:
        return [{"path": None, "text": source.strip()}]
    blocks = []
    for i, mark in enumerate(marks):
        start = mark.end()
        end = marks[i + 1].start() if i + 1 < len(marks) else len(source)
        text = source[start:end].rstrip()
        if text.endswith("```"):
            text = text[: -len("```")].rstrip()
        blocks.append({"path": mark.group(1), "text": text})
    return blocks


def base_name(function_name):
    """Strip compiler-added suffixes: `remove_node.cold.1` -> `remove_node`."""
    if not function_name:
        return function_name
    return TRAILING_NUM_RE.sub("", SUFFIX_RE.sub("", function_name))


def extract_definition(text, name):
    """Best-effort carve-out of `name`'s definition from a C file body.

    Looks for `name (` at the start of a declarator, checks that a `{` follows
    the parameter list (i.e. it is a definition and not a call or prototype),
    then brace-matches to the end. Returns None when nothing matches; macro
    generated functions (box64's `my_<callback>_<n>` bridges) legitimately have
    no textual definition.
    """
    if not name:
        return None
    for match in re.finditer(r"(?<![\w.$])" + re.escape(name) + r"\s*\(", text):
        params_end = match_delimiter(text, match.end() - 1, "(", ")")
        if params_end is None:
            continue
        body_start = skip_to_body(text, params_end + 1)
        if body_start is None:
            continue
        body_end = match_delimiter(text, body_start, "{", "}")
        if body_end is None:
            continue
        return text[line_start(text, match.start()) : body_end + 1]
    return None


def skip_to_body(text, pos):
    """Index of the `{` opening the body at/after `pos`, or None.

    Whitespace and whole preprocessor lines may sit between the parameter list
    and the body -- box64's interpreter files guard the declarator itself:

        #ifdef TEST_INTERPRETER
        uintptr_t Test66F30F(x64test_t* test, rex_t rex, uintptr_t addr)
        #else
        uintptr_t Run66F30F(x64emu_t* emu, rex_t rex, uintptr_t addr)
        #endif
        {

    A prototype ends in `;` rather than `{`, so it is rejected here.
    """
    end = len(text)
    while pos < end:
        ch = text[pos]
        if ch in " \t\r\n":
            pos += 1
        elif ch == "#":
            newline = text.find("\n", pos)
            if newline == -1:
                return None
            pos = newline + 1
        elif ch == "{":
            return pos
        else:
            return None
    return None


def line_start(text, pos):
    """Walk back to the start of the declaration (return type, storage class)."""
    start = text.rfind("\n", 0, pos) + 1
    # Pull in a preceding line when the return type sits on its own line.
    prev_end = start - 1
    if prev_end > 0:
        prev_start = text.rfind("\n", 0, prev_end) + 1
        prev = text[prev_start:prev_end].strip()
        if (
            prev
            and not prev.endswith((";", "}", "{", ")", ":", "/"))
            and not prev.startswith("#")
        ):
            return prev_start
    return start


def match_delimiter(text, open_pos, open_ch, close_ch):
    """Index of the delimiter closing the one at `open_pos`, or None.

    Skips string and character literals, and both comment styles, so braces
    inside them do not unbalance the count.
    """
    if open_pos >= len(text) or text[open_pos] != open_ch:
        return None
    depth = 0
    i = open_pos
    end = len(text)
    while i < end:
        ch = text[i]
        if ch == "\\" and i + 1 < end:
            i += 2
            continue
        if ch in "\"'":
            quote = ch
            i += 1
            while i < end:
                if text[i] == "\\":
                    i += 2
                    continue
                if text[i] == quote:
                    break
                i += 1
        elif text.startswith("//", i):
            newline = text.find("\n", i)
            i = end if newline == -1 else newline
        elif text.startswith("/*", i):
            close = text.find("*/", i + 2)
            i = end if close == -1 else close + 1
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def build_record(index, row, args):
    """Turn one parquet row (as a dict) into an output record."""
    record = {"index": index}
    for column in META_COLUMNS:
        if column in row:
            record[column] = row[column]

    blocks = split_source(row.get("source"))
    name = base_name(row.get("function_name"))

    definition = None
    definition_path = None
    for block in blocks:
        definition = extract_definition(block["text"], name)
        if definition is not None:
            definition_path = block["path"]
            break

    record["function_source"] = definition
    record["function_source_path"] = definition_path
    if args.source == "files":
        record["source_files"] = blocks
    elif args.source == "paths":
        record["source_files"] = [block["path"] for block in blocks]

    if args.include_disassembly and "disassembly" in row:
        record["disassembly"] = row["disassembly"]
    if args.include_decompilation and "decompilation" in row:
        record["decompilation"] = row["decompilation"]
    if args.include_bytes and row.get("code") is not None:
        record["code_base64"] = base64.b64encode(row["code"]).decode("ascii")
    return record


def sanitize(name, fallback="function"):
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", name or "")[:120]
    return cleaned or fallback


def resolve_inputs(patterns):
    paths = []
    for pattern in patterns:
        if os.path.isdir(pattern):
            matches = sorted(glob.glob(os.path.join(pattern, "*.parquet")))
        elif any(ch in pattern for ch in "*?["):
            matches = sorted(glob.glob(pattern))
        else:
            matches = [pattern]
        if not matches:
            sys.exit(f"no parquet files matched: {pattern}")
        paths.extend(matches)
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        sys.exit("not a file: " + ", ".join(missing))
    return paths


def iter_records(paths, args):
    """Stream records across every input shard, in file then row order."""
    index = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        available = set(parquet.schema_arrow.names)
        wanted = [c for c in META_COLUMNS + ["source"] if c in available]
        if args.include_disassembly and "disassembly" in available:
            wanted.append("disassembly")
        if args.include_decompilation and "decompilation" in available:
            wanted.append("decompilation")
        if args.include_bytes and "code" in available:
            wanted.append("code")
        if "source" not in available:
            sys.exit(f"{path}: no `source` column (has: {sorted(available)})")

        for batch in parquet.iter_batches(batch_size=args.batch_size, columns=wanted):
            for row in batch.to_pylist():
                if args.limit is not None and index >= args.limit:
                    return
                yield build_record(index, row, args)
                index += 1


def main():
    parser = argparse.ArgumentParser(
        description="Export per-function source code from Undertale parquet shards to JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("inputs", nargs="+", help="parquet file(s), directory, or glob")
    parser.add_argument(
        "-o",
        "--out",
        default="functions.json",
        help="output file (ignored with --split)",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="write newline-delimited JSON instead of one array",
    )
    parser.add_argument(
        "--split", metavar="DIR", help="write one JSON file per function into DIR"
    )
    parser.add_argument(
        "--source",
        choices=["files", "paths", "none"],
        default="files",
        help="include full source file bodies, just their paths, or neither",
    )
    parser.add_argument("--include-disassembly", action="store_true")
    parser.add_argument("--include-decompilation", action="store_true")
    parser.add_argument(
        "--include-bytes",
        action="store_true",
        help="include the raw `code` column, base64-encoded",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=None,
        help="pretty-print with this indent (array/split output only)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="stop after N functions"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="parquet rows held in memory at a time",
    )
    args = parser.parse_args()

    paths = resolve_inputs(args.inputs)
    records = iter_records(paths, args)
    written = 0
    no_definition = 0

    if args.split:
        os.makedirs(args.split, exist_ok=True)
        for record in records:
            name = sanitize(record.get("function_name"))
            out_path = os.path.join(args.split, f"{record['index']:07d}_{name}.json")
            with open(out_path, "w") as handle:
                json.dump(
                    record, handle, indent=args.indent if args.indent is not None else 1
                )
                handle.write("\n")
            written += 1
            no_definition += record["function_source"] is None
        destination = args.split
    else:
        with open(args.out, "w") as handle:
            if args.jsonl:
                for record in records:
                    handle.write(json.dumps(record) + "\n")
                    written += 1
                    no_definition += record["function_source"] is None
            else:
                handle.write("[\n" if args.indent else "[")
                for record in records:
                    if written:
                        handle.write(",\n" if args.indent else ",")
                    handle.write(json.dumps(record, indent=args.indent))
                    written += 1
                    no_definition += record["function_source"] is None
                handle.write("\n]\n" if args.indent else "]\n")
        destination = args.out

    print(f"{written} functions from {len(paths)} shard(s) -> {destination}")
    if written:
        share = 100 * no_definition / written
        print(
            f"{no_definition} ({share:.1f}%) had no extractable definition "
            "(macro-generated or defined in a file not carried in `source`)"
        )


if __name__ == "__main__":
    main()
