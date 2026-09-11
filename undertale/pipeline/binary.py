"""Binary segmentation, disassembly, and decompilation."""

from typing import Dict, List

from pandas import DataFrame, Series, read_parquet

from ..exceptions import EnvironmentError as LocalEnvironmentError
from ..logging import get_logger
from ..models.tokenizer import TOKEN_NEXT
from ..schema import BinaryDataset, validate_dataset
from ..utils import assert_path_exists, get_or_create_file, write_parquet
from .disassembly import pretokenize_disassembly

logger = get_logger(__name__)


def segment_and_disassemble(
    row: Series,
) -> List[Dict[str, str | bytes]]:
    import binaryninja
    from binaryninja import SymbolType

    binaryninja.disable_default_log()

    with binaryninja.load(source=row["binary"]) as view:
        # Only x86 is supported for now.
        if view.arch.name not in ["x86", "x86_64", "aarch64"]:
            raise ValueError(f"unsupported architecture: {view.arch.name}")

        # Undefine all data variables.
        #
        # This prevents Binary Ninja from rendering things like `jmp data_var[42]`
        # and instead requires it to use real addresses.
        for address in list(view.data_vars.keys()):
            view.undefine_data_var(address, blacklist=True)

        functions = []
        for function in view.functions:
            # Exclude thunks.
            if function.is_thunk:
                continue

            # Exclude imported, external, or symbolic functions.
            if function.symbol.type in [
                SymbolType.ImportedFunctionSymbol,
                SymbolType.ExternalSymbol,
                SymbolType.ImportAddressSymbol,
                SymbolType.SymbolicFunctionSymbol,
            ]:
                continue

            # Binary Ninja does not guarantee block order.
            blocks_by_address = sorted(function.basic_blocks, key=lambda b: b.start)

            binary = b""
            skipped_reason = None
            disassembled_blocks = []
            for block in blocks_by_address:
                if block.has_invalid_instructions:
                    skipped_reason = "invalid instruction"

                    break

                # There are issues with this approach to extracting function bytes.
                #
                # It is possible that this approach does not preserve relative
                # instruction logic in the case that a function has gaps of empty
                # space between its basic blocks. I'm not sure how commonly this
                # happens in practice, but this is ultimately not very likely to
                # disassemble to *exactly* the same thing as the original.
                #
                # If/when we end up using this for something we might need to
                # consider revising this approach. For now, this is just included
                # for completeness and to match the schema requirements.
                binary += view.read(block.start, block.length)

                disassembled_blocks.append(block)

            if not skipped_reason:
                disassembly = pretokenize_disassembly(
                    disassembled_blocks, next_token=TOKEN_NEXT
                )

            if skipped_reason:
                function_disassembly = ""
                for block in function.basic_blocks:
                    for line in block.disassembly_text:
                        function_disassembly += f"0x{line.address:x}: "
                        for token in line.tokens:
                            function_disassembly += token.text
                        function_disassembly += "\n"

                message = f"failed to disassemble function, {skipped_reason} (id: {row['id']}, function: {function.name!r}@0x{function.start:x}):\n"
                message += "=" * 80 + "\n"
                message += function_disassembly
                message += "=" * 80

                logger.warning(message)

                continue

            function = {
                **row.to_dict(),
                "name": function.name,
                "binary": binary,
                "disassembly": " ".join(disassembly),
            }

            if "source" in row:
                function["source"] = row["source"]

            functions.append(function)

    return functions


def segment_and_disassemble_binary(input: str, output: str) -> str:
    """Segment and disassemble a binary code dataset.

    Note: this has not yet been tested with shellcode - only fully-formed
    binary formats. Adding support for that may come later.

    Arguments:
        input: Path to the binary dataset.
        output: Path where the segmented, disassembled binary dataset should be
            written.

    Returns:
        The path to the generated dataset.

    Raises:
        SchemaError: If the input dataset does not match
            :py:class:`BinaryDataset <undertale.schema.BinaryDataset>`.
    """

    input = assert_path_exists(input)
    output, created = get_or_create_file(output)

    if not created:
        return output

    try:
        import binaryninja  # noqa: F401
    except EnvironmentError:
        raise LocalEnvironmentError("Binary Ninja API bindings are not installed")

    logger.info(f"segmenting and disassembling binaries {input!r} to {output!r}")

    frame = read_parquet(input)
    validate_dataset(frame, BinaryDataset)

    segmented = []
    for _, row in frame.iterrows():
        logger.info(f"segmenting and disassembling {row['id']}")

        functions = segment_and_disassemble(row)

        segmented.extend(functions)

    segmented = DataFrame(segmented)

    logger.info(
        f"successfully segmented and disassembled {len(segmented)} functions from {len(frame)} binaries"
    )

    write_parquet(segmented, output)

    return output


__all__ = ["segment_and_disassemble_binary"]
