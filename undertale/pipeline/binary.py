"""Binary segmentation, disassembly, and decompilation."""

from typing import List

from pandas import DataFrame, Series, concat, read_parquet
from pandera.errors import SchemaError as PanderaSchemaError

from ..exceptions import EnvironmentError as LocalEnvironmentError
from ..exceptions import SchemaError
from ..logging import get_logger
from ..schema import BinaryDataset
from ..utils import assert_path_exists, get_or_create_file

logger = get_logger(__name__)


def segment_and_disassemble(row: Series) -> DataFrame:
    import binaryninja
    from binaryninja import InstructionTextTokenType, SymbolType

    binaryninja.disable_default_log()

    view = binaryninja.load(source=row["binary"])

    logger.debug(f"segmenting and disassembling {row['id']}")

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
        disassembly: List[str] = []
        for block in blocks_by_address:
            if block.has_invalid_instructions:
                raise ValueError(
                    f"invalid instructions found in {block} of row {row['id']}"
                )

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

            annotation = False
            for line in block.disassembly_text:
                for token in line.tokens:
                    if annotation:
                        if (
                            token.type == InstructionTextTokenType.AnnotationToken
                            and token.text.strip() == "}"
                        ):
                            annotation = False
                        continue
                    match token.type:
                        # New Instruction - emit a separator.
                        case InstructionTextTokenType.AddressSeparatorToken:
                            if disassembly:
                                disassembly.append("<NEXT>")
                        # Emit token verbatim.
                        #
                        # Instruction mnemonics, registers, braces (memory access).
                        case (
                            InstructionTextTokenType.InstructionToken
                            | InstructionTextTokenType.RegisterToken
                            | InstructionTextTokenType.BraceToken
                        ):
                            disassembly.append(token.text.strip())
                        # Addresses - emit as integers.
                        #
                        # Integers, Immediate values, imports (relative),
                        # addresses, symbols.
                        case (
                            InstructionTextTokenType.IntegerToken
                            | InstructionTextTokenType.FloatingPointToken
                            | InstructionTextTokenType.PossibleAddressToken
                            | InstructionTextTokenType.ImportToken
                            | InstructionTextTokenType.CodeRelativeAddressToken
                            | InstructionTextTokenType.DataSymbolToken
                            | InstructionTextTokenType.CodeSymbolToken
                            | InstructionTextTokenType.ExternalSymbolToken
                        ):
                            disassembly.append(str(token.value))
                        # Keyword token - parsing required.
                        #
                        # Binary Ninja seems to lump together a lot of
                        # miscellaneous tokens as `KeywordTokens`. We need to
                        # handle some of them separately.
                        case InstructionTextTokenType.KeywordToken:
                            text = token.text.strip()
                            match text:
                                case "byte" | "word" | "dword" | "qword" | "xmmword":
                                    disassembly.append(text)
                                # Instruction pointer relative address.
                                case "rel":
                                    disassembly.extend(["rip", "+"])
                                case _:
                                    raise ValueError(
                                        f"unhandled keyword token: {line} ({token})"
                                    )
                        # Operation token - parsing required.
                        case InstructionTextTokenType.OperationToken:
                            text = token.text.strip()
                            match text:
                                # Arethmetic operators:
                                case "+" | "-" | "*":
                                    disassembly.append(text)
                                # Immediate value prefix - ignored.
                                case "#":
                                    pass
                                case _:
                                    raise ValueError(
                                        f"unhandled operation token: {line} ({token})"
                                    )
                        # Ignore token.
                        #
                        # Text (spacing, formatting, etc.), separators
                        # (commas), memory operator annotation, tags.
                        case (
                            InstructionTextTokenType.TextToken
                            | InstructionTextTokenType.OperandSeparatorToken
                            | InstructionTextTokenType.BeginMemoryOperandToken
                            | InstructionTextTokenType.EndMemoryOperandToken
                            | InstructionTextTokenType.TagToken
                        ):
                            pass
                        # Annotation token.
                        #
                        # Ignore all tokens until the next annotation token is reached.
                        case InstructionTextTokenType.AnnotationToken:
                            if token.text.strip().startswith("{"):
                                annotation = True
                            else:
                                raise ValueError(
                                    f"unexpected annotation token {line} ({token})"
                                )
                        case _:
                            raise ValueError(
                                f"unhandled token type: {line} ({token}:{token.type.name}))"
                            )

        function = {
            "id": row.id,
            "name": function.name,
            "binary": binary,
            "disassembly": " ".join(disassembly),
        }

        if "source" in row:
            function["source"] = row["source"]

        functions.append(function)

    logger.info(f"successfully segmented and disassembled {row['id']}")

    return DataFrame(functions)


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

    try:
        BinaryDataset.validate(frame)
    except PanderaSchemaError as e:
        logger.error("dataset does not match the expected schema")
        raise SchemaError(str(e))

    segmented = [segment_and_disassemble(r) for _, r in frame.iterrows()]
    segmented = concat(segmented)

    logger.info(
        f"successfully segmented and disassembled {len(segmented)} functions from {len(input)} binaries"
    )

    segmented.to_parquet(output)

    return output


__all__ = ["segment_and_disassemble_binary"]
