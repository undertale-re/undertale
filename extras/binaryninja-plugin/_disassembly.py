"""Convert Binary Ninja disassembly tokens into the pretokenized instruction
stream expected by the Undertale tokenizer (mnemonics, registers, and
immediates space-separated; punctuation and formatting tokens dropped).

This module has no dependencies beyond Binary Ninja's own API, this makes it
safe to incorporate directly into contexts that cannot depend on the rest of
Undertale's (much heavier) dependency set. As an example, the Binary Ninja
plugin at extras/binaryninja-plugin/_disassembly.py keeps a byte-for-byte copy
of this file for exactly that reason. If you change this file, update that copy
too (tests/unit.py checks that they match).
"""

from typing import List


def pretokenize_disassembly(blocks, next_token: str = "[NEXT]") -> List[str]:
    """Convert basic blocks' disassembly text into pretokenized tokens.

    Walks each block's disassembly text in order and classifies every token,
    dropping punctuation and formatting tokens (commas, memory operand
    braces' annotations, etc.) so the result is a flat sequence of
    mnemonics, registers, operators, and immediates - one token per element,
    ready to be joined with a single space per element.

    Arguments:
        blocks: Binary Ninja basic blocks, in the order their instructions
            should appear.
        next_token: The token to insert between instructions. Defaults to [NEXT]
            and must match the tokenizer's special token for instruction boundaries
            (see :py:data:`undertale.models.tokenizer.TOKEN_NEXT`).

    Returns:
        The pretokenized disassembly tokens.
    """

    from binaryninja import InstructionTextTokenType

    disassembly: List[str] = []
    for block in blocks:
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
                            disassembly.append(next_token)
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
                            # Memory reference size specifiers.
                            case (
                                "byte"
                                | "word"
                                | "dword"
                                | "qword"
                                | "tword"
                                | "xmmword"
                                | "ymmword"
                                | "zmmword"
                            ):
                                disassembly.append(text)
                            # Instruction pointer relative address.
                            case "rel":
                                disassembly.append(text)
                            case _:
                                raise ValueError(
                                    f"unhandled keyword token: {line} ({token})"
                                )
                    # Operation token - parsing required.
                    case InstructionTextTokenType.OperationToken:
                        text = token.text.strip()
                        match text:
                            # Arithmetic operators.
                            case "+" | "-" | "*":
                                disassembly.append(text)
                            # Immediate value prefix - ignored.
                            case "#":
                                pass
                            # Colon operator is somewhat complex.
                            case ":":
                                # Segment register offset syntax (x86).
                                if disassembly[-1] in [
                                    "cs",
                                    "ds",
                                    "es",
                                    "ss",
                                    "fs",
                                    "gs",
                                ]:
                                    disassembly.append("+")
                                else:
                                    raise ValueError(
                                        f"unhandled ':' operator: {line} ({token})"
                                    )
                            # Special case: x86 rip-relative call.
                            #
                            # This can sometimes appear in the form of
                            # `call $+5` which really just means `call` the
                            # next instruction.
                            case "$+5":
                                disassembly.extend(["rel", "5"])
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
                        | InstructionTextTokenType.GotoLabelToken
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

    return disassembly


__all__ = ["pretokenize_disassembly"]
