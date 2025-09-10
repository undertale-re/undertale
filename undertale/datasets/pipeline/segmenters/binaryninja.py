"""The BinaryNinja reverse engineering tool.

Requires a license.

BinaryNinja: https://binary.ninja/."""

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep


class BinaryNinjaFunctionSegmenter(PipelineStep):

    type = "✂️ - SEGMENTER"
    name = "B - Binary Ninja"

    def __init__(self):
        super().__init__()

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:
        """"""

        import pickle
        import re

        import binaryninja
        import networkx as nx
        from binaryninja import SymbolType
        from binaryninja.enums import InstructionTextTokenType
        from datatrove.data import Document

        def remove_braces(text):
            # Matches ' {' followed by any characters (non-greedy) until the next '}'
            return re.sub(r" \{.*?\}", "", text)

        SKIP_TYPES = [
            SymbolType.ImportedFunctionSymbol,
            SymbolType.ExternalSymbol,
            SymbolType.LibraryFunctionSymbol,
            SymbolType.ImportAddressSymbol,
            SymbolType.SymbolicFunctionSymbol,
        ]

        SKIP_TOKENS = [
            InstructionTextTokenType.StackVariableToken,
            InstructionTextTokenType.TagToken,
        ]

        SYMBOL_TOKENS = [
            InstructionTextTokenType.CodeSymbolToken,
            InstructionTextTokenType.DataSymbolToken,
            InstructionTextTokenType.ExternalSymbolToken,
        ]

        for document in data:
            code = document.text
            data_buffer = binaryninja.DataBuffer(code)
            bv = binaryninja.load(source=data_buffer)

            for fn in bv.functions:
                if (
                    fn.is_thunk
                    or fn.symbol.type in SKIP_TYPES
                    or fn.name.startswith("_Z")
                ):
                    continue
                fn_name = fn.name

                disassembly = []
                graph = nx.Graph()
                nodes = {}
                code = bv.read(fn.start, fn.total_bytes)

                for block in fn.basic_blocks:
                    block_disassembly = []
                    for line in block.disassembly_text:
                        idx, symbol_token = next(
                            (
                                (i, token)
                                for i, token in enumerate(line.tokens)
                                if token.type in SYMBOL_TOKENS
                            ),
                            (None, None),  # default if not found
                        )
                        if symbol_token:
                            line.tokens[idx].text = f"0x{symbol_token.value:x}"
                        disasm_str = "".join(
                            token.text
                            for token in line.tokens
                            if token.type not in SKIP_TOKENS
                        )
                        disasm_str = " ".join(disasm_str.strip().split())
                        if "Does" in disasm_str:  # { Does not return }
                            continue
                        if any("{" in token.text for token in line.tokens):
                            disasm_str = remove_braces(disasm_str)
                        if "sub_0" in disasm_str:
                            idx = next(
                                (
                                    i
                                    for i, token in enumerate(line.tokens)
                                    if token.text == "sub_0"
                                ),
                                -1,
                            )
                            disasm_str = disasm_str.replace(
                                "sub_0", str(line.tokens[idx].value)
                            )
                        if "retn" in disasm_str:
                            disasm_str = disasm_str[: disasm_str.find("n")]
                        if disasm_str != "":
                            block_disassembly.append(disasm_str)
                    block_disassembly = "\n".join(block_disassembly)
                    node = (block.start, block_disassembly)
                    graph.add_node(node)
                    nodes[block.start] = node

                for block in fn.basic_blocks:
                    outgoing_edges = block.outgoing_edges
                    dst_nodes = [
                        str(edge)[str(edge).find("@") + 1 : -1]
                        for edge in outgoing_edges
                    ]
                    for i in dst_nodes:
                        graph.add_edge(nodes[block.start], nodes[int(i, 16)])

                for _, block in sorted(graph.nodes, key=lambda n: n[0]):
                    disassembly.append(block)
                disassembly = "\n".join(disassembly)

                decompilation = []
                for block in fn.hlil:
                    for instr in block:
                        decompilation.append(str(instr))

                decompilation = "\n".join(decompilation)

                metadata = document.metadata.copy()

                metadata["cfg"] = pickle.dumps(graph)
                metadata["disassembly"] = disassembly
                metadata["function_name"] = fn_name
                metadata["decompilation"] = decompilation

                yield Document(
                    id=f"{document.id}:{fn.start}",
                    text=code,
                    metadata=metadata,
                )

                self.stat_update("functions")

            self.stat_update("binaries")
