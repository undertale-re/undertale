"""The BinaryNinja reverse engineering tool.

Requires a license.

BinaryNinja: https://binary.ninja/
"""

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep


class BinaryNinjaDisassembler(PipelineStep):
    type = "🔧 - DISASSEMBLER"
    name = "B - Binary Ninja"

    def __init__(self):
        super().__init__()

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:
        """."""

        import pickle
        import re

        import binaryninja
        import networkx as nx
        from binaryninja.architecture import Architecture
        from binaryninja.enums import InstructionTextTokenType
        from datatrove.data import Document

        def remove_braces(text):
            # Matches ' {' followed by any characters (non-greedy) until the next '}'
            return re.sub(r" \{.*?\}", "", text)

        SKIP_TOKENS = [
            InstructionTextTokenType.StackVariableToken,
            InstructionTextTokenType.TagToken,
        ]

        architectures = {"x86": "x86", "x64": "x86_64", "arm64": "aarch64"}

        for document in data:
            code = document.text
            base_addr = 0
            bv = binaryninja.BinaryView.new(code)
            bv.arch = Architecture[architectures[document.metadata["architecture"]]]
            bv.platform = bv.arch.standalone_platform

            bv.add_entry_point(base_addr)
            bv.create_user_function(base_addr)
            bv.update_analysis_and_wait()

            fn = bv.get_function_at(base_addr)
            graph = nx.Graph()
            nodes = {}

            for block in fn.basic_blocks:
                block_disassembly = []
                for line in block.disassembly_text:
                    disasm_str = "".join(
                        token.text
                        for token in line.tokens
                        if token.type not in SKIP_TOKENS
                    )
                    disasm_str = " ".join(disasm_str.strip().split())
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
                    block_disassembly.append(disasm_str)
                block_disassembly = "\n".join(block_disassembly)
                node = (block.start, block_disassembly)
                graph.add_node(node)
                nodes[block.start] = node

            for block in fn.basic_blocks:
                outgoing_edges = block.outgoing_edges
                dst_nodes = [
                    str(edge)[str(edge).find("@") + 1 : -1] for edge in outgoing_edges
                ]
                for i in dst_nodes:
                    graph.add_edge(nodes[block.start], nodes[int(i, 16)])

            disassembly = []
            for block in sorted(fn.basic_blocks, key=lambda b: b.start):
                for line in block.disassembly_text:
                    disasm_str = "".join(
                        token.text
                        for token in line.tokens
                        if token.type not in SKIP_TOKENS
                    )
                    disasm_str = " ".join(disasm_str.strip().split())
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
                    disassembly.append(disasm_str)
            disassembly = "\n".join(disassembly)

            decompilation = []
            if fn.hlil is not None:
                for block in fn.hlil:
                    for instr in block:
                        decompilation.append(str(instr))

            decompilation = "\n".join(decompilation)

            metadata = document.metadata.copy()

            metadata["cfg"] = pickle.dumps(graph)
            metadata["disassembly"] = disassembly
            metadata["decompilation"] = decompilation

            yield Document(
                id=f"{document.id}:{fn.start}",
                text=code,
                metadata=metadata,
            )

            self.stat_update("functions")
