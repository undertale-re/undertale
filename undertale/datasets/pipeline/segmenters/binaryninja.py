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

        import binaryninja
        from binaryninja import SymbolType

        SKIP_TYPES = [
            SymbolType.ImportedFunctionSymbol,
            SymbolType.ExternalSymbol,
            SymbolType.LibraryFunctionSymbol,
            SymbolType.ImportAddressSymbol, 
            SymbolType.SymbolicFunctionSymbol
        ]
        
        SKIP_TOKENS = [
                InstructionTextTokenType.AnnotationToken,
                InstructionTextTokenType.StackVariableToken
            ]

        SYMBOL_TOKENS = [
            InstructionTextTokenType.CodeSymbolToken,
            InstructionTextTokenType.DataSymbolToken,
            InstructionTextTokenType.ExternalSymbolToken
        ]

        for document in data:
            code = document.text
            data_buffer = binaryninja.DataBuffer(code)
            bv = binaryninja.load(source=data_buffer)

            for fn in bv.functions:
                if fn.is_thunk or fn.symbol.type in SKIP_TYPES:
                    continue
                
                disassembly = []
                graph = nx.Graph()
                nodes = {}
                code = bv.read(fn.start, fn.total_bytes)
                
                for block in fn.basic_blocks:
                    block_disassembly = []
                    for line in block.disassembly_text:
                        idx, symbol_token = next(
                            ((i, token) for i, token in enumerate(line.tokens) if token.type in symbol_tokens),
                            (None, None)  # default if not found
                        )
                        if symbol_token:
                            line.tokens[idx].text = f"0x{symbol_token.value:x}"
                        disasm_str = ''.join(token.text for token in line.tokens if token.type not in skip_tokens)
                        disasm_str = ' '.join(disasm_str.strip().split())
                        disassembly.append(disasm_str)
                    block_disassembly = "\n".join(block_disassembly)
                    node = (block.start, block_disassembly)
                    graph.add_node(node)
                    nodes[block.start] = node
                
                for block in fn.basic_blocks:
                    outgoing_edges = block.outgoing_edges
                    dst_nodes = [str(edge)[str(edge).find('@')+1:-1] for edge in outgoing_edges]
                    for i in dst_nodes:
                        graph.add_edge(nodes[block.start], nodes[int(i, 16)])

                for _, block in sorted(graph.nodes, key=lambda n: n[0]):
                    disassembly.append(block)
                
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

            self.stat_update("binaries")

