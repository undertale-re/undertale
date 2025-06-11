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
                
                code = bv.read(fn.start, fn.total_bytes)

                sorted_blocks = sorted(func.basic_blocks, key=lambda b: b.start)
                
                for block in sorted_blocks:
                    disassembly = []
                    for line in block.disassembly_text:
                        idx, symbol_token = next(
                            ((i, token) for i, token in enumerate(line.tokens) if token.type in symbol_tokens),
                            (None, None)  # default if not found
                        )
                        if symbol_token:
                            line.tokens[idx].text = f"0x{symbol_token.value:x}"
                            # breakpoint()
                        disasm_str = ''.join(token.text for token in line.tokens if token.type not in skip_tokens)
                        disasm_str = ' '.join(disasm_str.strip().split())
                        disassembly.append(disasm_str)
                        # print(disasm_str)
                    disassembly = "\n".join(disassembly)
                    print(disassembly)
                
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

