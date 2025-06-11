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
            SymbolType.SymbolicFunctionSymbol]

        for document in data:
            code = document.text
            data_buffer = binaryninja.DataBuffer(code)
            bv = binaryninja.load(source=data_buffer)

            for fn in bv.functions:
                if fn.is_thunk or fn.symbol.type in SKIP_TYPES:
                    continue
                
                code = bv.read(fn.start, fn.total_bytes)
                
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

