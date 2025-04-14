from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep

from ..disassemblers.ghidra import build_control_flow_graph


class GhidraFunctionSegmenter(PipelineStep):
    """Segments the given binaries into individual functions.

    Input:
        Whole binaries in some executable format (ELF, PE, DLL, Mach-O, etc.)

    Output:
        Yields documents for each function in the given binary. Also
        disassembles, decompiles, and generates the CFG for each function.
        Raises exceptions if Ghidra auto-analysis does not work for some
        reason.
    """

    type = "✂️ - SEGMENTER"
    name = "🐲 Ghidra"

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:
        import os
        import pickle
        import tempfile

        import pyhidra
        from datatrove.data import Document

        if not data:
            return

        for document in data:
            with self.track_time():
                code = document.text

                working = tempfile.TemporaryDirectory()
                binary = os.path.join(working.name, "binary")

                with open(binary, "wb") as f:
                    f.write(code)

                with pyhidra.open_program(binary) as api:
                    program = api.getCurrentProgram()
                    listing = program.getListing()

                    for function in listing.getFunctions(True):
                        # Skip non-local functions.
                        if function.isExternal() or function.isThunk():
                            continue

                        base = program.getAddressMap().getImageBase().getOffset()
                        body = function.getBody()
                        start = body.getMinAddress().getOffset()
                        end = body.getMaxAddress().getOffset()

                        code = code[start - base : end - base]

                        # Also disassemble, decompile, and build the CFG.
                        graph, disassembly, decompilation = build_control_flow_graph(
                            api, function.getEntryPoint(), ipcfg=False
                        )

                        metadata = document.metadata.copy()

                        metadata["cfg"] = pickle.dumps(graph)
                        metadata["disassembly"] = disassembly
                        metadata["decompilation"] = decompilation

                        yield Document(
                            id=f"{document.id}:{start}",
                            text=code,
                            metadata=metadata,
                        )

                        self.stat_update("functions")

                self.stat_update("binaries")
