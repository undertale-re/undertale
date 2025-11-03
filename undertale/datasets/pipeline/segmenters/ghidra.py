"""The Ghidra reverse engineering tool.

Ghidra: https://github.com/NationalSecurityAgency/ghidra."""

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
        """"""
        import os
        import pickle
        import tempfile

        import pyghidra
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

                with pyghidra.open_program(binary) as api:
                    program = api.getCurrentProgram()
                    listing = program.getListing()

                    for function in listing.getFunctions(True):
                        # Skip non-local functions.
                        from ghidra.program.model.block import BasicBlockModel
                        from ghidra.util.task import TaskMonitor

                        block = BasicBlockModel(program).getCodeBlockAt(
                            function.getEntryPoint(), TaskMonitor.DUMMY
                        )
                        if function.isExternal() or function.isThunk() or not block:
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
                        metadata["function_name"] = function.getName()

                        yield Document(
                            id=f"{document.id}:{start}",
                            text=code,
                            metadata=metadata,
                        )

                        self.stat_update("functions")

                self.stat_update("binaries")
