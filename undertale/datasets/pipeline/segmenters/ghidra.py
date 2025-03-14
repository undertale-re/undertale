import logging

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep

logger = logging.getLogger(__name__)


class GhidraFunctionSegmenter(PipelineStep):
    """Segments the given binaries into individual functions.

    Input:
        Whole binaries in some executable format (ELF, PE, DLL, Mach-O, etc.)

    Output:
        Yields documents for each function in the given binary. Raises
        exceptions if Ghidra auto-analysis does not work for some reason.
    """

    type = "✂️ - SEGMENTER"
    name = "🐲 Ghidra"

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:
        import os
        import tempfile

        import pyhidra
        from datatrove.data import Document

        if not data:
            return

        for document in data:
            with self.track_time():
                binary = document.text

                working = tempfile.TemporaryDirectory()
                binary = os.path.join(working.name, "binary")

                with open(binary, "wb") as f:
                    f.write(binary)

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

                        code = binary[start - base : end - base]
                        metadata = document.metadata.copy()

                        yield Document(
                            id=f"{document.id}:{start}",
                            text=code,
                            metadata=metadata,
                        )

                        self.stat_update("functions")

                self.stat_update("binaries")
