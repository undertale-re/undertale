"""Rizin, a popular fork of Radare2.

Rizin: https://rizin.re/."""

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep


class RizinFunctionSegmentAndDisassemble(PipelineStep):
    """Segments the given binaries into individual functions.

    Input:
        Whole binaries in some executable format (ELF, PE, DLL, Mach-O, etc.).

    Output:
        Yields documents for each function in the given binary. Also
        disassemblesd each function.
    Notes:
        Rizin might also do decompilation and certainly also can give you CFG.
        Not sure exactly how to extract this info.
    """

    type = "✂️ - SEGDISASMENTER"
    name = "Z Rizin"

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:
        """"""

        import json
        import os
        import tempfile

        import rzpipe
        from datatrove.data import Document
        from datatrove.utils.logging import logger

        if not data:
            return

        for document in data:
            with self.track_time():
                code = document.text
                working = tempfile.TemporaryDirectory()
                binary = os.path.join(working.name, "binary")
                logger.info(f"working on binary={binary}")
                with open(binary, "wb") as f:
                    f.write(code)
                r = rzpipe.open(binary)
                r.cmd("aaaa")
                ij = json.loads(r.cmd("ij"))
                # this should be list of functions in this binary
                fns = json.loads(r.cmd("aflj"))
                for fn in fns:
                    fun_name = fn["name"]
                    logger.info(f"segmented fun={fun_name}")
                    if fun_name == "main" or fun_name.startswith("fcn"):
                        # this is a function actually in the elf
                        # rather than sym.XX which are lib fns etc.
                        pdfj = json.loads(r.cmd(f"pdfj @ {fun_name}"))

                        disassembly = []
                        if "ops" in pdfj.keys():
                            for i in range(len(pdfj["ops"])):
                                if "disasm" in pdfj["ops"][i].keys():
                                    disassembly.append(pdfj["ops"][i]["disasm"])
                        disassembly = "\n".join(disassembly)
                        yield Document(
                            id=f"{document.id}:{fun_name}",
                            text=code,
                            metadata={
                                "disassembly": disassembly,
                                "function_name": fun_name,
                                "architecture": ij["bin"]["arch"],
                            },
                        )

                        self.stat_update("functions")

                self.stat_update("binaries")
