from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep


class RizinFunctionSegmenter(PipelineStep):
    """Segments the given binaries into individual functions.

    Input:
        Whole binaries in some executable format (ELF, PE, DLL, Mach-O, etc.)

    Output:
        Yields documents for each function in the given binary. Also
        disassemblesd each function.  
    Notes:
        Rizin might also do decompilation and certainly also can give you CFG.  
        Not sure exactly how to extract this info.
    """

    type = "✂️ - SEGMENTER"
    name = "Z Rizin"

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:

        import json
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
                r = rzpipe.open(binary)
                r.cmd("aaaa")
                ij = json.loads(r.cmd("ij"))
                # this should be list of functions in this binary
                fns = json.loads(r.cmd("aflj"))                
                for fn in fns:
                    n = fn["name"]
                    if n=="main" or n.startswith("fcn"):
                        # this is a function actually in the elf
                        # rather than sym.XX which are lib fns etc.
                        pdfj = r.cmd(f"pdfj @ {n}")
                        disassembly = []
                        if "ops" in d.keys():
                            for i in range(len(d["ops"])):
                                if "disasm" in d["ops"][i].keys():
                                    disassembly.append(d["ops"][i]["disasm"])

                        disassembly = "\n".join(disassembly)

                        document.metadata["disassembly"] = disassembly
                        document.metadata["function_name"] = n
                        
                        yield Document(
                            id=f"{document.id}:{n}",
                            architecture=ij["bin"]["arch"],
                            text=code,                            
                            metadata=metadata,
                        )

                        self.stat_update("functions")

                self.stat_update("binaries")

