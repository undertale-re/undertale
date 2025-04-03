import json
import os
import tempfile

import r2pipe
from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep


class RadareDisassembler(PipelineStep):
    def disas_buf(self, r, buf):
        # r.cmd("s 0")
        # resize the "file" radare is working on to fit the fn
        r.cmd(f"r {len(buf)}")
        # write the fn bytes into radare's "file"
        r.cmd("wx " + (" ".join([f"{i:02x}" for i in buf])))
        # light analysis which gets fns (presumably it also
        # does more than just linear disassembly since it
        # says it finds functions
        r.cmd("aa")
        # pdf is "print disassembly of function" j means json
        pdf = r.cmd("pdfj")
        try:
            pdf_dict = json.loads(pdf)
            return pdf_dict
        except:
            return {}

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:
        if not data:
            return

        for document in data:
            with self.track_time():
                code = document.text

                working = tempfile.TemporaryDirectory()

                binary = os.path.join(working.name, "binary")

                with open(binary, "wb") as f:
                    f.write(code)

                r = r2pipe.open(binary)

                d = self.disas_buf(r, code)

                disassembly = []
                if "ops" in d.keys():
                    for i in range(len(d["ops"])):
                        disassembly.append(d["ops"][i]["disasm"])

                disassembly = "\n".join(disassembly)
                document.metadata["disassembly"] = disassembly

                yield document
                self.stat_update("disassembled")
