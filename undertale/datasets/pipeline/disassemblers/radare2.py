import r2pipe
from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep


class RadareDisassembler(PipelineStep):
    type = "🔧 - DISASSEMBLER"
    name = "R - Radare"

    def __init__(self):
        super().__init__()

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:

        import json
        from datatrove.utils.logging import logger

        import r2pipe
        from datatrove.data import DocumentsPipeline
        from datatrove.pipeline.base import PipelineStep

        def disas_buf(self, buf):
            self.r.cmd("s 0")
            # resize the "file" radare is working on to fit the fn
            self.r.cmd(f"r {len(buf)}")
            # write the fn bytes into radare's "file"
            self.r.cmd("wx " + (" ".join([f"{i:02x}" for i in buf])))
            # light analysis which gets fns (presumably it also
            # does more than just linear disassembly since it
            # says it finds functions
            self.r.cmd("aa")
            # pdf is "print disassembly of function" j means json
            pdf = self.r.cmd("pdfj")
            try:
                pdf_dict = json.loads(pdf)
                return pdf_dict
            except:
                return {}
        if not data:
            return

        logger.info("beginning r2 disassembly")

        code_max = 65536
        self.r = r2pipe.open(f"malloc://{code_max}", flags=["-2"])

        ii = 0
        num_too_big = 0
        for document in data:
            with self.track_time():
                ii += 1

                code = document.text
                # logger.info(f"ii={ii} -- {len(code)} bytes -- {num_too_big} skipped bc too big")

                if len(code) > code_max:
                    num_too_big += 1
                    continue

                d = self.disas_buf(code)

                disassembly = []
                if "ops" in d.keys():
                    for i in range(len(d["ops"])):
                        disassembly.append(d["ops"][i]["disasm"])

                disassembly = "\n".join(disassembly)

                document.metadata["disassembly"] = disassembly

                yield document
                self.stat_update("disassembled")

        logger.info(f"FYI: {num_too_big} of {ii} skipped bc too big")
