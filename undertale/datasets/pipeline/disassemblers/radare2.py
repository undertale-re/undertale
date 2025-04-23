from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep


class RadareDisassembler(PipelineStep):

    type = "🔧 - DISASSEMBLER"
    name = "R - Radare"

    def __init__(self):
        super().__init__()
        self.debug = False

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:

        import json

        import capstone as cs
        import r2pipe
        from datatrove.utils.logging import logger

        def disas_buf_capstone(code):
            md = cs.Cs(cs.CS_ARCH_X86, cs.CS_MODE_64)
            n = len(code)
            d = ""
            for i in md.disasm(code, 0x0):
                if i.address > n:
                    break
                d = d + "0x%x:\t%s\t%s\n" % (i.address, i.mnemonic, i.op_str)
            return d

        def disas_buf(self, buf):
            # buf is a function
            n = len(buf)
            # write the fn bytes into radare's "file"
            self.r.cmd("wx " + (" ".join([f"{i:02x}" for i in buf])))
            self.r.cmd("s 0")
            self.r.cmd("aa")
            # we are choosing to believe the function bounds. That is, all
            # of the code handed us *is* part of the function. This means
            # we just linearly disassemble.
            jd = self.r.cmd(f"pDj {n}")
            # trying to maintain all NOPS in the buffer *after* done with
            # a function, but efficiently
            self.r.cmd("wx " + "0x90" * n)
            try:
                d = json.loads(jd)
                return "\n".join([x["disasm"] for x in d])
            except:
                return None

        logger.info("beginning r2 disassembly ")

        code_max = 65536
        self.r = r2pipe.open(f"malloc://{code_max}", flags=["-2"])
        # we are going to work hard to maintain this buffer with all NOPs
        # which makes disassembly maybe nicer when there are gaps.
        self.r.cmd("wx " + "0x90" * code_max)

        ii = 0
        num_too_big = 0
        for document in data:
            with self.track_time():
                ii += 1
                code = document.text
                if len(code) > code_max:
                    num_too_big += 1
                    continue
                disassembly = self.disas_buf(code)
                if disassembly is None:
                    continue
                document.metadata["disassembly"] = disassembly
                # this is for debugging but maybe keep it pls?
                if self.debug:
                    # compare r2 disassembly with capstone for sanity check
                    cd = disas_buf_capstone(code)
                    dl1 = disassembly.split("\n")
                    dl2 = cd.split("\n")
                    l1 = len(dl1)
                    l2 = len(dl2)
                    logger.info(f"l1={l1:x} l2={l2:x}")
                    for i in range(max(l1, l2)):
                        if i < l1:
                            logger.info(f"{dl1[i]:40}", end="")
                        else:
                            x = "..."
                            logger.info(f"{x:40}", end="")
                        if i < l2:
                            logger.info(f"{dl2[i]:40}", end="")
                        else:
                            x = "..."
                            logger.info(f"{x:40}", end="")
                        logger.info(" ")
                yield document
                self.stat_update("disassembled")

        logger.info(f"FYI: {num_too_big} of {ii} skipped bc too big")
