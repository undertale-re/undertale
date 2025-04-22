import json
import os
import tempfile
import logging

import r2pipe
from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep

from capstone import *

md = Cs(CS_ARCH_X86, CS_MODE_64)

def cdisas(code):
    global md
    n = len(code)
    d = ""
    for i in md.disasm(code, 0x0):
        if i.address > n:
            break
        d = d + "0x%x:\t%s\t%s\n" %(i.address, i.mnemonic, i.op_str)
    return d

logger = logging.getLogger(__name__)

class RadareDisassembler(PipelineStep):

    type = "🔧 - DISASSEMBLER"
    name = "R - Radare"

    def __init__(self):
        super().__init__()
        self.code_max = 65536
        self.debug = False

    def disas_buf(self, buf):
        # buf is a function
        n = len(buf)
        self.r.cmd("s 0")
        # write the fn bytes into radare's "file"
        self.r.cmd("s 0")
        self.r.cmd("wx " + (" ".join([f"{i:02x}" for i in buf])))
        self.r.cmd("aa")
        # we are choosing to believe the function bounds. That is, all
        # of the code handed us *is* part of the function
        jd = self.r.cmd(f"pDj {n}")
        # trying to maintain all NOPS in the buffer *after* done with
        # a function, but efficiently
        self.r.cmd("wx " +  "0x90" * n)
        try:
            d = json.loads(jd)
            return "\n".join([x["disasm"] for x in d])
        except:
            return None

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:
        if not data:
            return

        logger.info(f"beginning r2 disassembly ")


        self.code_max = 65536
        self.r = r2pipe.open(f"malloc://{self.code_max}", flags=['-2'])
        # we are going to work hard to maintain this buffer with all NOPs
        # which makes disassembly maybe nicer for gaps.        
        self.r.cmd("wx " +  "0x90" * self.code_max)

        ii = 0
        num_too_big = 0
        for document in data:

            with self.track_time():            
    
                ii +=1

                code = document.text

                if len(code) > self.code_max:
                    num_too_big += 1
                    continue

                disassembly = self.disas_buf(code)

                if disassembly is None:
                    continue
                
                document.metadata["disassembly"] = disassembly

                if self.debug:
                    # compare r2 disassembly with capstone for sanity check
                    cd = cdisas(code)
                    dl1 = disassembly.split('\n')
                    dl2 = cd.split('\n')
                    l1 = len(dl1)
                    l2 = len(dl2)
                    print(f"l1={l1:x} l2={l2:x}")
                    for i in range(max(l1, l2)):
                        if i < l1:
                            print(f"{dl1[i]:40}", end="")
                        else:
                            x="..."
                            print(f"{x:40}", end="")
                        if i < l2:
                            print(f"{dl2[i]:40}", end="")
                        else:
                            x="..."
                            print(f"{x:40}", end="")
                        print(" ")

                yield document
                self.stat_update("disassembled")

        logger.info(f"FYI: {num_too_big} of {ii} skipped bc too big")
