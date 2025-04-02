import json

import rzpipe

from .. import transform


class RizinDisassembleError(Exception):
    """Raised when an issue occured during this transform."""


class RizinDisassemble(transform.Map):
    def __init__(self):
        self.r = rzpipe.open("malloc://4096")
        # self.r.cmd("e io.cache=true")

    def disas_buf(self, buf):
        # self.r.cmd("o-")
        # self.r.cmd("s 0")
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

    def __call__(self, sample):
        code = sample["code"]

        d = self.disas_buf(code)

        disassembly = []
        if "ops" in d.keys():
            for i in range(len(d["ops"])):
                disassembly.append(d["ops"][i]["disasm"])

        disassembly = "\n".join(disassembly)

        return {"disassembly": disassembly}
