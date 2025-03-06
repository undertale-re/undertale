import logging

import capstone

from .. import transform

logger = logging.getLogger(__name__)


class CapstoneDisassemble(transform.Map):
    """Disassemble binary code with Capstone.

    Arguments:
        architecture: Capstone architecture constant.
        mode: Capstone mode constant.
        base: Base address.
    """

    def __init__(
        self,
        architecture: int = capstone.CS_ARCH_X86,
        mode: int = capstone.CS_MODE_64,
        base: int = 0x1000,
    ):
        self.architecture = architecture
        self.mode = mode
        self.base = base

    def __call__(self, sample):
        code = sample["code"]

        engine = capstone.Cs(self.architecture, self.mode)
        engine.detail = True

        instructions = engine.disasm(code, self.base)

        disassembly = []
        for instruction in instructions:
            disassembly.append(f"{instruction.mnemonic} {instruction.op_str}")

        disassembly = "\n".join(disassembly)

        return {"disassembly": disassembly}
