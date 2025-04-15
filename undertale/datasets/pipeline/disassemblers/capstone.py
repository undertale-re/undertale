import capstone
from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep


class CapstoneDisassembler(PipelineStep):
    """Disassembles the given code with Capstone.

    Arguments:
        architecture: A Capstone architecture constant.
        mode: A Capstone mode constant.
        base: The base address to use for disassembly.

    Input:
        Raw shellcode (or compiled, individual functions).

    Output:
        Adds a field to the metadata called `disassembly` containing the
        disassembled code produced by Capstone. Does not modify the `text`
        field.
    """

    type = "🔧 - DISASSEMBLER"
    name = "🔺 Capstone"

    def __init__(
        self,
        architecture: int = capstone.CS_ARCH_X86,
        mode: int = capstone.CS_MODE_64,
        base: int = 0x1000,
    ):
        super().__init__()

        self.architecture = architecture
        self.mode = mode
        self.base = base

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:
        import capstone

        if not data:
            return

        for document in data:
            with self.track_time():
                code = document.text

                engine = capstone.Cs(self.architecture, self.mode)
                engine.detail = True

                instructions = engine.disasm(code, self.base)

                disassembly = []
                for instruction in instructions:
                    disassembly.append(f"{instruction.mnemonic} {instruction.op_str}")

                disassembly = "\n".join(disassembly)

                document.metadata["disassembly"] = disassembly

                yield document

                self.stat_update("disassembled")
