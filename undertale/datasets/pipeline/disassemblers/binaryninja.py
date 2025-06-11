from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep


class BinaryNinjaDisassembler(PipelineStep):
    type = "🔧 - DISASSEMBLER"
    name = "B - Binary Ninja"

    def __init__(self):
        super().__init__()

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:

        import binaryninja

        for document in data:
            code = document.text
            data_buffer = binaryninja.DataBuffer(code)
            bv = binaryninja.load(source=data_buffer)