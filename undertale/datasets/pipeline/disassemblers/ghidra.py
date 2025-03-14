import logging
import os
from typing import Callable, Optional

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep

logger = logging.getLogger(__name__)


def build_control_flow_graph(api, entry, ipcfg=False):
    """Build the Control Flow Graph starting at the given entrypoint.

    Start at the specified entrypoint and slice forward to build the graph.

    Arguments:
        api: The Ghidra FlatAPI.
        entry: The entry address from which to start building the graph
        ipcfg: If `True`, generate an interprocedural CFG (i.e., allow the
            traversal to exit the current function and ignore function
            boundaries).

    Returns:
        A graph of the (IP)CFG starting at the specified entry point, a string
        containing the disassembled basic blocks from the CFG in address order,
        and a string containing the decompiled function(s) in address order.
    """

    import networkx
    from ghidra.app.decompiler import DecompileOptions
    from ghidra.app.decompiler.flatapi import FlatDecompilerAPI
    from ghidra.program.model.block import BasicBlockModel
    from ghidra.util.task import TaskMonitor

    program = api.getCurrentProgram()
    listing = program.getListing()

    model = BasicBlockModel(program)
    monitor = TaskMonitor.DUMMY

    decompiler = FlatDecompilerAPI(api)
    decompiler.initialize()
    decompiler.decompiler.setOptions(DecompileOptions())

    decompiled_functions = {}

    def process(block, graph=None):
        """Recursively process blocks into a graph via DFS."""

        graph = graph or networkx.Graph()

        address = block.getMinAddress()

        # Check if we're at the start of a function.
        function = listing.getFunctionAt(address)
        if function:
            # If we've reached a non-local function, skip.
            if function.isExternal() or function.isThunk():
                return graph, None

            # If we've exited the function, and we're not in IPCFG mode, skip.
            if len(decompiled_functions) > 0 and not ipcfg:
                return graph, None

            # If we're at the start of a new function, decompile.
            if address.getOffset() not in decompiled_functions:
                decompiled_functions[address.getOffset()] = decompiler.decompile(
                    function
                ).strip()

        # Disassemble.
        disassembly = []
        instructions = listing.getInstructions(block, True)
        while instructions.hasNext():
            instruction = instructions.next()
            disassembly.append(f"{instruction}")

        disassembly = "\n".join(disassembly)

        node = (address.getOffset(), disassembly)

        if graph.has_node(node):
            # Reached a repeated block - graph cycle.
            return graph, node

        graph.add_node(node)

        destinations = block.getDestinations(monitor)
        while destinations.hasNext():
            next = destinations.next().getDestinationBlock()

            graph, next = process(next, graph)

            if next is not None:
                graph.add_edge(node, next)

        return graph, node

    block = model.getCodeBlockAt(entry, monitor)

    graph, _ = process(block)

    # Sort blocks by address and store straightline disassembly.
    #
    # This just matches what the Capstone disassembler transform does for
    # now. Note that this will disregard empty space between valid basic
    # blocks.
    disassembly = []
    for _, block in sorted(graph.nodes, key=lambda n: n[0]):
        disassembly.append(block)
    disassembly = "\n".join(disassembly)

    # Sort functions by address and concatenate decompilation.
    decompilation = "\n".join(
        [decompiled_functions[a] for a in sorted(decompiled_functions)]
    )

    return graph, disassembly, decompilation


class GhidraDisassembler(PipelineStep):
    """Disassembles the given code with Ghidra.

    Arguments:
        language: A Ghidra language identifier. If not provided, Ghidra will
            attempt to auto-detect the input language - if this fails, an
            exception will be raised.
        entry: A function to determine the entry address. If not provided,
            Ghidra will attempt to auto-detect the entrypoint - if this fails,
            an exception will be raised.

    Input:
        Raw shellcode (or compiled, individual functions).

    Output:
        Adds the fields `disassembly`, `decompilation`, and `cfg` to the
        document metadata, produced by Ghidra. Does not modify the `text`
        field.
    """

    type = "🔧 - DISASSEMBLER"
    name = "🐲 Ghidra"

    @staticmethod
    def ENTRY_IMAGE_BASE(api):
        return api.getCurrentProgram().getImageBase()

    @staticmethod
    def ENTRY_ADDRESS_ZERO(api):
        return api.getAddressFactory().getDefaultAddressSpace().getAddress(0)

    def __init__(
        self,
        language: Optional[str] = None,
        entry: Optional[Callable] = None,
    ):
        super().__init__()

        self.language = language
        self.entry = entry or self.ENTRY_IMAGE_BASE

        if "GHIDRA_INSTALL_DIR" not in os.environ:
            message = "cannot find Ghidra - please set the `GHIDRA_INSTALL_DIR` environment variable"

            logger.error(message)
            raise EnvironmentError(message)

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:
        import os
        import pickle
        import tempfile

        import pyhidra

        if not data:
            return

        for document in data:
            with self.track_time():
                code = document.text

                working = tempfile.TemporaryDirectory()

                binary = os.path.join(working.name, "binary")

                with open(binary, "wb") as f:
                    f.write(code)

                with pyhidra.open_program(
                    binary, language=self.language, analyze=False
                ) as api:
                    program = api.getCurrentProgram()
                    entry = self.entry(api)

                    api.addEntryPoint(entry)
                    api.analyzeAll(program)

                    graph, disassembly, decompilation = build_control_flow_graph(
                        api, entry, ipcfg=True
                    )

                document.metadata["cfg"] = pickle.dumps(graph)
                document.metadata["disassembly"] = disassembly
                document.metadata["decompilation"] = decompilation

                yield document

                self.stat_update("disassembled")
