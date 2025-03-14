import logging
import os
from typing import Callable, Optional

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep

logger = logging.getLogger(__name__)


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

    type = "🔧 - Disassemble"
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

            logging.error(message)
            raise EnvironmentError(message)

    def build_control_flow_graph(self, api, entry):
        """Build the Inter-Procedural Control Flow Graph.

        Start at the specified entrypoint and slice forward to build the graph.

        Arguments:
            api: The Ghidra FlatAPI.
            entry: The entry address from which to start building the graph

        Returns:
            A graph of the IPCFG starting at the specified entry point, and a
            dictionary mapping addresses to decompiled functions for each
            function identified by Ghidra in the traversed program slice.
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

        functions = {}

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

                # If we're at the start of a new function, decompile.
                if address.getOffset() not in functions:
                    functions[address.getOffset()] = decompiler.decompile(
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

        return graph, functions

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

                    graph, functions = self.build_control_flow_graph(api, entry)

                # Sort blocks by address and store straightline disassembly.
                #
                # This just matches what the Capstone disassembler transform does for
                # now.
                disassembly = []
                for _, block in sorted(graph.nodes, key=lambda n: n[0]):
                    disassembly.append(block)
                disassembly = "\n".join(disassembly)

                # Sort functions by address and store all of their decompilation.
                decompilation = "\n".join([functions[a] for a in sorted(functions)])

                document.metadata["disassembly"] = disassembly
                document.metadata["decompilation"] = decompilation
                document.metadata["cfg"] = pickle.dumps(graph)

                yield document

                self.stat_update("disassembled")
