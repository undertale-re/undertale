import logging
import math
import os
import pickle
import tempfile

import networkx
import psutil
import pyhidra

from .. import transform

logger = logging.getLogger(__name__)


class GhidraDisassembleError(Exception):
    """Raised when an issue occured during this transform."""


class GhidraDisassemble(transform.Map):
    """Disassemble binary code with Ghidra.

    Arguments:
        entry: A function to determine the entry address. By default, this
            assumes the base address of whatever is passed to Ghidra. Several
            options are defined as constants on this class.
    """

    batched = True
    indices = True

    @staticmethod
    def ENTRY_IMAGE_BASE(api):
        return api.getCurrentProgram().getImageBase()

    @staticmethod
    def ENTRY_ADDRESS_ZERO(api):
        return api.getAddressFactory().getDefaultAddressSpace().getAddress(0)

    ENTRY_DEFAULT = ENTRY_IMAGE_BASE

    def __init__(self, entry=None):
        self.entry = entry or self.ENTRY_DEFAULT

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

    def __call__(self, batch, index):
        sample = {key: value[0] for key, value in batch.items()}
        code = sample["code"]

        working = tempfile.TemporaryDirectory()

        binary = os.path.join(working.name, "binary")

        with open(binary, "wb") as f:
            f.write(code)

        architecture = sample.get("architecture")

        if not architecture:
            raise GhidraDisassembleError(
                "dataset does not specify an architecture — please provide one"
            )

        if architecture == "x64":
            language = "x86:LE:64:default"
        elif architecture == "x86":
            language = "x86:LE:32:default"
        elif architecture == "arm64":
            language = "AARCH64:LE:64:v8A"
        else:
            raise GhidraDisassembleError(
                f"invalid architecture '{architecture}' for sample at row {index}; options are x64, x86, arm64"
            )

        # Sets Java max heap size to available system RAM
        if not pyhidra.launcher.jpype.isJVMStarted():
            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024**3)
            jvm_args = [f"-Xmx{math.floor(available_gb)}g"]
            launcher = pyhidra.launcher.PyhidraLauncher()
            launcher.add_vmargs(*jvm_args)

        try:
            with pyhidra.open_program(binary, language=language, analyze=False) as api:
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

                batch["disassembly"] = [disassembly]
                batch["decompilation"] = [decompilation]
                batch["control-flow-graph"] = [pickle.dumps(graph)]

                return batch
        except Exception as e:
            logger.error(
                f"failed to process sample at row {index} with error: {e} - removed from dataset"
            )

            delete_batch = {}
            for key in list(batch.keys()) + [
                "disassembly",
                "decompilation",
                "control-flow-graph",
            ]:
                delete_batch[key] = []

            return delete_batch
