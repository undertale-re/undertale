import logging
import os
import tempfile

import pyhidra

from ..transform import Map

logger = logging.getLogger(__name__)


class GhidraFunctionSegment(Map):
    """Segment whole binaries into individual functions using Ghidra.

    Note: This transform requires a known file type that Ghidra can auto parse (i.e., some executable like ELF, PE, etc.).
    """

    batched = True
    indices = True

    def __call__(self, batch, indices):
        binaries = batch.pop("binary")
        rows = zip(binaries, indices)

        batch_functions = []
        for idx, (sample, row_idx) in enumerate(rows):
            working = tempfile.TemporaryDirectory()
            binary = os.path.join(working.name, "binary")
            sample_functions = []

            with open(binary, "wb") as f:
                f.write(sample)

            with pyhidra.open_program(binary) as api:
                program = api.getCurrentProgram()
                listing = program.getListing()

                for function in listing.getFunctions(True):
                    # Skip non-local functions.
                    if function.isExternal() or function.isThunk():
                        continue

                    base = program.getAddressMap().getImageBase().getOffset()
                    body = function.getBody()
                    start = body.getMinAddress().getOffset()
                    end = body.getMaxAddress().getOffset()

                    function = {
                        "code": sample[start - base : end - base],
                        "function_name": function.getName(),
                    }

                    # Duplicate remaining fields from `batch`.
                    #
                    # This allows us to propagate any fields defined on the
                    # whole binary to each function in that binary.
                    for key in batch:
                        function[key] = batch[key][idx]

                    sample_functions.append(function)
            if len(sample_functions) > 0:
                batch_functions.extend(sample_functions)
            else:
                logger.warning(
                    f"Ghidra couldn't find local functions for sample at row {row_idx} - removed from dataset"
                )

        # Convert to the format that the Datasets library expects.
        #
        # They want a dictionary of lists, rather than a list of dictionaries.
        if len(batch_functions) == 0:
            delete_batch = {}
            for key in batch:
                delete_batch[key] = []
            delete_batch["code"] = delete_batch["function_name"] = []
            return delete_batch

        batch_functions = {
            k: [f[k] for f in batch_functions] for k in batch_functions[0]
        }

        return batch_functions


__all__ = ["GhidraFunctionSegment"]
