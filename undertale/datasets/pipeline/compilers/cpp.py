"""Compile C or C++ via ``gcc``/``g++``."""

import logging

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep

logger = logging.getLogger(__name__)


class CppCompiler(PipelineStep):
    """Compiles the given C/C++ code.

    Input:
        C or C++ source code that can be written to a single file and compiled.

    Output:
        Replaces the current `text` field with the raw bytes of the compiled
        code and stores the original source in a metadata field called
        ``source``.

    Discards any samples that fail to compile and logs the compilation errors.
    """

    type = "🔨 - COMPILER"
    name = "🟥 C++ Compiler"

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:
        """"""
        import os
        import subprocess
        import tempfile

        from datatrove.data import Document

        if not data:
            return

        for document in data:
            with self.track_time():
                source = document.text

                working = tempfile.TemporaryDirectory()

                sourcefile = os.path.join(working.name, "source.cpp")

                with open(sourcefile, "w") as f:
                    f.write(source)

                objectfile = os.path.join(working.name, "source.o")

                process = subprocess.run(
                    f"g++ -c {sourcefile} -o {objectfile}",
                    cwd=working.name,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                if process.returncode == 0:
                    with open(objectfile, "rb") as f:
                        code = f.read()

                    metadata = document.metadata.copy()
                    metadata["source"] = document.text

                    yield Document(id=document.id, text=code, metadata=metadata)

                    self.stat_update("succeeded")
                else:
                    message = "failed to compile source:\n"
                    message += "=" * 80 + "\n"
                    message += source.strip() + "\n"
                    message += "-" * 36 + " stdout " + "-" * 36 + "\n"
                    message += process.stdout.decode().strip() + "\n"
                    message += "-" * 36 + " stderr " + "-" * 36 + "\n"
                    message += process.stderr.decode().strip() + "\n"
                    message += "=" * 80

                    logger.warning(message)

                    self.stat_update("failed")
