import logging
import os
import subprocess
import tempfile

from . import transform

logger = logging.getLogger(__name__)

COMPILE_ERROR_STRING = b"ERROR"


class Compile(transform.Map):
    def __call__(self, sample):
        source = sample["source"]

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

            code = COMPILE_ERROR_STRING

        return {"code": code}


class CompileErrorsFilter(transform.Filter):
    def __call__(self, sample):
        return sample["code"] != COMPILE_ERROR_STRING


__all__ = ["Compile", "CompileErrorsFilter"]
