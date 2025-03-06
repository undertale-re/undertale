import abc
import copy
import logging
import os
import subprocess
import tempfile

from . import transform

logger = logging.getLogger(__name__)

COMPILE_ERROR_STRING = b"ERROR"


class Compile(transform.Map, metaclass=abc.ABCMeta):
    """The callable base class for all the Undertale compiler classes.

    This class provides the interface for defining and working with compiler transforms.
    Subclasses must implement the `command_template` property to format the compilation
    command for a specific compiler.

    Attributes:
        sourcefile (str): Name of the file where source code will be written.
        outputfile (str): Name of the file where output from the compiler will be stored.
        command_template (str): Template string of the command used to invoke a compiler.
    """

    @property
    def sourcefile(self) -> str:
        """Name of the file where source code will be written."""
        return "source"

    @property
    def outputfile(self) -> str:
        """Name of the file where output from the compiler will be stored."""
        return "output"

    @property
    @abc.abstractmethod
    def command_template(self):
        """Template string of the command used to invoke a compiler.

        It should take `sourcefile` and `outputfile` as parameters."""
        pass

    def __call__(self, sample):
        """Compiles source code."""
        source = sample["source"]

        working = tempfile.TemporaryDirectory()

        sourcefile = os.path.join(working.name, self.sourcefile)

        with open(sourcefile, "w") as f:
            f.write(source)

        outputfile = os.path.join(working.name, self.outputfile)

        process = subprocess.run(
            self.command_template.format(
                sourcefile=sourcefile, outputfile=self.outputfile
            ),
            cwd=working.name,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if process.returncode == 0:
            with open(outputfile, "rb") as f:
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


class CompileCpp(Compile):
    @property
    def sourcefile(self):
        return "source.cpp"

    @property
    def outputfile(self):
        return "output.o"

    @property
    def command_template(self):
        return "g++ -c {sourcefile} -o {outputfile}"


class CompileGo(Compile):
    """Compilation transform for Golang source with specifications.

    Arguments:
        os (str): Target operating system. For available options run `go tool dist list`.
        arch (str): Target architecture. For available options run `go tool dist list`.
        compile_flags (list): Arguments to pass to the compiler. For available options run
            `go build -gcflags --help` or [see documentation](https://pkg.go.dev/cmd/compile).
        link_flags (list): Arguments to pass to the linked. For available options run
            `go build -ldflags --help` or [see documentation](https://pkg.go.dev/cmd/link@go1.23.3).
    """

    @property
    def sourcefile(self):
        return "source.go"

    @property
    def outputfile(self):
        return "output.o"

    @property
    def command_template(self):
        return self._command_template

    def __init__(self, os=None, arch=None, compile_flags=[], link_flags=[]):
        # adds cross-compilation options
        command_extras = []
        if os or arch:
            command_extras.append("env")
            if os:
                command_extras.append("GOOS={}".format(os))
            if arch:
                command_extras.append("GOARCH={}".format(arch))

        command_extras.append("go build ")

        # adds optimization settings
        if compile_flags:
            command_extras.append('-gcflags="' + " ".join(compile_flags) + '"')
        if link_flags:
            command_extras.append('-ldflags="' + " ".join(link_flags) + '"')

        command_extras.append("-o {outputfile} {sourcefile}")

        self._command_template = " ".join(
            ["go mod init tmp && ", "goimports -w {} && ".format(self.sourcefile)]
        ) + " ".join(command_extras)

    def __call__(self, sample):
        """Overriding this method to increase chances of compiling golang functions.

        Adds supplementary logic to the source code of golang functions to increase the chances
        of a successful compilation. It assumes that functions have a missing header and imports.
        Just before compilation it appends to the beginning of each function:
          * `package ini` - used to compile as a reusable library; it will produce an object file
            rather than a standalone executable binary
          * a set of known go imports
        This update to source code is made just before compilation to avoid modifying the original
        source code.
        """
        sample_copy = copy.deepcopy(sample)

        sample_copy["source"] = (
            r"""
package ini

import (
    "io"
    "os"
    "fmt"
    "log"
    "math"
    "path"
    "sync"
    "time"
    "regexp"
    "context"
    "strings"
    "strconv"
    "os/exec"
    "net/http"
    "go/format"
    "io/ioutil"
    "math/rand"
    "encoding/csv"
    "encoding/json"
    "path/filepath"
)

"""
            + sample_copy["source"]
        )
        return super().__call__(sample_copy)


class CompileErrorsFilter(transform.Filter):
    def __call__(self, sample):
        return sample["code"] != COMPILE_ERROR_STRING


__all__ = ["CompileCpp", "CompileGo", "CompileErrorsFilter"]
