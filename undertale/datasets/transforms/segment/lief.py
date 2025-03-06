import io
import logging

import cpp_demangle
import itanium_demangler
import lief
import rust_demangler
import unix_ar

from ..transform import Map

logger = logging.getLogger(__name__)


def demangle(func_name):
    if func_name.startswith("_R"):
        try:
            # v0 style rust demangled
            new_name = rust_demangler.demangle(func_name)
            if new_name == "None":
                logger.debug(f"rust parse failure: {func_name}")
                raise Exception()
            return new_name
        except Exception:
            return func_name
    elif func_name.startswith("_Z"):
        try:
            new_name = cpp_demangle.demangle(func_name)
            if new_name == "None":
                logger.debug(f"cpp_demangle parse failure: {func_name}")
                raise Exception()
            return new_name
        except Exception:
            try:
                # old style rust mangled
                new_name = rust_demangler.demangle(func_name)
                if new_name == "None":
                    logger.debug(f"rust parse failure: {func_name}")
                    raise Exception()
                return new_name
            except Exception:
                try:
                    new_name = str(itanium_demangler.parse(func_name))
                    if new_name == "None":
                        logger.debug(f"itanium parse failure: {func_name}")
                        raise Exception()
                    return new_name
                except:
                    return func_name
    else:
        # not mangled
        return func_name


class LIEFFunctionSegment(Map):
    """Segment whole binaries into individual functions using LIEF.

    Note: This transform requires binaries to be valid binary files with symbol data
    """

    batched = True
    indices = True

    def __call__(self, batch, indices):
        binaries = batch.pop("binary")
        functions = []

        def parse_binary(idx, sample, subpath=None):
            dataset_index = indices[idx]
            fname = (
                f"{batch['filename'][idx]}/{subpath}"
                if subpath
                else batch["filename"][idx]
            )
            logger.debug(f"segmenting {fname}")
            sample_bytes_io = io.BufferedRandom(io.BytesIO(sample))
            binary = lief.parse(sample_bytes_io)
            if not binary:
                if sample[:7] == b"!<arch>":
                    try:
                        ar_bytes_io = io.BufferedRandom(io.BytesIO(sample))
                        arfile = unix_ar.open(ar_bytes_io)
                    except Exception as e:
                        logger.warn(f"error decoding ar file {fname}: {e}")
                        return
                    for objfile in arfile.infolist():
                        if objfile.name == b"/":
                            continue
                        try:
                            obj_data = arfile.open(
                                str(objfile.name.decode())
                            ).getvalue()
                            parse_binary(idx, obj_data, subpath=objfile.name.decode())
                        except Exception as e:
                            logger.warn(
                                f"error parsing static library {fname}/{objfile.name.decode()} {e}"
                            )
                            continue
                    return
                else:
                    logger.warning(f"error parsing symbols from {fname}")
                    return

            # Helper: Given a section, determine if it is likely executable.
            def is_section_executable(section):
                fmt = binary.format
                if fmt == lief.Binary.FORMATS.ELF:
                    # For ELF, check if the section flags include EXECINSTR.
                    return lief.ELF.Section.FLAGS.EXECINSTR in section.flags_list
                elif fmt == lief.Binary.FORMATS.PE:
                    # For PE, check if the section characteristics include MEM_EXECUTE.
                    return section.has_characteristic(
                        lief.PE.Section.CHARACTERISTICS.MEM_EXECUTE
                    )
                elif fmt == lief.Binary.FORMATS.MACHO:
                    # For Mach-O, a common executable section is "__text".
                    # (There is no direct “executable” flag in LIEF for Mach-O, so we use a heuristic.)
                    return section.name == "__text"
                else:
                    return False

            # Iterate over all symbols in the binary.
            for symbol in binary.symbols:
                # Skip symbols without a name or with zero size.
                if not symbol.name or symbol.size == 0:
                    logger.debug(f" binary {fname} - skipping symbol {symbol}")
                    continue

                logger.debug(f" binary {fname} - symbol: {symbol.name}")

                is_func = False

                # First, if available, check the symbol type.
                try:
                    # For ELF binaries, LIEF provides a symbol type enum.
                    if (
                        binary.format == lief.Binary.FORMATS.ELF
                        and symbol.type == lief.ELF.Symbol.TYPE.FUNC
                    ):
                        is_func = True
                except Exception:
                    # Not all formats or symbols provide a reliable type.
                    pass

                # If the type isn’t conclusive, try to locate the section containing the symbol
                # and verify that it is executable.
                section = binary.section_from_virtual_address(symbol.value)
                if section and is_section_executable(section):
                    is_func = True

                # If we aren’t confident this symbol is a function, skip it.
                if not is_func:
                    continue

                # Attempt to extract the bytes of the function using the symbol’s virtual address and size.
                try:
                    # get_content_from_virtual_address returns a list of integers.
                    content = binary.get_content_from_virtual_address(
                        symbol.value, symbol.size
                    )
                    if (len(content) == 0) and (subpath):
                        start = symbol.section.file_offset + symbol.value
                        content = sample[start : start + symbol.size]
                    logger.debug(
                        f" binary {fname} - successfully cut {symbol.section.name}:{symbol.name} ({len(content)})"
                    )
                    function = {
                        "code": bytes(content),
                        "function_name": demangle(symbol.name),
                    }
                    for key in batch:
                        if (key == "filename") and (subpath is not None):
                            function[key] = fname
                        else:
                            function[key] = batch[key][idx]

                    functions.append(function)
                except Exception as e:
                    logger.warn(
                        f"Failed to carve function content {e=} {len(content)=}"
                    )
                    continue
            logger.info(
                f" binary ({dataset_index}):{fname} completed, parsed {len(functions)} functions"
            )

        for idx, sample in enumerate(binaries):
            parse_binary(idx, sample)

        # Convert to the format that the Datasets library expects.
        #
        # They want a dictionary of lists, rather than a list of dictionaries.
        if len(functions):
            functions = {k: [f[k] for f in functions] for k in functions[0]}
        else:
            functions = {"code": [], "function_name": []}
            for key in batch:
                if key == "binary":
                    continue
                functions[key] = []

        return functions


__all__ = ["LIEFFunctionSegment"]
