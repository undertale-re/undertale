"""Library to Instrument Executable Formats (LIEF).

LIEF: https://lief.re/
"""

import io

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep


class LIEFFunctionSegmenter(PipelineStep):
    type = "✂️ - SEGMENTER"
    name = "🪓 LIEF"

    def demangle(self, func_name):
        import cpp_demangle
        import itanium_demangler
        import rust_demangler
        from datatrove.utils.logging import logger

        if func_name.startswith("_R"):
            try:
                # v0 style rust demangled
                new_name = rust_demangler.demangle(func_name)
                if new_name == "None":
                    logger.debug(f"rust parse failure: {func_name}")
                    raise Exception()
                self.stat_update("demangle.success")
                return new_name
            except Exception:
                self.stat_update("demangle.failed.rust_parse")
                self.stat_update("demangle.failed")
                return func_name
        elif func_name.startswith("_Z"):
            try:
                new_name = cpp_demangle.demangle(func_name)
                if new_name == "None":
                    logger.debug(f"cpp_demangle parse failure: {func_name}")
                    self.stat_update("demangle.failed.cpp_demangle")
                    raise Exception()
                self.stat_update("demangle.success")
                return new_name
            except Exception:
                try:
                    # old style rust mangled
                    new_name = rust_demangler.demangle(func_name)
                    if new_name == "None":
                        logger.debug(f"rust parse failure: {func_name}")
                        self.stat_update("demangle.failed.rust_parse")
                        raise Exception()
                    self.stat_update("demangle.success")
                    return new_name
                except Exception:
                    try:
                        new_name = str(itanium_demangler.parse(func_name))
                        if new_name == "None":
                            logger.debug(f"itanium parse failure: {func_name}")
                            self.stat_update("demangle.failed.itanium_parse")
                            raise Exception()
                        self.stat_update("demangle.success")
                        return new_name
                    except:
                        self.stat_update("demangle.failed")
                        return func_name
        else:
            # not mangled
            self.stat_update("demangle.not_mangled")
            return func_name

    def run(self, docPipeline: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        """"""

        import lief
        import unix_ar
        from datatrove.data import Document
        from datatrove.utils.logging import logger

        def parse_binary(rank, idx, sample, doc, subpath=None):
            doc_id = f"func-{rank}-{idx}/{subpath}" if subpath else f"func-{rank}-{idx}"
            fname = f"{doc['filename']}/{subpath}" if subpath else doc["filename"]
            if subpath:
                self.stat_update("binaries-nested")
            num_funcs = 0
            logger.debug(f"segmenting {fname}")
            sample_bytes_io = io.BufferedRandom(io.BytesIO(sample))
            binary = lief.parse(sample_bytes_io)
            if not binary:
                if sample[:7] == b"!<arch>":
                    try:
                        ar_bytes_io = io.BufferedRandom(io.BytesIO(sample))
                        arfile = unix_ar.open(ar_bytes_io)
                    except Exception as e:
                        self.stat_update("failure.ar_parse")
                        logger.warning(f"error decoding ar file {fname}: {e}")
                        return
                    for objfile in arfile.infolist():
                        if objfile.name == b"/":
                            continue
                        try:
                            obj_data = arfile.open(
                                str(objfile.name.decode())
                            ).getvalue()
                            parse_binary(
                                rank, idx, obj_data, doc, subpath=objfile.name.decode()
                            )
                        except Exception as e:
                            self.stat_update("failure.ar_read")
                            logger.warning(
                                f"error parsing static library {fname}/{objfile.name.decode()} {e}"
                            )
                            continue
                    return
                else:
                    self.stat_update("failure.ar")
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
                    self.stat_update("symbols.empty")
                    continue

                self.stat_update("symbols")
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
                    self.stat_update("symbol.bad_type")
                    pass

                # If the type isn’t conclusive, try to locate the section containing the symbol
                # and verify that it is executable.
                section = binary.section_from_virtual_address(symbol.value)
                if section and is_section_executable(section):
                    is_func = True
                else:
                    self.stat_update("failed.non_exec_section")

                # If we aren’t confident this symbol is a function, skip it.
                if not is_func:
                    self.stat_update("failed.not_func")
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
                    self.stat_update("functions")
                    self.stat_update("function_size", value=symbol.size)
                    logger.debug(
                        f" binary {fname} - successfully cut {symbol.section.name}:{symbol.name} ({len(content)})"
                    )
                    num_funcs += 1
                    func_name = self.demangle(symbol.name)
                    yield Document(
                        id=doc_id,
                        text=func_name,
                        metadata={
                            "code": bytes(content),
                            "function_name": func_name,
                            "filename": fname,
                            "path": doc["path"],
                            "package": doc["package"],
                            "flake": doc["properties"]["flake"],
                            "language": doc["properties"]["language"],
                        },
                    )
                except Exception as e:
                    self.stat_update("failure.carve")
                    logger.warning(
                        f"Failed to carve function content {e=} {len(content)=}"
                    )
                    continue
            self.stat_update("functions_per_binary", value=num_funcs)
            logger.info(f" binary ({rank}):{fname} completed")

        for idx, doc in enumerate(docPipeline):
            self.stat_update("binaries")
            with self.track_time():
                yield from parse_binary(rank, idx, doc.text, doc.metadata)
