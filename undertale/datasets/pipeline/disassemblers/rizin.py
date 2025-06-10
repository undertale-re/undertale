from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep


class RizinDisassembler(PipelineStep):
    type = "🔧 - DISASSEMBLER"
    name = "R - Rizin"

    _requires_dependencies = [
        "rzpipe",
    ]

    def __init__(self):
        super().__init__()

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:

        import json
        import pickle

        import networkx as nx
        import rzpipe
        from datatrove.utils.logging import logger

        def disas_buf(buf):
            self.r.cmd("s 0")
            # resize the "file" radare is working on to fit the fn
            self.r.cmd(f"r {len(buf)}")
            # write the fn bytes into radare's "file"
            self.r.cmd("wx " + (" ".join([f"{i:02x}" for i in buf])))
            # light analysis which gets fns (presumably it also
            # does more than just linear disassembly since it
            # says it finds functions
            self.r.cmd("aa")
            # pdf is "print disassembly of function" j means json
            pdf = self.r.cmd("pdfj")
            try:
                pdf_dict = json.loads(pdf)
                return pdf_dict
            except:
                return {}

        def extract_cfg(fn_name):
            nodes = {}
            graph_json = self.r.cmd(f"agf json @ {fn_name}")
            cfg = json.loads(graph_json)

            graph = nx.DiGraph()
            for node in cfg["nodes"]:
                offset = node["offset"]
                disassembly = node["body"]
                node = (offset, disassembly)
                nodes[offset] = node
                graph.add_node(node)

            for node in cfg["nodes"]:
                offset = node["offset"]
                for dst_id in node["out_nodes"]:
                    dst_offset = cfg["nodes"][dst_id]["offset"]
                    graph.add_edge(nodes[offset], dst_offset)

            return graph

        if not data:
            return

        logger.info("beginning rz disassembly")

        code_max = 65536
        self.r = rzpipe.open(f"malloc://{code_max}", flags=["-2"])

        ii = 0
        num_too_big = 0
        for document in data:
            with self.track_time():
                ii += 1

                code = document.text

                if len(code) > code_max:
                    num_too_big += 1
                    continue

                d = disas_buf(code)
                fn_name = document.metadata["function_name"]
                graph = extract_cfg(fn_name)

                disassembly = []
                if "ops" in d.keys():
                    for i in range(len(d["ops"])):
                        if "disasm" in d["ops"][i].keys():
                            disassembly.append(d["ops"][i]["disasm"])

                disassembly = "\n".join(disassembly)

                document.metadata["disassembly"] = disassembly
                document.metadata["cfg"] = pickle.dumps(graph)

                yield document
                self.stat_update("disassembled")

        if num_too_big:
            logger.warning(f"{num_too_big} of {ii} skipped because they were too large")
