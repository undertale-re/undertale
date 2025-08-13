from datatrove.data import Document, DocumentsPipeline
from datatrove.pipeline.base import PipelineStep


class PairwiseContrastive(PipelineStep):
    type = "P - PAIRS"
    name = "C - Constrastive"

    _requires_dependencies = []

    def __init__(self, num_samples: int, negative_multiple: float):
        """
        Arguments:
            num_samples:  number of constrastive pairs to generate
            neg_multiple: generate neg_multiple as many negative examples as positive examples

        """
        super().__init__()
        self.num_samples = num_samples
        self.negative_multiple = negative_multiple

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:
        """
        Input:
            `data` is an iterable the documents in which have a `text` field
            that contains the binary data for a single function.
            Note: It is assumed that a document also has an `equiv_class`
            field in its `metadata` dictionary. If two documents have the same
            value for `equiv_class` then they are the same function (same source,
            same program), perhaps compiled with different compilers / settings.

        Output:
            Yields new documents with pairs of docs and a specified similarity.

            Note we can't just stick the pair of docs in this doc, say, in
            `metadata["variant1"]` and `metadata["variant2"]`. That fails.
            Instead, we copy all the fields and their values for doc1 and doc2
            into `metadata`. The fields for doc1 get the "_d1" suffix" while
            the field for doc2 get "_d2". Thus, if we start with, e.g.,

            doc1.metadata['disassembly']    disassembly for func 1
            doc1.metadata['text']           binary for func 1

            doc2.metadata['disassembly']    disassembly for func 2
            doc2.metadata['text']           binary for func 2

            Then, if doc1 and doc2 are a pos or neg pair, we'll have, yielded
            by this `run`,

            doc.metadata["disassembly_d1"]
            doc.metadata["text_d1"]
            doc.metadata["disassembly_d2"]
            doc.metadata["text_d2"]
            (and other fields)

            That is, all of the original info from doc1 and doc2 are here but
            with different but distinguishable fields.

            The similarity value (0 is not similar, 1 is similar) is in the
            `metadata` field's dictionary.

        """

        import random

        from datatrove.utils.logging import logger

        if not data:
            return

        # iterate over all the fns in this shard and collect functions
        # into equivalence classes
        logger.info("Collecting equivalence classes")
        equivalence_classes = {}
        for document in data:
            with self.track_time():
                # this is a function's worth of binary data.
                ec = document.metadata["equiv_class"]
                if not (ec in equivalence_classes):
                    equivalence_classes[ec] = []
                equivalence_classes[ec].append(document)

        ec2 = {}
        # keep only equiv classes with 2 or more items
        for ec, s in equivalence_classes.items():
            if len(s) >= 2:
                ec2[ec] = s
        equivalence_classes = ec2

        self.num_samples = int(self.num_samples / world_size)

        nec = len(equivalence_classes)
        logger.info(
            f"{nec} equivalence classes in this shard. {self.num_samples} required."
        )
        if nec < self.num_samples:
            logger.info(
                "*** That's fewer than the number of samples required -- downgrading."
            )
            self.num_samples = nec

        logger.info(f"Generating {self.num_samples} pairs")

        aecl = list(equivalence_classes.keys())

        ecl = list(equivalence_classes.keys())
        random.shuffle(ecl)

        # generate this many pairs of similar and non-similar functions
        # note num_samples is across all worlds.
        for i in range(self.num_samples):

            def make_doc_pair_doc(d1, d2, sim):
                d = Document(
                    # this "document" is a pair so concat the ids for individual docs?
                    id=f"{d1.id}:{d2.id}",
                    # this text field can't really contain the pair of functions..
                    text="n/a",
                    metadata={"similarity": sim},
                )

                def copy_meta(df, dt, suff):
                    # copy metadata from df into dt,
                    # adding suffix to keys
                    for key, val in df.metadata.items():
                        dt.metadata[key + suff] = val

                copy_meta(d1, d, "_d1")
                copy_meta(d2, d, "_d2")
                return d

            p = random.random()
            if p < self.negative_multiple / (1.0 + self.negative_multiple):
                # generate a negative sample:
                # choose two different equiv classes
                # and pick a single doc from each
                ec1 = random.choice(aecl)
                ec2 = random.choice(aecl)
                logger.info(f"{i} neg ec1={ec1} ec2={ec2}")
                d1 = random.choice(equivalence_classes[ec1])
                d2 = random.choice(equivalence_classes[ec2])
                yield make_doc_pair_doc(d1, d2, 0.0)
            else:
                logger.info("generating positive sample")
                # generate a positive sample:
                # first choose an equiv class at random
                ec = ecl.pop()
                ind = list(range(len(equivalence_classes[ec])))
                random.shuffle(ind)
                # pick two variants at random from those in the clas
                d1 = equivalence_classes[ec][ind[0]]
                d2 = equivalence_classes[ec][ind[1]]
                logger.info(f"{i} pos ec={ec} {ind[0]},{ind[1]} d1={d1.id} d2={d2.id}")
                yield make_doc_pair_doc(d1, d2, 1.0)

        logger.info("Looks like we generated all the samples we wanted")
