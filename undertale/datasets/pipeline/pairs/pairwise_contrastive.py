import random

from datatrove.data import Document, DocumentsPipeline
from datatrove.pipeline.base import PipelineStep


class PairwiseContrastive(PipelineStep):

    type = "P - PAIRS"
    name = "C - Constrastive"

    def __init__(self, num_samples:int, negative_multiple:float):
        """
        Arguments:
            num_samples: number of constrastive pairs to generate
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
            Yields documents the `text` field of which are pairs of either
            similar or dissimilar functions for contrastive training. The
            similarity value (0 is not similar, 1 is similar) is in the
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
                ec = document["metadata"]["equiv_class"]
                if not (ec in equivalence_classes):
                    equivalence_classes[ec] = []
                equivalence_classes[ec].append(document)

        ec2 = []    
        # keep only equiv classes with 2 or more items
        for ec,s in equivalence_classes.items():
            if len(s) >= 2:
                ec2[ec] = s
        equivalence_classes = ec2
                
        nec = len(equivalence_classes)
        logger.info(f"{nec} equivalence classes in this shard")
        if nec < self.num_samples:
            logger.info("*** That's fewer than the number of samples required -- downgrading.")
            self.num_samples = nec
            
        logger.info(f"Generating {self.num_samples} pairs")        

        ecl = list(equivalence_classes.keys())
        random.shuffle(ecl)

        # generate this many pairs of similar and non-similar functions
        for i in range(self.num_samples):

            def yield_pair_doc(d1, d2, sim):
                yield Document(
                    # this "document" is a pair so concat the ids
                    id=f"{d1.id}:{d2.id}",
                    # this text field will contain the pair of documents,
                    # each of which is a function
                    #text=(d1, d2),
                    metadata={
                        "similarity": sim,
                        "variant1": d1,
                        "variant2": d2 
                    },
                )

            p = random.random()
            if p < self.negative_multiple / (1.0 + self.negative_multiple):
                # generate a negative sample:
                # choose two different equiv classes
                # and pick a single row from each
                d1 = random.choice(random.choice(equivalence_classes))
                d2 = random.choice(random.choice(equivalence_classes))
                yield_pair_doc(d1, d2, 0.0)
            else:
                # generate a positive sample:
                # first choose an equiv class at random
                ec = ecl.pop()
                # pick two variants at random from those in the clas
                ind = list(range(len(equivalence_classes[ec])))
                random.shuffle(ind)
                d1 = equivalenc_class[ec][ind[0]]
                d2 = equivalenc_class[ec][ind[1]]               
                yield_pair_doc(d1, d2, 1.0)
