import random

from datatrove.data import Document, DocumentsPipeline
from datatrove.pipeline.base import PipelineStep


class PairwiseContrastive(PipelineStep):

    def __init__(self, num_samples, negative_multiple):
        """

        Arguments:
            num_samples: number of constrastive pairs to generate
            neg_multiple: generate neg_multiple as many negative examples as positive examples

        """
        self.num_samples = num_samples
        self.negative_multiple = negative_multiple

    type = "PAIRWISE-CONTRASTIVE"
    name = "ALEX-CHRIS"

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

        # iterate over all the fns in this shard and collect functions
        # into equivalence classes
        equivalence_classes = {}
        for document in data:
            with self.track_time():
                # this is a function's worth of binary data.
                ec = document["metadata"]["equiv_class"]
                if not (ec in equivalence_classes):
                    equivalence_classes[ec] = []
                equivalence_classes[ec].append(document)

        # generate this many pairs of similar and non-similar functions
        for i in range(self.num_samples):

            def yield_pair_doc(d1, d2, sim):
                yield Document(
                    id=f"{d1.id}:{d2.id}",
                    # this text field will contain the pair of documents,
                    # each of which is a function
                    text=(d1, d2),
                    metadata={"similarity": 0.0},
                )

            p = random.random()
            if p < self.negative_multiple / (1.0 + self.negative_multiple):
                # generate a negative sample:
                # choose two different equiv classes
                # and pick a single row from each
                ec1 = random.choice(equivalence_classes)
                ec2 = random.choice(equivalence_classes)
                yield_pair_doc(random.choice(ec1), random.choice(ec2), 0.0)
            else:
                # generate a positive sample:
                # choose an equiv class that contains at least
                # 2 versions of a function
                while True:
                    ec = random.choice(equivalence_classes)
                    if len(equivalence_classes[ec]) > 1:
                        break
                # pick two versions at random from those in the
                # equiv class
                ind = list(range(len(equivalence_classes[ec])))
                random.shuffle(ind)
                yield_pair_doc(ec[ind[0]], ec[ind[1]], 1.0)
