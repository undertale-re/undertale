import abc
import logging
import typing

from datasets import fingerprint

logger = logging.getLogger(__name__)


class Transform(metaclass=abc.ABCMeta):
    """A transformation that can be applied a dataset.

    This could represent some sort of mutation or augmentation of dataset
    samples, or some labeling strategy.
    """

    @abc.abstractmethod
    def __call__(self, sample: typing.Dict[str, typing.Any]):
        """Transform a single sample from the dataset.

        The Datasets library will handle parallelizing this across the whole
        dataset.

        Arguments:
            sample: A single sample from the dataset, in dictionary form.

        Returns:
            Whatever value is expected by the `apply` method (e.g., for
            `.filter()` this would be a boolean).
        """

        pass

    @abc.abstractmethod
    def apply(self, dataset, processes=None):
        """Apply this to the given dataset.

        Arguments:
            dataset: The dataset to transform.
            processes: The number of parallel processes to use.

        Returns:
            A transformed dataset.
        """

        pass


class Map(Transform):
    """A sample-wise dataset transform."""

    batched: bool = False
    """Control batched map behavior.

    If `True` then batches of samples will be passed to `__call__()` for this
    transform. This also enabled yielding multiple samples per row in a
    dataset (because batches are returned).
    """

    batch_size: int = 1
    """The number of samples in each batch."""

    indices: bool = False
    """Control dataset index behavior.

    If `True` then the `__call__` method for this transform should take an
    extra argument `index` (or `indices` if `batched`) which will be the
    dataset index (or a list of indices if `batched`) of the sample(s) being
    processed.
    """

    def apply(self, dataset, processes=None):
        logging.info(
            f"applying {self.__class__.__name__} to {dataset.__class__.__name__}"
        )
        logging.debug(
            f"applying {self.__class__.__name__} (hash: {fingerprint.Hasher.hash(self)}) to {dataset.__class__.__name__} (fingerprint: {dataset._fingerprint})"
        )

        processed = dataset.map(
            self,
            num_proc=processes,
            batched=self.batched,
            batch_size=self.batch_size,
            with_indices=self.indices,
        )

        # Make sure we return a Dataset of this type.
        #
        # By default `datasets.Dataset` functions will return a Dataset
        # instance, not an instance of our subclass. If we want access to all
        # of our class an instance methods, we need to adjust the class of the
        # returned object.
        processed.__class__ = dataset.__class__

        logging.debug(
            f"{self.__class__.__name__}({dataset.__class__.__name__}) (fingerprint: {dataset._fingerprint})"
        )

        return processed


class Filter(Transform):
    """A filtering dataset transform."""

    def apply(self, dataset, processes=None):
        logging.info(
            f"applying {self.__class__.__name__} to {dataset.__class__.__name__}"
        )

        processed = dataset.filter(self, num_proc=processes)

        # Make sure we return a Dataset of this type.
        #
        # See above for details on why this is necessary.
        processed.__class__ = dataset.__class__

        return processed
