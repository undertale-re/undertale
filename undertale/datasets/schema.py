import inspect
import logging

import datasets

logger = logging.getLogger(__name__)


class InvalidSchemaError(Exception):
    """Raised when a dataset does not match the schema."""


class Schema:
    """A base schema class for a particular type of dataset.

    This defines the included fields on a given dataset.
    """

    @classmethod
    def get_members(cls):
        """Get members of this class.

        Returns:
            A list of non-class, non-function members of this class as a tuple
            of name and value.
        """

        for name, member in inspect.getmembers(cls):
            if name.startswith("__"):
                continue

            if inspect.isroutine(member):
                continue

            if inspect.isclass(member) and name == "Optional":
                continue

            yield name, member

    @classmethod
    def to_features(cls):
        """Produce a set of required features.

        Returns:
            A Features object in Datasets library form.
        """

        features = {}
        for name, member in cls.get_members():
            if inspect.isclass(member) and issubclass(member, Schema):
                member = member.to_features()

            features[name] = member

        return datasets.Features(**features)

    @classmethod
    def validate(cls, dataset):
        """Validate that a dataset matches this schema.

        Raises:
            InvalidSchemaError: if required fields are missing or unknown
                fields are included.
        """

        def validate_schema(schema, features, required=True):
            for name, member in schema.get_members():
                if name not in features:
                    if not required:
                        continue

                    raise InvalidSchemaError(
                        f"{dataset.__class__.__name__} is missing required feature {name!r} for schema {schema.__name__}"
                    )

                feature = features.pop(name)

                if inspect.isclass(member) and issubclass(member, Schema):
                    validate_schema(member, feature, required=required)
                elif feature != member:
                    raise InvalidSchemaError(
                        f"{dataset.__class__.__name__} feature {name!r} is the wrong type for schema {schema.__name__}: found {feature!r}, expected {member!r}"
                    )

        features = dataset.features.copy()
        validate_schema(cls, features)

        logger.info(f"dataset {dataset} matches {cls.__name__} requirements 🎉")

        if hasattr(cls, "Optional"):
            validate_schema(cls.Optional, features, required=False)

        if features:
            raise InvalidSchemaError(
                f"{dataset.__class__.__name__} has unsupported features for schema {cls.__name__}: {', '.join(features)}"
            )


class WholeBinary(Schema):
    """A dataset of entire binaries.

    This dataset requires further processing before it is useful for
    training, but this schema might still be useful.
    """

    binary = datasets.Value("binary")
    """The entire binary as bytes."""

    class Optional(Schema):
        source = datasets.Value("string")
        """The source code used to build the entire binary.

        This should be a serialized JSON object mapping file paths relative to
        the root of a project directory to their content. Ideally it should be
        possible to reconstruct an entire source tree from this field.

        Example:
            {
                'hello.cpp': 'void main() { printf("hello world\n"); }',
                'assets/data.csv': 'name,description,comment\nfoo,bar,baz',
            }
        """

        architecture = datasets.Value("string")
        """The architecture this binary was compiled on."""

        compiler = datasets.Value("string")
        """The compiler."""


class Function(Schema):
    """A dataset of individual functional pieces of code.

    Note:
        The name `Function` is a bit of a misnomer - this is really just any
        subgraph of the Inter-Procedural Control Flow Graph (IPCFG). While it
        will often just be a single function, it doesn't necessarily have to
        be.
    """

    code = datasets.Value("binary")
    """The executable bytes of this function."""

    disassembly = datasets.Value("string")
    """Instructions disassembled from the `code` field.

    Instructions should be newline separated, lowercase, in intel syntax, and
    appear in address order.

    This disassembly should exclude unreachable instructions if possible.
    """

    class Optional(Schema):
        id = datasets.Value("string")
        """An identifier relevant to this dataset."""

        source = datasets.Value("string")
        """The source code for this function, as a single string."""

        architecture = datasets.Value("string")
        """The architecture this binary was compiled on."""

        compiler = datasets.Value("string")
        """The compiler."""

        function_name = datasets.Value("string")
        """The name of this function."""


class PairwiseContrasting(Schema):
    """A pairwise contrastive learning dataset.

    Contains pairs of samples mapped to ground-truth similarity values for
    pairwise contrastive loss.
    """

    first = Function
    """The first sample."""

    second = Function
    """The second sample."""

    similarity = datasets.Value("float")


class TripletContrasting(Schema):
    """A triplet loss contrastive learning dataset.

    Contains an anchor and a positive and negative sample for triplet
    contrastive loss.
    """

    anchor = Function
    """The anchor sample."""

    positive = Function
    """Another sample with a high degree of similarity."""

    negative = Function
    """Another sample with a low degree of similarity."""


class SummarizedFunction(Function):
    """Function with a functional summary.

    This dataset can be used for multi-modal fine-tuning for a full
    summarization pipeline.
    """

    summary = datasets.Value("string")
