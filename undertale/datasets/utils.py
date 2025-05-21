import importlib


def from_specifier(specifier: str):
    """Fetch a dataset from a specifier string.

    This both identifies the dataset class and loads it by calling
    `Dataset.fetch()`.

    Arguments:
        specifier: The module path and dataset class name to load (format:
            `{module.path}:{DatasetClass}`)

    Returns:
        The requested dataset.

    Raises:
        ValueError: If there are issues loading the dataset from the given
            specifier.
    """

    try:
        module_name, class_name = specifier.split(":")
    except ValueError:
        raise ValueError(
            f"malformed dataset specifier:{specifier} (format: `{{module.path}}:{{DatasetClass}}`)"
        )

    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ValueError(f"could not import module {module_name!r}: - {e}")

    if not hasattr(module, class_name):
        raise ValueError(
            f"module {module_name!r} contains no class named {class_name!r}"
        )

    dataset_class = getattr(module, class_name)

    try:
        dataset = dataset_class().fetch()
    except Exception as e:
        raise ValueError(f"failed to load {class_name}: {e}")

    return dataset


__all__ = ["from_specifier"]
