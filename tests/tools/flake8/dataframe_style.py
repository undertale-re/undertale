import ast


class DataFrameToParquetChecker:
    """This checks for direct usage of ``DataFrame.to_parquet()``.

    We have a helper method implemented at ``utils.write_parquet()`` that uses
    common defaults.

    Note:
        This could have false positives - it currently flags any call to any
        object's method if the method is named `to_parquet`. Doing this in a
        way that is fully complete and sound would require some serious AST
        processing.
    """

    name = "dataframe-to-parquet"
    version = "1.0.0"

    def __init__(self, tree):
        self.tree = tree

    def run(self):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "to_parquet":
                    yield (
                        node.lineno,
                        node.col_offset,
                        "UT001: direct usage of DataFrame.to_parquet() is forbidden; use utils.write_parquet() instead.",
                        type(self),
                    )
