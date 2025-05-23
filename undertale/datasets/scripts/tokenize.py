import argparse
import os
from pathlib import Path

from datatrove.pipeline.readers import ParquetReader

from ... import logging as undertale_logging
from ..base import Dataset, build_parser
from ..pipeline.formatters import ITEMTokenizer


def adapt_dataset_from_parquet(
    self, data: dict, path: str, id_in_file: int | str
) -> dict:
    return {
        "id": data.pop("id", id_in_file),
        "text": data.pop("code"),
        "metadata": data,
    }


class Tokenizer(Dataset):
    def __init__(self, *args, tokenizer: str, **kwargs):
        super().__init__(*args, **kwargs)

        self.tokenizer = tokenizer

    def get_pipeline(self, input, writer, parallelism):
        steps = [
            ParquetReader(
                input,
                adapter=adapt_dataset_from_parquet,
            ),
            ITEMTokenizer(self.tokenizer),
        ]
        steps.extend(writer)

        return self.get_executor(
            steps,
            venv_path=os.path.join(f"{Path.home()}/.conda/envs", "undertale"),
            time="48:00:00",
            cpus_per_task=1,
            mem_per_cpu_gb=8,
            tasks=parallelism,
            job_name="tokenize",
            partition="xeon-p8",
            sbatch_args={
                "distribution": "cyclic:cyclic",
            },
        )


if __name__ == "__main__":
    undertale_logging.setup_logging()

    parser = argparse.ArgumentParser(
        description="parsing utilities for Tokenizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    build_parser(parser)

    parser.add_argument(
        "-t", "--tokenizer", required=True, help="path to trained tokenizer file"
    )

    arguments = parser.parse_args()

    dataset = Tokenizer(
        writer=arguments.writer,
        executor=arguments.executor,
        tokenizer=arguments.tokenizer,
    )
    dataset.build(
        input=arguments.input,
        output=arguments.output,
        parallelism=arguments.parallelism,
    )
