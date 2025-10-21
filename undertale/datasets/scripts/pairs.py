import argparse
import os
from pathlib import Path

from datatrove.pipeline.readers import ParquetReader

from ... import logging as undertale_logging
from ..base import Dataset, build_parser
from ..pipeline.pairs import PairwiseContrastive


def adapt_dataset_from_parquet(
    self, data: dict, path: str, id_in_file: int | str
) -> dict:
    return {
        "id": data.pop("id", id_in_file),
        "text": "PAIRWISE",  # TODO needed this hardcoded value for the adapter to work
        "metadata": data,
    }


class Pairs(Dataset):
    def __init__(self, *args, num_samples: int, negative_multiple: float, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_samples = num_samples
        self.negative_multiple = negative_multiple
        print(
            f"PairwiseContrastive in PairwiseContrastive.py: num_samples={num_samples} negative_multiple={negative_multiple}"
        )

    def get_pipeline(self, input, writer, parallelism):
        steps = [
            ParquetReader(
                input,
                adapter=adapt_dataset_from_parquet,
            ),
            PairwiseContrastive(self.num_samples, self.negative_multiple),
        ]
        steps.extend(writer)

        return self.get_executor(
            steps,
            venv_path=os.path.join(f"{Path.home()}/venv", "undertale"),
            time="48:00:00",
            cpus_per_task=1,
            mem_per_cpu_gb=8,
            tasks=parallelism,
            job_name="PairWiseContrastive",
            partition="xeon-p8",
            sbatch_args={
                "distribution": "cyclic:cyclic",
            },
        )


if __name__ == "__main__":
    undertale_logging.setup_logging()

    parser = argparse.ArgumentParser(
        description="parsing utilities for PairWiseContrastive",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    build_parser(parser)

    parser.add_argument("-s", "--num_samples", required=True, help="number of samples")

    parser.add_argument(
        "-m", "--negative_multiple", required=True, help="negative multiple"
    )

    arguments = parser.parse_args()

    dataset = Pairs(
        writer=arguments.writer,
        executor=arguments.executor,
        num_samples=int(arguments.num_samples),
        negative_multiple=float(arguments.negative_multiple),
    )
    dataset.build(
        input=arguments.input,
        output=arguments.output,
        parallelism=arguments.parallelism,
    )
