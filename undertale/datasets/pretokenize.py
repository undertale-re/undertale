import os
import uuid
from pathlib import Path

from datatrove.pipeline.readers import ParquetReader

from .base import Dataset, main
from .pipeline.formatters import ITEMPretokenizer


def adapt_dataset_from_parquet(
    self, data: dict, path: str, id_in_file: int | str
) -> dict:
    return {
        "id": data.pop("id", id_in_file),
        "text": data.pop("code"),
        "metadata": data,
    }


class Pretokenizer(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.logging_directory = f"{self.logging_directory}-{uuid.uuid4()}"

    def get_pipeline(self, input, writer, parallelism):
        steps = [
            ParquetReader(
                input,
                adapter=adapt_dataset_from_parquet,
            ),
            ITEMPretokenizer(),
        ]
        steps.extend(writer)

        return self.get_executor(
            steps,
            venv_path=os.path.join(f"{Path.home()}/.conda/envs", "undertale"),
            time="48:00:00",
            cpus_per_task=2,
            mem_per_cpu_gb=40,
            tasks=parallelism,
            job_name="pretokenize",
            partition="xeon-p8",
            sbatch_args={
                "distribution": "cyclic:cyclic",
                "chdir": Path.home(),
            },
        )


if __name__ == "__main__":
    main(Pretokenizer)
