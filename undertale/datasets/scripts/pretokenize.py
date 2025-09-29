import os
from pathlib import Path

from datatrove.pipeline.readers import ParquetReader

from ..base import Dataset, main
from ..pipeline.formatters import ITEMPretokenizer


def adapt_dataset_from_parquet(
    self, data: dict, path: str, id_in_file: int | str
) -> dict:
    return {
        "id": data.pop("id", id_in_file),
        # "text": data.pop("code"),
        "text": data.pop("binary"),
        "metadata": data,
    }


class Pretokenizer(Dataset):
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
            # venv_path=os.path.join(f"{Path.home()}/.conda/envs", "undertale"),
            venv_path=os.path.join(f"{Path.home()}/venv", "undertale"),
            time="48:00:00",
            cpus_per_task=1,
            mem_per_cpu_gb=8,
            tasks=parallelism,
            job_name="pretokenize",
            partition="xeon-p8",
            sbatch_args={
                "distribution": "cyclic:cyclic",
            },
        )


if __name__ == "__main__":
    main(Pretokenizer)
