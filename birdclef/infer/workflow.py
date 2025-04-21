import json
from pathlib import Path

import bioacoustics_model_zoo as bmz
import luigi
import typer
from contexttimer import Timer
from rich import print

app = typer.Typer()


class OptionsMixin:
    input_root = luigi.Parameter(
        description="Directory containing audio files to process",
    )
    output_root = luigi.Parameter(
        description="Directory to save the output files",
    )
    model_name = luigi.ChoiceParameter(
        choices=list(bmz.list_models().keys()),
        default="BirdNET",
        description="Model to use for processing audio",
    )
    num_partitions = luigi.IntParameter(
        default=200,
        description="Number of partitions to split the audio files into",
    )
    limit = luigi.IntParameter(
        default=-1,
        description="Limit the number of audio files to process",
    )


class ProcessPartition(luigi.Task, OptionsMixin):
    part = luigi.IntParameter(description="Partition number to process")

    def output(self):
        part_name = f"part_{self.part:04d}"
        output_dir = Path(self.output_root).expanduser() / "parts"
        return {
            "embed": luigi.LocalTarget(output_dir / f"embed/{part_name}.parquet"),
            "predict": luigi.LocalTarget(output_dir / f"predict/{part_name}.parquet"),
            "timing": luigi.LocalTarget(output_dir / f"timing/{part_name}.json"),
        }

    def run(self):
        """Process a single partition of audio files."""
        model = bmz.list_models()[self.model_name]()
        audio_files = sorted(Path(self.input_root).expanduser().glob("**/*.ogg"))

        # Determine the subset of audio files for this partition
        audio_files_subset = [
            p for i, p in enumerate(audio_files) if i % self.num_partitions == self.part
        ]

        if not audio_files_subset:
            print(f"No files found for partition {self.part}. Skipping.")
            # Create empty outputs if no files
            for target in self.output().values():
                target.makedirs()
                if target.path.endswith(".json"):
                    with target.open("w") as f:
                        json.dump(
                            {
                                "part_name": f"part_{self.part:04d}",
                                "num_files": 0,
                                "elapsed": 0,
                            },
                            f,
                        )
                else:  # parquet
                    import pandas as pd

                    pd.DataFrame().to_parquet(target.path)
            return

        for target in self.output().values():
            target.makedirs()

        with Timer() as t:
            results = model.embed(
                [p.as_posix() for p in audio_files_subset],
                return_preds=True,
            )

        embed_df, predict_df = results
        embed_df.reset_index().to_parquet(self.output()["embed"].path, index=False)
        predict_df.reset_index().to_parquet(self.output()["predict"].path, index=False)

        with self.output()["timing"].open("w") as f:
            json.dump(
                {
                    "part_name": f"part_{self.part:04d}",
                    "num_files": len(audio_files_subset),
                    "elapsed": t.elapsed,
                },
                f,
            )


class ProcessAudio(luigi.Task, OptionsMixin):
    def requires(self):
        """Define dependencies: one ProcessPartition task for each partition."""
        num_parts_to_process = self.limit if self.limit > 0 else self.num_partitions
        for part in range(num_parts_to_process):
            yield ProcessPartition(
                input_root=self.input_root,
                output_root=self.output_root,
                model_name=self.model_name,
                num_partitions=self.num_partitions,
                part=part,
            )

    def output(self):
        """Output is a success flag indicating all partitions are done."""
        return luigi.LocalTarget(f"{self.output_root}/_SUCCESS")

    def run(self):
        """Create the success flag file once all dependencies are met."""
        # Logic moved to ProcessPartition. This task just aggregates.
        print(f"All partitions processed for {self.model_name}.")
        with self.output().open("w") as f:
            f.write("")


@app.command()
def list_models():
    """List available models."""
    models = bmz.list_models()
    print(models)


@app.command()
def process_audio(
    input_root: str,
    output_root: str,
    model_name: str = "BirdNET",
    num_partitions: int = 200,
    limit: int = -1,
    num_workers: int = 1,
):
    """Process audio under a directory using parallel Luigi workers."""
    luigi.build(
        [
            ProcessAudio(
                input_root=input_root,
                output_root=f"{output_root}/{model_name}",
                model_name=model_name,
                num_partitions=num_partitions,
                limit=limit,
            )
        ],
        local_scheduler=True,
        workers=num_workers,
    )


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn")
    app()
