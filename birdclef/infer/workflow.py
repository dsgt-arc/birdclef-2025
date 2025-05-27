import json
from pathlib import Path

import bioacoustics_model_zoo as bmz
import luigi
import typer
from contexttimer import Timer
from rich import print
from birdclef import is_gpu_enabled

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
    clip_step = luigi.FloatParameter(
        default=5.0,
        description="The increment in seconds between starts of consecutive clips",
    )
    num_partitions = luigi.IntParameter(
        default=200,
        description="Number of partitions to split the audio files into",
    )
    limit = luigi.IntParameter(
        default=-1,
        description="Limit the number of audio files to process",
    )
    use_subset = luigi.BoolParameter(
        default=False,
        description="If True, process only a subset of the audio files for debugging or testing.",
    )
    subset_size = luigi.IntParameter(
        default=5, description="Number of audio files to process if use_subset=True."
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

        if self.use_subset:
            print(
                f"[Subset] Processing only {len(audio_files_subset)} files in partition {self.part}"
            )
            audio_files_subset = audio_files_subset[: self.subset_size]

        if not audio_files_subset:
            print(f"No files found for partition {self.part}. Skipping.")

        # for target in self.output().values():
        #     target.makedirs()
        for key in ["embed", "predict", "timing"]:
            Path(self.output()[key].path).parent.mkdir(parents=True, exist_ok=True)

        with Timer() as t:
            results = model.embed(
                [p.as_posix() for p in audio_files_subset],
                return_preds=True,
                clip_step=self.clip_step,
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
        if self.use_subset:
            parts_to_process = [0]
        else:
            limit = self.limit if self.limit > 0 else self.num_partitions
            parts_to_process = range(limit)

        for part in parts_to_process:
            yield ProcessPartition(
                input_root=self.input_root,
                output_root=self.output_root,
                model_name=self.model_name,
                clip_step=self.clip_step,
                num_partitions=self.num_partitions,
                use_subset=self.use_subset,
                subset_size=self.subset_size,
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
    clip_step: float = 5.0,
    num_partitions: int = 200,
    use_subset: bool = False,
    subset_size: int = 5,
    limit: int = -1,
    num_workers: int = 1,
    assert_gpu: bool = False,
):
    """Process audio under a directory using parallel Luigi workers."""
    if assert_gpu and not is_gpu_enabled():
        raise RuntimeError(
            "GPU is not enabled. Please check your PyTorch or TensorFlow installation."
        )
    luigi.build(
        [
            ProcessAudio(
                input_root=input_root,
                output_root=f"{output_root}/{model_name}",
                model_name=model_name,
                clip_step=clip_step,
                num_partitions=num_partitions,
                use_subset=use_subset,
                subset_size=subset_size,
                limit=limit,
            )
        ],
        local_scheduler=True,
        workers=num_workers,
    )


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn")
    app()
