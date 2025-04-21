import json
from pathlib import Path

import bioacoustics_model_zoo as bmz
import luigi
import typer
from contexttimer import Timer
from rich import print
from tqdm import tqdm

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
    num_workers = luigi.IntParameter(
        default=1,
        description="Number of workers to use for processing",
    )


class ProcessAudio(luigi.Task, OptionsMixin):
    def output(self):
        return luigi.LocalTarget(f"{self.output_root}/parts/_SUCCESS")

    def run(self):
        """Process audio files using the specified model.

        We process the audio so we get both the predictions and the embeddings.
        """
        model = bmz.list_models()[self.model_name]
        audio_files = sorted(Path(self.input_root).expanduser().glob("**/*.ogg"))

        for part in tqdm(list(range(self.num_partitions))):
            if self.limit > 0 and part >= self.limit:
                break
            # determine the subset of audio files to process for idempotent processing
            audio_files_subset = [
                p for i, p in enumerate(audio_files) if i % self.num_partitions == part
            ]

            # generate the output paths
            part_name = f"part_{part:04d}"
            output_paths = [
                Path(self.output_root).expanduser()
                / f"parts/{method_name}/{part_name}.parquet"
                for method_name in ["embed", "predict"]
            ]
            for output_path in output_paths:
                output_path.parent.mkdir(parents=True, exist_ok=True)
            if all([p.exists() for p in output_paths]):
                print(f"Skipping {part_name} as it already exists.")
                continue

            # process the actual files
            with Timer() as t:
                results = model.embed(
                    audio_files_subset,
                    return_preds=True,
                    num_workers=self.num_workers,
                )
            for path, df in zip(output_paths, results):
                df.reset_index().to_parquet(path, index=False)
            # write out the timing information
            path = (
                Path(self.output_root).expanduser() / f"parts/timing/{part_name}.json"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w") as f:
                json.dump(
                    {
                        "part_name": part_name,
                        "num_files": len(audio_files_subset),
                        "elapsed": t.elapsed,
                    },
                    f,
                )
        with self.output().open("w") as f:
            f.write("")
        print(f"Finished processing {len(audio_files)} files.")


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
    num_workers: int = 0,
):
    """Process audio under a directory."""
    luigi.build(
        [
            ProcessAudio(
                input_root=input_root,
                output_root=f"{output_root}/{model_name}",
                model_name=model_name,
                num_partitions=num_partitions,
                limit=limit,
                num_workers=num_workers,
            )
        ]
    )


if __name__ == "__main__":
    app()
