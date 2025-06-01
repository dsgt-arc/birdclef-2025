import json
from pathlib import Path

import librosa
import luigi
import pandas as pd
import typer
from contexttimer import Timer
from opensoundscape import Audio
from opensoundscape.spectrogram import MelSpectrogram
from rich import print

app = typer.Typer()


class OptionsMixin:
    input_root = luigi.Parameter(
        description="Directory containing audio files to process",
    )
    output_root = luigi.Parameter(
        description="Directory to save the output files",
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
        output_dir = Path(self.output_root).expanduser()
        return {
            "data": luigi.LocalTarget(output_dir / f"data/part={self.part}"),
            "timing": luigi.LocalTarget(
                output_dir / f"timing/part={self.part}/part.json"
            ),
        }

    def _process(self, path, n_mfcc=20):
        """Process a single audio file and return its mel spectrogram."""
        audio = Audio.from_file(path.as_posix(), sample_rate=32000)
        spec = MelSpectrogram.from_audio(
            audio,
            n_mels=128,
            fft_size=8192,
            window_samples=8000,
            overlap_fraction=0.5,
            # dont want to double log things
            dB_scale=False,
        )
        mfccs = librosa.feature.mfcc(
            S=spec.spectrogram,
            sr=spec.audio_sample_rate,
            n_mfcc=n_mfcc,
        )
        # and now we return a pandas dataframe with the data
        df = pd.DataFrame(
            {
                "file": [path.as_posix()] * mfccs.shape[1],
                "timestamp": spec.times,
                "mfcc": [mfccs[:, i] for i in range(mfccs.shape[1])],
            }
        )
        return df

    def run(self):
        """Process a single partition of audio files."""

        input_path = Path(self.input_root).expanduser()
        audio_files = sorted(input_path.glob("**/*.ogg"))
        audio_files_subset = [
            p for i, p in enumerate(audio_files) if i % self.num_partitions == self.part
        ]

        if not audio_files_subset:
            print(f"No files found for partition {self.part}. Skipping.")
            return  # skip if no files are found

        for key in ["data", "timing"]:
            Path(self.output()[key].path).parent.mkdir(parents=True, exist_ok=True)

        processed = 0
        with Timer() as t:
            for path in audio_files_subset:
                output = Path(self.output()["data"].path) / f"{path.stem}.parquet"
                if output.exists():
                    continue
                output.parent.mkdir(parents=True, exist_ok=True)
                df = self._process(path)
                df.reset_index().to_parquet(output, index=False)
                processed += 1

        output = Path(self.output()["timing"].path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w") as f:
            json.dump(
                {
                    "part_name": f"part_{self.part:04d}",
                    "num_files": len(audio_files_subset),
                    "elapsed": t.elapsed,
                    "processed": processed,
                },
                f,
            )


class ProcessAudio(luigi.WrapperTask, OptionsMixin):
    def requires(self):
        """Define dependencies: one ProcessPartition task for each partition."""
        limit = self.limit if self.limit > 0 else self.num_partitions
        parts_to_process = range(limit)

        for part in parts_to_process:
            yield ProcessPartition(
                input_root=self.input_root,
                output_root=self.output_root,
                num_partitions=self.num_partitions,
                part=part,
            )


@app.command()
def process_audio(
    input_root: str,
    output_root: str,
    num_partitions: int = 100,
    limit: int = -1,
    num_workers: int = 1,
):
    """Process audio under a directory using parallel Luigi workers."""
    luigi.build(
        [
            ProcessAudio(
                input_root=input_root,
                output_root=output_root,
                num_partitions=num_partitions,
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
