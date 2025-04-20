""" "Check how long each of the audio tracks are."""

import luigi
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
import librosa
from pathlib import Path


class DurationTask(luigi.Task):
    """
    Task to check the duration of the audio files under a specific directory.
    """

    input_path = luigi.Parameter()
    output_path = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.output_path)

    def get_audio_files(self):
        return Path(self.input_path).glob("**/*.ogg")

    def get_duration(self, audio_file):
        return {
            "path": audio_file.as_posix(),
            "duration": librosa.get_duration(path=audio_file),
        }

    def run(self):
        paths = list(self.get_audio_files())
        durations = []
        with Pool() as pool:
            for duration in tqdm(
                pool.imap_unordered(self.get_duration, paths),
                total=len(paths),
            ):
                durations.append(duration)
        df = pd.DataFrame(durations)
        df["path"] = df.path.apply(
            lambda x: Path(x).relative_to(self.input_path).as_posix()
        )
        output_path = Path(self.output().path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)

        # now check the output
        df = pd.read_parquet(output_path)
        print(df.head())
        print(f"total of {df.duration.sum() / 3600} hours of audio")


def duration(root: str = "~/shared/birdclef"):
    root = Path(root).expanduser().resolve()
    luigi.build(
        [
            DurationTask(
                input_path=f"{root}/raw/birdclef-2024",
                output_path=f"{root}/processed/birdclef-2024/durations.parquet",
            ),
            DurationTask(
                input_path=f"{root}/raw/birdclef-2025",
                output_path=f"{root}/processed/birdclef-2025/durations.parquet",
            ),
        ],
        local_scheduler=True,
    )
