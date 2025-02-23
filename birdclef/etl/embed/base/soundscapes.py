import binascii
from pathlib import Path

import luigi
import tqdm
from birdclef.luigi import maybe_gcs_target


class BaseEmbedSoundscapesAudio(luigi.Task):
    """Embed soundscapes embeddings

    This generally works on a single batch of files at a time, and is meant to
    be processed by multiple workers in parallel.
    """

    audio_path = luigi.Parameter()
    output_path = luigi.Parameter()

    total_batches = luigi.IntParameter()
    batch_number = luigi.IntParameter()

    @property
    def resources(self):
        return {self.output().path: 1}

    def output(self):
        return maybe_gcs_target(f"{self.output_path}/{self.batch_number:03d}/_SUCCESS")

    def run(self):
        paths = sorted(Path(f"{self.audio_path}").glob("*.ogg"))
        # now only keep the audio files that belong to the same hash
        paths = [
            path
            for path in paths
            if (binascii.crc32(path.stem.encode()) % self.total_batches)
            == self.batch_number
        ]

        inference = self.get_inference()
        for path in tqdm.tqdm(paths):
            out_path = f"{self.output_path}/{self.batch_number:03d}/{path.stem}.parquet"
            if maybe_gcs_target(out_path).exists():
                continue
            df = inference.predict_df(path.parent, path.name)
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out_path, index=False)

        # write success
        with self.output().open("w") as f:
            f.write("")

    def get_inference(self):
        raise NotImplementedError()


class BaseEmbedSoundscapesAudioWorkflow(luigi.WrapperTask):
    audio_path = luigi.Parameter()
    intermediate_path = luigi.Parameter()
    output_path = luigi.Parameter()

    total_batches = luigi.IntParameter(default=200)
    limit = luigi.OptionalIntParameter(default=None)

    def get_task(self, batch_number: int) -> BaseEmbedSoundscapesAudio:
        raise NotImplementedError()

    def requires(self):
        batch_numbers = list(range(self.total_batches))
        if self.limit is not None and self.limit > 0:
            batch_numbers = batch_numbers[: self.limit]

        yield [self.get_task(batch_number) for batch_number in batch_numbers]
