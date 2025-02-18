import binascii
from pathlib import Path

import luigi
import tqdm


from birdclef.luigi import maybe_gcs_target
from birdclef.spark import spark_resource


class BaseEmbedSoundscapesAudio(luigi.Task):
    """Embed soundscapes embeddings"""

    audio_path = luigi.Parameter()
    metadata_path = luigi.Parameter()
    output_path = luigi.Parameter()

    total_batches = luigi.IntParameter(default=100)
    batch_number = luigi.IntParameter()

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
            df.to_parquet(out_path, index=False)

        # write success
        with self.output().open("w") as f:
            f.write("")

    def get_inference(self):
        raise NotImplementedError()


class BaseEmbedSoundscapesAudioWorkflow(luigi.Task):
    remote_root = luigi.Parameter()
    local_root = luigi.Parameter()

    audio_path = luigi.Parameter()
    metadata_path = luigi.Parameter()
    intermediate_path = luigi.Parameter()
    output_path = luigi.Parameter()

    total_batches = luigi.IntParameter(default=100)
    num_partitions = luigi.IntParameter(default=16)

    def get_task(self, batch_number: int) -> BaseEmbedSoundscapesAudio:
        raise NotImplementedError()

    def output(self):
        return maybe_gcs_target(f"{self.output_path}/_SUCCESS")

    def run(self):
        yield [self.get_task(i) for i in range(self.total_batches)]

        with spark_resource() as spark:
            (
                spark.read.parquet(f"{self.intermediate_path}/*/*.parquet")
                .repartition(self.num_partitions)
                .write.parquet(f"{self.output_path}", mode="overwrite")
            )
