import luigi
import pandas as pd


from birdclef.luigi import maybe_gcs_target
from birdclef.spark import spark_resource


class BaseEmbedSpeciesAudio(luigi.Task):
    """Embed all audio files for a species and save to a parquet file."""

    audio_path = luigi.Parameter()
    metadata_path = luigi.Parameter()
    output_path = luigi.Parameter()
    species = luigi.Parameter()

    @property
    def resources(self):
        return {self.output().path: 1}

    def output(self):
        return maybe_gcs_target(f"{self.output_path}/{self.species}.parquet")

    def run(self):
        inference = self.get_inference()
        out_path = f"{self.output_path}/{self.species}.parquet"
        df = inference.predict_species_df(
            f"{self.audio_path}",
            self.species,
            out_path,
        )
        print(df.head())

    def get_inference(self):
        raise NotImplementedError()


class BaseEmbedSpeciesWorkflow(luigi.Task):
    """Embed all audio files for all species and save to a parquet file."""

    audio_path = luigi.Parameter()
    output_path = luigi.Parameter()
    metadata_path = luigi.Parameter()
    intermediate_path = luigi.Parameter()

    partitions = luigi.IntParameter(default=16)

    def get_species_list(self):
        metadata = pd.read_csv(f"{self.metadata_path}")
        return metadata["primary_label"].unique()

    def get_task(self, species):
        raise NotImplementedError()

    def output(self):
        return maybe_gcs_target(f"{self.output_path}/_SUCCESS")

    def run(self):
        species_list = self.get_species_list()
        tasks = []
        for species in species_list:
            task = self.get_task(species)
            tasks.append(task)
        yield tasks

        with spark_resource() as spark:
            (
                spark.read.parquet(f"{self.intermediate_path}/*.parquet")
                .repartition(self.partitions)
                .write.parquet(f"{self.output_path}", mode="overwrite")
            )
