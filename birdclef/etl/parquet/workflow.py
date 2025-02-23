import luigi
from birdclef.luigi import maybe_gcs_target
from birdclef.spark import spark_resource
import typer
from typing_extensions import Annotated

app = typer.Typer(no_args_is_help=True)


class RepartitionParquet(luigi.Task):
    """Repartition the parquet files into a single file."""

    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    num_partitions = luigi.IntParameter(default=16)

    @property
    def resources(self):
        return {self.output().path: 1}

    def output(self):
        return maybe_gcs_target(f"{self.output_path}/_SUCCESS")

    def run(self):
        with spark_resource() as spark:
            (
                spark.read.parquet(self.input_path)
                .repartition(self.num_partitions)
                .write.parquet(self.output_path, mode="overwrite")
            )


@app.command("repartition")
def repartition(
    input_path: Annotated[str, typer.Argument(help="Path to input data")],
    output_path: Annotated[str, typer.Argument(help="Path to output data")],
    num_partitions: Annotated[
        int, typer.Option(help="Number of final parquet partitions")
    ] = 16,
    scheduler_host: Annotated[str, typer.Option(help="Scheduler host")] = None,
):
    """Embed soundscapes using BirdNet."""
    luigi.build(
        [
            RepartitionParquet(
                input_path=input_path,
                output_path=output_path,
                num_partitions=num_partitions,
            )
        ],
        **(
            {"scheduler_host": scheduler_host}
            if scheduler_host
            else {"local_scheduler": True}
        ),
    )
