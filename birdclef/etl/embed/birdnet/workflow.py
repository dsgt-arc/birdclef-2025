import luigi
import typer
from typing_extensions import Annotated

from birdclef.etl.embed.base.soundscapes import (
    BaseEmbedSoundscapesAudio,
    BaseEmbedSoundscapesAudioWorkflow,
)
from birdclef.inference.birdnet import BirdNetInference


class BirdNetEmbedSoundscapesAudio(BaseEmbedSoundscapesAudio):
    def get_inference(self):
        return BirdNetInference(metadata_path=self.metadata_path)


class BirdNetEmbedSoundscapesAudioWorkflow(BaseEmbedSoundscapesAudioWorkflow):
    def get_task(self, batch_number):
        return BirdNetEmbedSoundscapesAudio(
            audio_path=self.audio_path,
            metadata_path=self.metadata_path,
            output_path=self.intermediate_path,
            batch_number=batch_number,
        )


def main(
    audio_path: Annotated[str, typer.Argument(help="Path to audio files")],
    metadata_path: Annotated[str, typer.Argument(help="Path to metadata")],
    intermediate_path: Annotated[str, typer.Argument(help="Path to intermediate data")],
    output_path: Annotated[str, typer.Argument(help="Path to output data")],
    total_batches: Annotated[int, typer.Option(help="Total number of batches")] = 100,
    num_partitions: Annotated[int, typer.Option(help="Number of partitions")] = 16,
    scheduler_host: Annotated[str, typer.Option(help="Scheduler host")] = None,
):
    """Embed soundscapes using BirdNet."""
    kwargs = (
        {"scheduler_host": scheduler_host}
        if scheduler_host
        else {"local_scheduler": True}
    )

    luigi.build(
        [
            BirdNetEmbedSoundscapesAudioWorkflow(
                audio_path=audio_path,
                metadata_path=metadata_path,
                intermediate_path=intermediate_path,
                output_path=output_path,
                total_batches=total_batches,
                num_partitions=num_partitions,
            )
        ],
        **kwargs,
    )
