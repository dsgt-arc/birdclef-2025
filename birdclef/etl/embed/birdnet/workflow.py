from typing import Optional
import luigi
import typer
from typing_extensions import Annotated

from birdclef.etl.embed.base.soundscapes import (
    BaseEmbedSoundscapesAudio,
    BaseEmbedSoundscapesAudioWorkflow,
)
from birdclef.inference.birdnet import BirdNetInference

app = typer.Typer(no_args_is_help=True)


class BirdNetEmbedSoundscapesAudio(BaseEmbedSoundscapesAudio):
    def get_inference(self):
        return BirdNetInference()


class BirdNetEmbedSoundscapesAudioWorkflow(BaseEmbedSoundscapesAudioWorkflow):
    def get_task(self, batch_number):
        return BirdNetEmbedSoundscapesAudio(
            audio_path=self.audio_path,
            output_path=self.output_path,
            batch_number=batch_number,
            total_batches=self.total_batches,
        )


@app.command("soundscapes")
def embed_soundscapes(
    audio_path: Annotated[str, typer.Argument(help="Path to audio files")],
    output_path: Annotated[str, typer.Argument(help="Path to output data")],
    total_batches: Annotated[int, typer.Option(help="Total number of batches")] = 200,
    limit: Annotated[Optional[int], typer.Option(help="Limit the number of files")] = None,
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
                output_path=output_path,
                total_batches=total_batches,
                limit=limit,
            )
        ],
        **kwargs,
    )
