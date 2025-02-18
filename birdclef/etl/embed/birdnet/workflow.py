from birdclef.etl.embed.base.soundscapes import (
    BaseEmbedSoundscapesAudio,
    BaseEmbedSoundscapesAudioWorkflow,
)
from birdclef.inference.birdnet import BirdNetInference


class BirdNetEmbedSoundscapesAudio(BaseEmbedSoundscapesAudio):
    def get_inference(self):
        return BirdNetInference(
            metadata_path=f"{self.remote_root}/{self.metadata_path}",
        )


class BirdNetEmbedSoundscapesAudioWorkflow(BaseEmbedSoundscapesAudioWorkflow):
    def get_task(self, batch_number):
        return BirdNetEmbedSoundscapesAudio(
            audio_path=self.audio_path,
            metadata_path=self.metadata_path,
            output_path=self.intermediate_path,
            batch_number=batch_number,
        )
