import numpy as np
import torch
import torchaudio
from birdnetlib import RecordingBuffer
from birdnetlib.analyzer import Analyzer
from torchaudio.transforms import Resample
from birdclef.inference.base import BaseInference
from typing import Iterable, Tuple, Optional


class BirdNetInference(BaseInference):
    """Class to perform inference on audio files using a Google Vocalization model."""

    def __init__(
        self,
        max_length: int = 0,
        use_gpu: bool = True,
    ):
        self._import_tensorflow(use_gpu)
        self.max_length = max_length
        self.resampler = Resample(32_000, 48_000)
        self.source_sr = 32_000
        self.target_sr = 48_000
        self.analyzer = Analyzer(verbose=False)

    def _import_tensorflow(self, use_gpu: bool) -> None:
        """Import tensorflow and return the version."""
        import tensorflow as tf

        if not use_gpu:
            # We don't want to run BirdNET on a GPU
            # https://datascience.stackexchange.com/a/76039
            try:
                # Disable all GPUS
                tf.config.set_visible_devices([], "GPU")
                visible_devices = tf.config.get_visible_devices()
                for device in visible_devices:
                    assert device.device_type != "GPU"
            except Exception:
                # Invalid device or cannot modify virtual devices once initialized.
                pass

    def load(self, path: str, window_sec: int = 5) -> torch.Tensor:
        """Load an audio file.

        :param path: The absolute path to the audio file.
        """
        audio, _ = torchaudio.load(str(path))
        audio = audio[0]
        # right pad the audio so we can reshape into a rectangle
        n = audio.shape[0]
        window = window_sec * self.source_sr
        if (n % window) != 0:
            audio = torch.concatenate([audio, torch.zeros(window - (n % window))])

        audio = self.resampler(audio)
        window = window_sec * self.target_sr
        # reshape the audio into windowsize chunks
        audio = audio.reshape(-1, window)
        if self.max_length > 0:
            audio = audio[: self.max_length]
        return audio

    def _infer(self, audio) -> torch.Tensor:
        recording = RecordingBuffer(
            self.analyzer,
            audio.squeeze(),
            self.target_sr,
            overlap=1,
            verbose=False,
        )
        recording.extract_embeddings()
        # concatenate the embeddings together, this should only be two of them
        return torch.stack(
            [
                torch.from_numpy(np.array(r["embeddings"])).to(dtype=torch.float32)
                for r in recording.embeddings
            ],
        ).mean(axis=0)

    def predict(
        self, path: str, window: int = 5, **kwargs
    ) -> Iterable[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]:
        """Get embeddings and logits for a single audio file.

        :param path: The absolute path to the audio file.
        :param window: The size of the window to split the audio into.
        """
        audio = self.load(path, window).numpy()
        for row in audio:
            yield self._infer(row), None
