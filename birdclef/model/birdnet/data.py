import math
from pathlib import Path

import lightning as pl
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from birdclef.inference import BaseInference, BirdNetInference
from torch.utils.data import DataLoader, IterableDataset


class AudioInferenceIterableDataset(IterableDataset):
    def _num_tracks(self) -> int:
        """The number of elements to split across workers."""
        raise NotImplementedError()

    def _model(self) -> BaseInference:
        """The inference model to use."""
        raise NotImplementedError()

    def _load_data(self, iter_start, iter_end):
        """Load the data."""
        raise NotImplementedError()

    def __iter__(self):
        # https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
        worker_info = torch.utils.data.get_worker_info()
        start, end = 0, self._num_tracks()
        if worker_info is None:
            iter_start = start
            iter_end = end
        else:
            per_worker = int(math.ceil((end - start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, end)

        return self._load_data(iter_start, iter_end)


class BirdNetSpeciesDataset(AudioInferenceIterableDataset):
    def __init__(
        self,
        audio_path: str,
        metadata: str,
        max_length: int = 4 * 60 / 5,
        limit=None,
    ):
        self.audio_path = audio_path
        self.max_length = int(max_length)
        self.metadata = metadata
        if limit is not None:
            self.metadata = self.metadata[:limit]

    def _num_tracks(self):
        return len(self.metadata)

    def _load_data(self, iter_start, iter_end):
        model = BirdNetInference()
        # over all rows in the dataset, we provide embeddings of the tracks
        # TODO: we also need to split tracks that are too long by reading them
        # in at most 10 minute chunks
        for i in range(iter_start, iter_end):
            row = self.metadata.iloc[i]
            path = Path(self.audio_path) / row["filename"]
            embeddings, _ = model.predict(path)
            embeddings = embeddings[: self.max_length].float()
            for idx, embedding in enumerate(embeddings):
                yield {
                    "row_id": f"{row['filename']}_{(idx + 1) * 5}",
                    "embedding": embedding,
                    "species": row["primary_label"],
                }


class BirdNetSoundscapeDataset(AudioInferenceIterableDataset):
    """Dataset meant for inference on soundscape data."""

    def __init__(
        self,
        soundscape_path: str,
        metadata_path: str,
        max_length: int = 4 * 60 / 5,
        limit=None,
    ):
        self.soundscapes = sorted(Path(soundscape_path).glob("**/*.ogg"))
        self.max_length = int(max_length)
        if limit is not None:
            self.soundscapes = self.soundscapes[:limit]
        self.metadata_path = metadata_path

    def _num_tracks(self):
        return len(self.soundscapes)

    def _load_data(self, iter_start, iter_end):
        model = BirdNetInference()
        for i in range(iter_start, iter_end):
            path = self.soundscapes[i]
            embeddings, _ = model.predict(path)
            embeddings = embeddings[: self.max_length].float()
            for idx, embedding in enumerate(embeddings):
                yield {
                    "row_id": f"{path.stem}_{(idx + 1) * 5}",
                    "embedding": embedding,
                }


class BirdNetSpeciesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        audio_path: str,
        metadata_path: str,
        batch_size: int = 32,
        num_workers: int = 0,
        limit=None,
        val_size: float = 0.15,
        test_size: float = 0.15,
    ):
        """Initialize the data module."""
        super().__init__()
        self.audio_path = audio_path
        self.metadata_path = metadata_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.limit = limit
        self.val_size = val_size
        self.test_size = test_size

    def setup(self, stage=None):
        """Split the metadata into train, validation, and test sets."""
        # Load the metadata
        metadata = pd.read_csv(self.metadata_path)
        if self.limit is not None:
            metadata = metadata.head(self.limit)

        # Stratified split based on species
        # First split off the test set
        train_val_data, test_data = train_test_split(
            metadata,
            test_size=self.test_size,
            stratify=metadata["primary_label"],
            random_state=42,
        )

        # Then split the remaining data into train and validation
        adjusted_val_size = self.val_size / (1 - self.test_size)
        train_data, val_data = train_test_split(
            train_val_data,
            test_size=adjusted_val_size,
            stratify=train_val_data["primary_label"],
            random_state=42,
        )

        self.train_metadata = train_data
        self.val_metadata = val_data
        self.test_metadata = test_data

        self.dataloader_kwargs = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
        }

    def train_dataloader(self):
        return DataLoader(
            BirdNetSpeciesDataset(
                self.audio_path, self.train_metadata, limit=self.limit
            ),
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            BirdNetSpeciesDataset(self.audio_path, self.val_metadata),
            **self.dataloader_kwargs,
        )

    def test_dataloader(self):
        return DataLoader(
            BirdNetSpeciesDataset(self.audio_path, self.test_metadata),
            **self.dataloader_kwargs,
        )

    def predict_dataloader(self):
        return self.test_dataloader()


class BirdNetSoundscapeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        soundscape_path: str,
        metadata_path: str,
        batch_size: int = 32,
        num_workers: int = 0,
        limit=None,
    ):
        """Initialize the data module."""
        super().__init__()
        self.dataloader = DataLoader(
            BirdNetSoundscapeDataset(soundscape_path, metadata_path, limit=limit),
            batch_size=batch_size,
            num_workers=num_workers,
        )

    def predict_dataloader(self):
        return self.dataloader
