from pathlib import Path

import lightning as pl
import pandas as pd
import torch
from birdclef.config import SPECIES
from birdclef.experiment.model import LinearClassifier
from lightning.pytorch.profilers import AdvancedProfiler
from tqdm import tqdm

from .data import BirdNetSoundscapeDataModule


class PassthroughModel(pl.LightningModule):
    def forward(self, x):
        return x

    def predict_step(self, batch, batch_idx):
        batch["prediction"] = torch.ones((len(batch["row_id"]), len(SPECIES))) * 0.5
        return batch


def make_submission(
    soundscape_path: str,
    metadata_path: str,
    output_csv_path: str,
    model_path: str,
    model_type: str = "passthrough",
    batch_size: int = 32,
    num_workers: int = 0,
    limit=None,
    should_profile=False,
    profile_path="logs/perf_logs",
):
    Path(output_csv_path).parent.mkdir(exist_ok=True, parents=True)
    dm = BirdNetSoundscapeDataModule(
        soundscape_path=soundscape_path,
        metadata_path=metadata_path,
        batch_size=batch_size,
        num_workers=num_workers,
        limit=limit,
    )
    kwargs = dict()
    if should_profile:
        profiler = AdvancedProfiler(dirpath="logs", filename=profile_path)
        kwargs["profiler"] = profiler
    trainer = pl.Trainer(**kwargs)

    if model_type == "passthrough":
        model = PassthroughModel
    elif model_type == "linear":
        model_class = LinearClassifier
    else:
        raise ValueError(f"invalid class: {model_type}")
    model = model_class.load_from_checkpoint(model_path)

    predictions = trainer.predict(model, dm)

    rows = []
    for batch in tqdm(predictions):
        for row_id, prediction in zip(batch["row_id"], batch["prediction"]):
            predictions = zip(SPECIES, prediction.numpy().tolist())
            row = {"row_id": row_id, **dict(predictions)}
            rows.append(row)
    submission_df = pd.DataFrame(rows)[["row_id", *SPECIES]]
    submission_df.to_csv(output_csv_path, index=False)
    return submission_df
