import datetime
from pathlib import Path
from typing import Optional
import shutil

import torch
import lightning as pl
import typer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from birdclef.config import get_species

from .data import BirdNetSpeciesDataModule
from .model import LinearClassifier


def train_model(
    audio_path: str = typer.Argument(..., help="Path to audio files directory"),
    metadata_path: str = typer.Argument(..., help="Path to metadata CSV file"),
    output_dir: str = typer.Argument(..., help="Directory to save model outputs"),
    batch_size: int = typer.Option(32, help="Batch size for training"),
    num_workers: int = typer.Option(4, help="Number of data loading workers"),
    max_epochs: int = typer.Option(30, help="Maximum number of training epochs"),
    learning_rate: float = typer.Option(0.002, help="Learning rate"),
    limit: Optional[int] = typer.Option(
        None, help="Limit dataset size (for debugging)"
    ),
    model_type: str = typer.Option("linear", help="Type of model to train"),
    val_size: float = typer.Option(0.15, help="Validation set size ratio"),
    test_size: float = typer.Option(0.15, help="Test set size ratio"),
    patience: int = typer.Option(5, help="Early stopping patience"),
    accelerator: str = typer.Option("cuda", help="Accelerator to use"),
):
    """Train a bird classification model using BirdNet embeddings."""
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # check cuda exists, and number of devices
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Configure trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_path / "checkpoints"),
        filename=f"{model_type}-{{epoch:02d}}-{{val_loss:.2f}}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=patience, mode="min", verbose=True
    )
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=TensorBoardLogger(
            save_dir=str(output_path / "logs"), name=f"{model_type}_{timestamp}"
        ),
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator=accelerator,
        devices="auto",
        strategy="auto",
    )

    # Create the model
    if model_type == "linear":
        model = LinearClassifier(
            num_features=1024,
            num_labels=len(get_species()),
            learning_rate=learning_rate,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(model)

    # Train the model
    data_module = BirdNetSpeciesDataModule(
        audio_path=audio_path,
        metadata_path=metadata_path,
        batch_size=batch_size,
        num_workers=num_workers,
        limit=limit,
        val_size=val_size,
        test_size=test_size,
    )

    trainer.fit(model, data_module)

    # Test the model
    trainer.test(model, datamodule=data_module)

    # Save the best model
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        final_model_path = output_path / f"{model_type}_best_model.ckpt"
        shutil.copy(best_model_path, final_model_path)
        print(f"Best model saved to {final_model_path}")

    return best_model_path


if __name__ == "__main__":
    typer.run(train_model)
