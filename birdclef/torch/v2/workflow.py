import os
import json
import typer
import pandas as pd
from sklearn.model_selection import train_test_split
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from .data import BirdDataModule
from ..model import LinearClassifier
import pytorch_lightning as pl

pl.seed_everything(42, workers=True)  # for reproducibility

app = typer.Typer()


def load_metadata(input_path: str) -> pd.DataFrame:
    df = pd.read_parquet(input_path, columns=["file"])
    # extract species_name
    df["species_name"] = df["file"].apply(
        lambda x: x.split("train_audio/")[1].split("/")[0]
    )
    # train/test split requires y label to have at least 2 samples
    # remove species with less than 2 samples
    species_count = df["species_name"].value_counts()
    valid_species = species_count[species_count >= 2].index
    df = df[df["species_name"].isin(valid_species)].reset_index(drop=True)
    return df


def perform_train_test_split(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple:
    # train/test split
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["species_name"],
        random_state=random_state,
    )

    # data shape
    print(f"train_df shape: {train_df.shape}")
    print(f"test_df shape: {test_df.shape}")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def label_index_mapping(df: pd.DataFrame, output_path: str) -> tuple:
    # create label index mapping
    unique_labels = sorted(df["species_name"].unique())
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}

    # save label mapping to disk
    label_map_path = f"{output_path}/label_to_idx.json"
    os.makedirs(os.path.dirname(label_map_path), exist_ok=True)
    with open(label_map_path, "w") as f:
        json.dump(label_to_idx, f)

    return label_to_idx


@app.command()
def main(
    input_path: str,
    output_path: str,
    model_name: str,  # Perch, BirdNET, etc.
    learning_rate: float = typer.Option(1e-3, help="Learning rate for the optimizer"),
    batch_size: int = typer.Option(64, help="Batch size for training and validation"),
):
    # load and preprocess data
    df = load_metadata(input_path)

    # train/test split
    train_df, test_df = perform_train_test_split(df)

    # create label index mapping
    label_to_idx = label_index_mapping(train_df, output_path)
    num_classes = len(label_to_idx)

    # instantiate DataModule
    data_module = BirdDataModule(
        train_df, test_df, label_to_idx, input_path, batch_size=batch_size
    )

    # get input dimension from sample data
    data_module.setup()  # prepare datasets
    sample_x, _ = data_module.train_dataset[0]
    input_dim = sample_x.shape[0]

    # instantiate model
    model = LinearClassifier(
        input_dim=input_dim, num_classes=num_classes, lr=learning_rate
    )

    # logger and callbacks
    logger = TensorBoardLogger("tb_logs", name="linear_classifier")

    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"{output_path}/checkpoints",
        filename=f"best-{model_name}-{{epoch:02d}}-{{val_loss:.2f}}",
        save_top_k=1,
        mode="min",
    )

    early_stopping_cb = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    # trainer
    trainer = Trainer(
        max_epochs=50,
        accelerator="auto",
        logger=logger,
        callbacks=[checkpoint_cb, early_stopping_cb],
    )

    # fit model
    trainer.fit(model, datamodule=data_module)

    # validate results
    val_results = trainer.validate(model, datamodule=data_module)
    print("Validation Results:", val_results)
    print(f"Model + checkpoints saved to {output_path}")


if __name__ == "__main__":
    app()
