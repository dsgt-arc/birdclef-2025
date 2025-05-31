import os
import json
import typer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from .data import BirdDataModule
from .model import LinearClassifier
import pytorch_lightning as pl

pl.seed_everything(42, workers=True)  # for reproducibility

app = typer.Typer()


def load_preprocess_data(input_path: str) -> pd.DataFrame:
    df = pd.read_parquet(input_path)
    # concatenate all embeddings into a single DataFrame
    df["species_name"] = df["file"].apply(
        lambda x: x.split("train_audio/")[1].split("/")[0]
    )
    # train/test split requries y label to have at least 2 samples
    # remove species with less than 2 samples
    species_count = df["species_name"].value_counts()
    valid_species = species_count[species_count >= 2].index
    filtered_df = df[df["species_name"].isin(valid_species)].reset_index(drop=True)
    # concatenate embeddings
    embed_cols = list(map(str, range(1280)))
    filtered_df["embeddings"] = filtered_df[embed_cols].values.tolist()
    # downsample for debugging
    df_embs = filtered_df[["species_name", "embeddings"]].copy()
    print(f"DataFrame shape: {df_embs.shape}")
    print(f"Embedding size: {len(df_embs['embeddings'].iloc[0])}")
    return df_embs


def perform_train_test_split(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple:
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        np.stack(df["embeddings"]),
        df["species_name"],
        test_size=test_size,
        stratify=df["species_name"],
    )

    # data shape
    print(f"X_train, X_test shape: {X_train.shape, X_test.shape}")
    print(f"y_train, y_test shape: {y_train.shape, y_test.shape}")
    return X_train, X_test, y_train, y_test


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
):
    # load and preprocess data
    df = load_preprocess_data(input_path)

    # train/test split
    X_train, X_test, y_train, y_test = perform_train_test_split(df)

    # create label index mapping
    label_to_idx = label_index_mapping(df, output_path)
    num_classes = len(label_to_idx)

    # instantiate DataModule
    data_module = BirdDataModule(X_train, X_test, y_train, y_test, label_to_idx)

    # instantiate model
    input_dim = X_train.shape[1]
    model = LinearClassifier(
        input_dim=input_dim, num_classes=num_classes, lr=learning_rate
    )

    # logger and callbacks
    logger = TensorBoardLogger("tb_logs", name="linear_classifier")

    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"{output_path}/checkpoints",
        filename=f"best-checkpoint-{model_name}-{{epoch:02d}}-{{val_loss:.2f}}",
        save_top_k=1,
        mode="min",
    )

    early_stopping_cb = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    # trainer
    trainer = Trainer(
        max_epochs=50,
        logger=logger,
        callbacks=[checkpoint_cb, early_stopping_cb],
        accelerator="auto",
    )

    # fit model
    trainer.fit(model, datamodule=data_module)

    # validate results
    val_results = trainer.validate(model, datamodule=data_module)
    print("Validation Results:", val_results)
    print(f"Model saved to {output_path}/checkpoints")


if __name__ == "__main__":
    app()
