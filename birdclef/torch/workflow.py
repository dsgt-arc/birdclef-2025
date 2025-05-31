import os
import typer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from .data import BirdDataModule
from .model import LinearClassifier

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


@app.command()
def main(
    input_path: str,
    output_path: str,
):
    # load and preprocess data
    df = load_preprocess_data(input_path)

    # train/test split
    X_train, X_test, y_train, y_test = perform_train_test_split(df)

    # create label index mapping
    unique_labels = sorted(df["species_name"].unique())
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    num_classes = len(label_to_idx)

    # instantiate DataModule
    data_module = BirdDataModule(X_train, X_test, y_train, y_test, label_to_idx)

    # instantiate model
    model = LinearClassifier(input_dim=1280, num_classes=num_classes)

    # logger and callbacks
    logger = TensorBoardLogger("tb_logs", name="linear_classifier")

    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss",
        dirpath="~/p-dsgt_clef2025-0/shared/birdclef/checkpoints",
        filename="best-checkpoint",
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

    # evaluate and save learner report
    report_path = output_path.replace(".pkl", "_report.txt")

    print("Training completed successfully!")


if __name__ == "__main__":
    app()
