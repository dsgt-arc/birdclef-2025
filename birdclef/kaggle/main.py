"""Module for running inference in the BirdCLEF Kaggle competition."""

import json
from pathlib import Path

import bioacoustics_model_zoo as bmz
import polars as pl
import tqdm
import typer
from rich import print

from birdclef.model_config import model_config
from birdclef.torch.model import LinearClassifier
import torch

app = typer.Typer()


@app.command()
def main(
    input_path: str = typer.Argument(..., help="Path to the input audio files."),
    output_path: str = typer.Argument(..., help="Path to save the output results."),
    model_path: str = typer.Argument(..., help="Path to the pre-trained model."),
    model_name: str = typer.Argument(..., help="Name of the embedding model."),
):
    # load the bmz model
    embedder = bmz.list_models()[model_name]()

    # load the classification head
    # NOTE: we could optimize this by compiling into onnx
    model_path = Path(model_path).expanduser()
    # checkpoint = list(model_path.glob("checkpoints/*.ckpt"))[0]
    # classifier = LinearClassifier.load_from_checkpoint(checkpoint.as_posix())'
    label_to_index = json.loads((model_path / "label_to_idx.json").read_text())
    classifier = LinearClassifier(
        model_config[model_name]["embed_size"], len(label_to_index)
    )

    # let's embed one file at a time; there's no need to batch them all into a single frame
    audio_files = sorted(Path(input_path).expanduser().glob("*.ogg"))
    for audio_file in tqdm.tqdm(audio_files, desc="Embedding audio files"):
        df = embedder.embed(
            audio_file.as_posix(),
            return_preds=False,
            clip_step=model_config[model_name]["clip_step"],
        )
        df = pl.from_pandas(df.reset_index())
        # generate the embedding vector
        df = df.select(
            "file",
            "start_time",
            "end_time",
            (
                pl.concat_list(df.columns[3:])
                .list.to_array(len(df.columns[3:]))
                .alias("embedding")
            ),
        ).sort("file", "start_time")
        # and now run inference on the embedding vector
        with torch.no_grad():
            X = df.get_column("embedding").to_torch().to(torch.float32)
            # make sure to softmax
            pred = classifier(X)
            pred = torch.softmax(pred, dim=1)
            # convert to polars DataFrame
        df = df.with_columns(
            pl.Series(
                "predictions",
                pred.numpy().tolist(),
            )
        )
        temp_file = Path(output_path) / f"intermediate/{audio_file.stem}.parquet"
        temp_file.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(temp_file.as_posix())

    # and now we generate the file submission
    df = pl.scan_parquet(f"{output_path}/intermediate")
    # output is row_id in format {stem}_{end_time}, and a column for each prediction
    df = df.select(
        pl.concat_str(
            [
                # split by "/" to get last part and replace ogg
                pl.col("file").str.split("/").list.last().str.replace(".ogg", ""),
                # end_time needs to be an integer
                pl.col("end_time").cast(int).cast(str),
            ],
            separator="_",
        ).alias("row_id"),
        *[
            pl.col("predictions").list.get(i).alias(label)
            for label, i in sorted(label_to_index.items(), key=lambda x: x[1])
        ],
    ).sort("row_id")
    # write to the final submission file
    output_file = Path(output_path) / "submission.csv"
    df.collect().write_csv(output_file.as_posix(), include_header=True)
    print(f"Submission file written to: {output_file.as_posix()}")


if __name__ == "__main__":
    app()
