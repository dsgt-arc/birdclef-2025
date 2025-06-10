"""Module for running inference in the BirdCLEF Kaggle competition."""

import json
from pathlib import Path

import bioacoustics_model_zoo as bmz
import polars as pl
import tqdm
import typer
from rich import print

from birdclef.config import model_config
from birdclef.torch.model import LinearClassifier
import torch
import multiprocessing as mp
from .compile import load_tflite_interpreter, run_perch_tflite

app = typer.Typer()


def process_part(
    input_path,
    output_path,
    model_path,
    model_name,
    part: int,
    total_parts: int,
    limit=None,
):
    """Process a single part of the audio files."""
    # load the bmz model
    model_path = Path(model_path).expanduser()
    embedder = bmz.list_models()[model_name]()
    if model_name == "Perch":
        # look for tflite file next to the label_to_idx...
        tflite_path = list(model_path.glob("*.tflite"))[0]
        interpreter = load_tflite_interpreter(tflite_path.as_posix())

        def embed_func(audio_file):
            return run_perch_tflite(
                interpreter, embedder.predict_dataloader([audio_file.as_posix()])
            )
    else:

        def embed_func(audio_file):
            return embedder.embed(
                [audio_file.as_posix()],
                return_preds=False,
                clip_step=model_config[model_name]["clip_step"],
            )

    # load the classification head
    # NOTE: we could optimize this by compiling into onnx
    label_to_index = json.loads((model_path / "label_to_idx.json").read_text())
    checkpoint = list(model_path.glob("checkpoints/*.ckpt"))[0]
    classifier = LinearClassifier.load_from_checkpoint(
        checkpoint.as_posix(),
        input_dim=model_config[model_name]["embed_size"],
        num_classes=len(label_to_index),
    )
    classifier.eval()
    # classifier = LinearClassifier(
    #     model_config[model_name]["embed_size"], len(label_to_index)
    # )

    # let's embed one file at a time; there's no need to batch them all into a single frame
    audio_files = sorted(Path(input_path).expanduser().glob("*.ogg"))
    if limit is not None:
        audio_files = audio_files[:limit]
    audio_files = [
        audio_file
        for i, audio_file in enumerate(audio_files)
        if i % total_parts == part
    ]
    for audio_file in tqdm.tqdm(
        audio_files, desc=f"Embedding audio files ({part + 1}/{total_parts})"
    ):
        df = embed_func(audio_file)
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
        X = df.get_column("embedding").to_torch().to(torch.float32)
        # convert to polars DataFrame
        with torch.no_grad():
            pred = torch.softmax(classifier(X), dim=1)
        df = df.with_columns(pl.Series("predictions", pred.numpy().tolist()))
        temp_file = Path(output_path) / f"intermediate/{audio_file.stem}.parquet"
        temp_file.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(temp_file.as_posix())


@app.command()
def main(
    input_path: str = typer.Argument(..., help="Path to the input audio files."),
    output_path: str = typer.Argument(..., help="Path to save the output results."),
    model_path: str = typer.Argument(..., help="Path to the pre-trained model."),
    model_name: str = typer.Argument(..., help="Name of the embedding model."),
    num_worker: int = typer.Option(1, help="Number of worker processes to use."),
    limit: int | None = typer.Option(
        None,
        help="Limit the number of audio files to process. If None, process all files.",
    ),
):
    # Parallelize the processing of audio files using mp.Pool
    with mp.Pool(num_worker) as pool:
        pool.starmap(
            process_part,
            [
                (
                    input_path,
                    output_path,
                    model_path,
                    model_name,
                    part,
                    num_worker,
                    limit,
                )
                for part in range(num_worker)
            ],
        )

    model_path = Path(model_path).expanduser()
    label_to_index = json.loads((model_path / "label_to_idx.json").read_text())
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
