"""Module for running inference in the BirdCLEF Kaggle competition."""

import json
from pathlib import Path

import bioacoustics_model_zoo as bmz
import polars as pl
import tqdm
import typer
from rich import print
from opensoundscape import Audio
from opensoundscape.spectrogram import MelSpectrogram
import librosa
from birdclef.config import model_config
from birdclef.torch.model import LinearClassifier
import torch
import multiprocessing as mp
from birdclef.kaggle.compile import load_tflite_interpreter, run_perch_tflite, run_birdnet_tflite
from birdclef.kaggle.offline.birdset_efficientnet import BirdSetEfficientNetB1Offline
from birdclef.kaggle.offline.birdset_convnext import BirdSetConvNeXTOffline
from birdclef.kaggle.offline.rana_sierrae_cnn import RanaSierraeCNNOffline
from birdclef.mel2vec.loaders import get_word_vectors, get_index
import numpy as np
import pandas as pd

app = typer.Typer()


def _process_mfcc(audio_path, index, word_vector, n_mfcc=20):
    """Process a single audio file and return its mel spectrogram."""
    audio = Audio.from_file(audio_path.as_posix(), sample_rate=32000)
    spec = MelSpectrogram.from_audio(
        audio,
        n_mels=128,
        fft_size=8192,
        window_samples=8000,
        overlap_fraction=0.5,
        # dont want to double log things
        dB_scale=False,
    )
    mfccs = librosa.feature.mfcc(
        S=spec.spectrogram,
        sr=spec.audio_sample_rate,
        n_mfcc=n_mfcc,
    )
    # and now we return a pandas dataframe with the data
    _, indices = index.search(mfccs.T, 1)
    # indices is a (n_frames, 1) array
    vectors = word_vector[indices.flatten()]

    # and now we average the word vectors on 5 second intervals
    # there must be a smarter way of doing this...
    groups = {}
    for i, t in enumerate(spec.times):
        group = int(t // 5)
        if group not in groups:
            groups[group] = []
        groups[group].append(vectors[i])

    df = pd.DataFrame(
        data=[np.mean(groups[i], axis=0) for i in sorted(groups.keys())],
        # index should include file and end_time
        index=pd.MultiIndex.from_frame(
            pd.DataFrame(
                {
                    "file": [audio_path.as_posix()] * len(groups),
                    "end_time": [(i + 1) * 5 for i in sorted(groups.keys())],
                }
            )
        ),
    )
    return df


def get_embed_func(model_name, model_path, clip_step, tflite_threads):
    if model_name == "mel2vec":
        index = get_index(model_path / "centroids.npy")
        word_vector = get_word_vectors(model_path / "word2vec.wordvectors")

        def embed_func(audio_file):
            return _process_mfcc(audio_file, index, word_vector, n_mfcc=20)
    else:
        if model_name == "BirdSetEfficientNetB1":
            embedder = BirdSetEfficientNetB1Offline(model_path)
        elif model_name == "BirdSetConvNeXT":
            embedder = BirdSetConvNeXTOffline(model_path)
        elif model_name == "RanaSierraeCNN":
            embedder = RanaSierraeCNNOffline(model_path)
        else:
            embedder = bmz.list_models()[model_name]()
        
        if model_name in ["BirdNET", "Perch"]:
            # look for tflite file next to the label_to_idx...
            tflite_path = list(model_path.glob("*.tflite"))[0]
            interpreter = load_tflite_interpreter(
                tflite_path.as_posix(), num_threads=tflite_threads
            )

            def embed_func(audio_file):
                if model_name == "BirdNET":
                    return run_birdnet_tflite(
                        interpreter,
                        embedder.predict_dataloader(
                            [audio_file.as_posix()], clip_step=clip_step
                        ),
                    )
                else:
                    return run_perch_tflite(
                        interpreter,
                        embedder.predict_dataloader(
                            [audio_file.as_posix()], clip_step=clip_step
                        ),
                    )
        else:

            def embed_func(audio_file):
                return embedder.embed(
                    [audio_file.as_posix()],
                    return_preds=False,
                    clip_step=clip_step,
                )

    return embed_func


def process_part(
    input_path,
    output_path,
    model_path,
    model_name,
    part: int,
    total_parts: int,
    limit=None,
    tflite_threads=None,
):
    """Process a single part of the audio files."""
    # load the bmz model
    model_path = Path(model_path).expanduser()
    # special casing for non-bmz models
    clip_step = (
        model_config[model_name]["clip_step"] if model_name in model_config else 5.0
    )
    embed_func = get_embed_func(model_name, model_path, clip_step, tflite_threads)

    # load the classification head
    # NOTE: we could optimize this by compiling into onnx
    label_to_index = json.loads((model_path / "label_to_idx.json").read_text())
    checkpoint = list(model_path.glob("checkpoints/*.ckpt"))[0]
    classifier = LinearClassifier.load_from_checkpoint(
        checkpoint.as_posix(),
        input_dim=(
            model_config[model_name]["embed_size"]
            if model_name in model_config
            else {"mel2vec": 384}[model_name]
        ),
        num_classes=len(label_to_index),
    )
    classifier.eval()

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

        if clip_step != 5.0:
            # aggregate predictions into 5-second intervals
            interval_length = 5
            df = df.with_columns(
                ((pl.col("start_time") + pl.col("end_time")) / 2 / interval_length)
                .cast(pl.Int64)
                .alias("interval")
            )
            # average predictions
            df = df.group_by(["file", "interval"]).agg(
                pl.col("*")
                .exclude(["file", "interval", "start_time", "end_time"])
                .mean()
            )
            # convert interval to end_time
            df = df.with_columns(
                pl.col("interval").add(1).mul(interval_length).alias("end_time")
            ).drop("interval")
        else:
            # ensure same format as above case with aggregation
            try:
                df = df.drop("start_time")
            except Exception:
                pass

        # generate the embedding vector
        df = df.select(
            "file",
            "end_time",
            (
                pl.concat_list(pl.all().exclude("file", "end_time"))
                .list.to_array(len(df.columns) - 2)
                .alias("embedding")
            ),
        ).sort("file", "end_time")
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
    tflite_threads: int | None = typer.Option(
        None,
        help="Number of threads to use for TFLite inference. If None, use the default number of threads.",
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
                    tflite_threads,
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
