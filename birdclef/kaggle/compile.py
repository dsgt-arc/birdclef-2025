from pathlib import Path

import bioacoustics_model_zoo as bmz
import numpy as np
import pandas as pd
import tensorflow as tf
import typer
from contexttimer import Timer
from birdclef.torch.model import LinearClassifier
from birdclef.config import model_config
import openvino as ov
import json
import torch

app = typer.Typer()


def load_tflite_interpreter(model_path: Path):
    """Load a TFLite interpreter for the given model path."""
    interpreter = tf.lite.Interpreter(
        model_path=Path(model_path).expanduser().as_posix()
    )
    interpreter.allocate_tensors()
    return interpreter


def run_perch_tflite(interpreter, dataloader) -> pd.DataFrame:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    res = []
    for batch in dataloader:
        interpreter.set_tensor(input_details[0]["index"], batch[0])
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[1]["index"])
        res.append(output_data)
    # https://github.com/kitzeslab/bioacoustics-model-zoo/blob/main/bioacoustics_model_zoo/perch.py
    return pd.DataFrame(
        data=np.stack(res).squeeze(),
        index=dataloader.dataset.dataset.label_df.index,
    )


@app.command()
def compile_perch(compiled_root: Path):
    perch = bmz.list_models()["Perch"]()
    # we still need to use this model to get the dataloader

    converter = tf.lite.TFLiteConverter.from_keras_model(perch.tf_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Flex ops: FlexEnsureShape, FlexStridedSlice, FlexTranspose
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    tflite_model = converter.convert()
    compiled_root = Path(compiled_root).expanduser()
    compiled_root.mkdir(parents=True, exist_ok=True)
    with (compiled_root / "perch.tflite").open("wb") as f:
        f.write(tflite_model)


@app.command()
def compile_classifier_head(compiled_root: Path, model_name: str, checkpoint: Path):
    """Compile the classifier head for the given model."""
    if model_name not in model_config:
        raise ValueError(f"Model {model_name} is not supported.")

    # this is annoying...
    checkpoint = Path(checkpoint).expanduser()
    label_to_index = json.loads(
        (checkpoint.parent.parent / "label_to_idx.json").read_text()
    )
    model = LinearClassifier.load_from_checkpoint(
        checkpoint,
        input_dim=model_config[model_name]["embed_size"],
        num_classes=len(label_to_index),
    )
    model.eval()
    # now compile to openvino and save for later use
    compiled_root = Path(compiled_root).expanduser()
    compiled_root.mkdir(parents=True, exist_ok=True)
    example_input = torch.from_numpy(
        np.random.rand(1, model_config[model_name]["embed_size"]).astype(np.float32)
    )
    model.to_onnx(
        compiled_root / f"{model_name}_head.onnx", example_input, export_params=True
    )
    print(f"Saved compiled model to {compiled_root / f'{model_name}_head.onnx'}")


@app.command()
def benchmark_perch(
    test_audio: Path,
    compiled_root: Path,
    checkpoint: Path,
    batch_size: int = 1,
):
    """Benchmark the Perch model."""
    audio = [p.as_posix() for p in Path(test_audio).expanduser().glob("*.ogg")]
    interpreter = load_tflite_interpreter(Path(compiled_root) / "perch.tflite")
    perch = bmz.list_models()["Perch"]()

    # initialize perch for xla compilation
    perch.embed(audio[:1], batch_size=batch_size)

    print("Running Perch")
    with Timer() as timer:
        res = perch.embed(audio, batch_size=batch_size)
    print(f"data in shape {res.shape}")
    print(f"Perch took {timer.elapsed:.2f} seconds for {len(audio)} files")
    print(res.head())

    print("Running Perch TFLite model...")
    with Timer() as timer:
        dataloader = perch.predict_dataloader(audio, batch_size=batch_size)
        res = run_perch_tflite(interpreter, dataloader)
    print(res.head())

    print(f"Perch TFLite took {timer.elapsed:.2f} seconds for {len(audio)} files")

    # run the classification head
    model_name = "Perch"
    checkpoint = Path(checkpoint).expanduser()
    label_to_index = json.loads(
        (checkpoint.parent.parent / "label_to_idx.json").read_text()
    )
    model = LinearClassifier.load_from_checkpoint(
        checkpoint,
        input_dim=model_config[model_name]["embed_size"],
        num_classes=len(label_to_index),
    )

    # let's now see how long it takes to run the classification head
    print("Running classification head")

    X = res.to_numpy().astype(np.float32)
    with Timer() as timer:
        with torch.no_grad():
            _ = torch.softmax(model(torch.from_numpy(X)), dim=1)
    print(
        f"Classification head took {timer.elapsed:.2f} seconds for {len(audio)} files"
    )

    # now do the same with openvino
    compiled_root = Path(compiled_root).expanduser()
    ov_model = ov.compile_model(
        compiled_root / f"{model_name}_head.onnx",
        device_name="CPU",
    )

    with Timer() as timer:
        for row in X:
            _ = ov_model(row.reshape(1, -1))
    print(
        f"OpenVINO classification head took {timer.elapsed:.2f} seconds for {len(audio)} files"
    )


if __name__ == "__main__":
    app()
