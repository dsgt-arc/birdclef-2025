import bioacoustics_model_zoo as bmz
from pathlib import Path
import typer
import tensorflow as tf
import numpy as np
from contexttimer import Timer
import pandas as pd

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
def benchmark_perch(test_audio: Path, compiled_root: Path, batch_size: int = 1):
    """Benchmark the Perch model."""
    audio = [p.as_posix() for p in Path(test_audio).expanduser().glob("*.ogg")]
    interpreter = load_tflite_interpreter(compiled_root / "perch.tflite")
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


if __name__ == "__main__":
    app()
