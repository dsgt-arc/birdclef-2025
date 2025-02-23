from birdclef.inference.birdnet import BirdNetInference
import pandas as pd


def test_birdnet_inference_predict(metadata_path):
    bi = BirdNetInference()
    embedding, _ = bi.predict(metadata_path.parent / "audio/file_0.ogg")
    # 10 seconds of audio means there are 10 rows
    assert embedding.shape == (2, 1024)


def test_birdnet_inference_predict_df(metadata_path):
    bi = BirdNetInference()
    df = bi.predict_df(metadata_path.parent, "audio/file_0.ogg")
    assert len(df) == 2
    assert set(df.columns) == {"name", "chunk_5s", "embedding"}


def test_birdnet_inference_predict_species_df(metadata_path):
    bi = BirdNetInference()
    out_path = metadata_path.parent / "audio/file_0.parquet"
    metadata_df = pd.read_csv(metadata_path)
    df = bi.predict_species_df(metadata_path.parent, metadata_df, "asbfly", out_path)
    assert len(df) == 2
    assert set(df.columns) == {"name", "chunk_5s", "embedding"}
    assert out_path.exists()
