from birdclef.etl.embed.birdnet.workflow import embed_soundscapes


def test_etl_embed_birdnet_workflow(tmp_path, soundscape_path):
    embed_soundscapes(
        audio_path=str(soundscape_path),
        intermediate_path=str(tmp_path / "intermediate"),
        output_path=str(tmp_path / "output"),
        total_batches=2,
        num_partitions=1,
    )
    assert (tmp_path / "output").exists()
    # count the number of parquet files
    assert len(list((tmp_path / "output").glob("*.parquet"))) == 1
    # check that there is a success file
    assert (tmp_path / "output" / "_SUCCESS").exists()
