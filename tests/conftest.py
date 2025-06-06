import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from birdclef.config import get_species
from birdclef.spark import spark_resource


@pytest.fixture(scope="session")  # Change scope to "session"
def spark(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("spark_data")
    with spark_resource(local_dir=tmp_path.as_posix()) as spark:
        yield spark


"""
Example metadata:
$ gcloud storage cat gs://dsgt-clef-birdclef-2024/data/raw/birdclef-2024/train_metadata.csv | head
primary_label,secondary_labels,type,latitude,longitude,scientific_name,common_name,author,license,rating,url,filename
asbfly,[],['call'],39.2297,118.1987,Muscicapa dauurica,Asian Brown Flycatcher,Matt Slaymaker,Creative Commons Attribution-NonCommercial-ShareAlike 3.0,5.0,https://www.xeno-canto.org/134896,asbfly/XC134896.ogg
asbfly,[],['song'],51.403,104.6401,Muscicapa dauurica,Asian Brown Flycatcher,Magnus Hellström,Creative Commons Attribution-NonCommercial-ShareAlike 3.0,2.5,https://www.xeno-canto.org/164848,asbfly/XC164848.ogg
asbfly,[],['song'],36.3319,127.3555,Muscicapa dauurica,Asian Brown Flycatcher,Stuart Fisher,Creative Commons Attribution-NonCommercial-ShareAlike 4.0,2.5,https://www.xeno-canto.org/175797,asbfly/XC175797.ogg
asbfly,[],['call'],21.1697,70.6005,Muscicapa dauurica,Asian Brown Flycatcher,vir joshi,Creative Commons Attribution-NonCommercial-ShareAlike 4.0,4.0,https://www.xeno-canto.org/207738,asbfly/XC207738.ogg
"""


@pytest.fixture
def species():
    return get_species(species_set="2024")


@pytest.fixture
def metadata_path(tmp_path, species):
    """Subset of the metadata for testing."""
    metadata = tmp_path / "metadata.csv"
    audio = tmp_path / "audio"
    audio.mkdir()
    rows = []
    for i in range(5):
        filename = audio / f"file_{i}.ogg"
        primary_label = species[i]
        rows.append(
            {
                "primary_label": primary_label,
                "filename": filename.relative_to(tmp_path).as_posix(),
            }
        )
        sr = 32_000
        # create 2 channel audio from random noise
        y = np.random.rand(10 * sr)
        sf.write(filename.as_posix(), y, sr)
    df = pd.DataFrame(rows)
    df.to_csv(metadata, index=False)
    return metadata


@pytest.fixture
def metadata_full_path(tmp_path, species):
    """A metadata file, without the associated audio."""
    metadata = tmp_path / "metadata.csv"
    rows = []
    for i, primary_label in enumerate(species):
        filename = f"file_{i}.ogg"
        rows.append(
            {
                "primary_label": primary_label,
                "filename": filename,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(metadata, index=False)
    return metadata


@pytest.fixture
def soundscape_path(tmp_path):
    """Subset of the metadata for testing."""
    soundscape = tmp_path / "soundscape"
    soundscape.mkdir()
    for i in range(5):
        filename = f"file_{i}.ogg"
        sr = 32_000
        # create 2 channel audio from random noise
        y = np.random.rand(10 * sr)
        sf.write(soundscape / filename, y, sr)
    return soundscape
