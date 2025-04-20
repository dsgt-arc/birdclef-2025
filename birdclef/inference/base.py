from functools import partial

import torch
from typing import Iterable, Tuple, Optional
import pandas as pd
from tqdm import tqdm


class BaseInference:
    """Class to perform inference on audio files."""

    def predict(
        self, path: str, **kwargs
    ) -> Iterable[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]:
        """Get embeddings and logits for a single audio file.

        :param path: The absolute path to the audio file.
        :param window: The size of the window to split the audio into.
        """
        raise NotImplementedError

    def predict_df(self, root, suffix) -> pd.DataFrame:
        """Embed a single audio file.

        :param root: The root directory of the audio files.
        :param suffix: The filename of the audio file.
        """
        path = f"{root}/{suffix}"
        embeddings = []
        logits = []
        # TODO: need to double check that this still works properly
        for embedding, logit in self.predict(path):
            embeddings.append(embedding)
            if logit is not None:
                logits.append(logit)
        indices = range(len(embeddings))
        df = pd.DataFrame(
            {
                "name": f"{suffix}",
                "chunk_5s": indices,
                "embedding": torch.stack(embeddings).tolist(),
            }
        )
        if logits:
            df["logits"] = torch.stack(logits).tolist()
        return df

    def predict_species_df(
        self,
        root: str,
        metadata: pd.DataFrame,
        species: str,
        out_path: str,
    ) -> pd.DataFrame:
        """Helper function to embed all the training data for a species in the training dataset.

        :param root: The root directory of the audio files.
        :param metadata: The metadata for the audio files.
        :param species: The species to embed.
        :param out_path: The path to save the embeddings.
        """
        tqdm.pandas()
        subset = metadata[metadata["primary_label"] == species]
        dfs = subset.filename.progress_apply(partial(self.predict_df, root)).tolist()
        df = pd.concat(dfs)
        df.to_parquet(out_path, index=False)
        return df
