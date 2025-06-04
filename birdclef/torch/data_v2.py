import torch
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader


class BirdDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_to_idx: dict, embed_dir: str):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx

        # Arrow dataset object for all part_*.parquet files
        self.ds = ds.dataset(embed_dir, format="parquet")

        # identify all embedding columns (everything except "file")
        non_embed_cols = {"file", "start_time", "end_time", "species_name"}
        self.embed_cols = [n for n in self.ds.schema.names if n not in non_embed_cols]

    def __len__(self):
        return len(self.df)

    def _fetch_embedding(self, file_path: str) -> np.ndarray:
        tbl = self.ds.to_table(
            columns=self.embed_cols,
            filter=ds.field("file") == file_path,
        )
        df = tbl.to_pandas()
        return np.asarray(df.iloc[0], dtype=np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        emb = self._fetch_embedding(row["file"])
        label = self.label_to_idx[row["species_name"]]
        return torch.tensor(emb), torch.tensor(label, dtype=torch.long)


class BirdDataModule(pl.LightningDataModule):
    def __init__(self, train_df, test_df, label_to_idx, embed_dir, batch_size=64):
        super().__init__()
        self.train_df = train_df
        self.test_df = test_df
        self.label_to_idx = label_to_idx
        self.embed_dir = embed_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = BirdDataset(
            self.train_df, self.label_to_idx, self.embed_dir
        )
        self.val_dataset = BirdDataset(self.test_df, self.label_to_idx, self.embed_dir)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
