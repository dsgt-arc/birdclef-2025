import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader


class BirdDataset(Dataset):
    def __init__(self, X, y, label_to_idx):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor([label_to_idx[label] for label in y], dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class BirdDataModule(pl.LightningDataModule):
    def __init__(self, X_train, X_test, y_train, y_test, label_to_idx, batch_size=64):
        super().__init__()
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.label_to_idx = label_to_idx
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = BirdDataset(self.X_train, self.y_train, self.label_to_idx)
        self.val_dataset = BirdDataset(self.X_test, self.y_test, self.label_to_idx)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
