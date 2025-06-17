import multiprocessing as mp
from pathlib import Path

import numpy as np
import polars as pl
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import KeyedVectors
from torch.utils.data import DataLoader, Dataset
import typer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

app = typer.Typer()


class STGTEmbeddingDataset(Dataset):
    def __init__(
        self,
        root: Path,
        wordvector_path: Path,
        split: str = "train",
        mask_prob: float = 0.1,
    ):
        self.split = split
        self.mask_prob = mask_prob

        # filter out any rows where the token length is less than 36, as these are
        # too short to apply masking
        df = pl.read_parquet(root)
        self.wordvectors = KeyedVectors.load(str(wordvector_path))
        self.embedding_dim = self.wordvectors.vector_size

        if split == "train":
            self.df = df.filter(pl.col("part") < 80)
        elif split == "val":
            self.df = df.filter(pl.col("part") >= 80)

    def __len__(self):
        return self.df.select(pl.len()).item()

    def __getitem__(self, idx):
        # Polars row() is slow; direct access is faster for large datasets
        row = self.df[idx]
        tokens = row.get_column("tokens").to_numpy()[0]
        target_logits = row.get_column("logits").to_numpy()[0]

        # make sure there are 40 tokens
        if len(tokens) < 40:
            # we only have 39 or 40 tokens in practice, just make a new array with tokens[-1]
            _tmp = np.zeros(40, tokens.dtype)
            _tmp[: len(tokens)] = tokens
            _tmp[-1] = tokens[-1]
            tokens = _tmp

        # Apply masking during training
        if self.split == "train" and self.mask_prob > 0:
            num_tokens_to_keep = 36
            chosen_indices = sorted(
                np.random.choice(
                    np.arange(len(tokens)), size=num_tokens_to_keep, replace=False
                )
            )
            tokens = tokens[chosen_indices]

        # The model will handle the embedding lookup
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(
            target_logits, dtype=torch.float32
        )


class LitStudentModel(L.LightningModule):
    def __init__(
        self,
        embedding_dim: int,
        output_dim: int,
        wordvectors: KeyedVectors,
        hidden_dim: int = 1024,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["wordvectors"])

        # Embedding layer, initialized with your pretrained vectors
        self.embedding = nn.Embedding.from_pretrained(
            torch.from_numpy(wordvectors.vectors), freeze=True
        )

        # 1D CNN to process the sequence of embeddings
        self.aggregator = nn.Sequential(
            nn.Conv1d(embedding_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.AdaptiveMaxPool1d(1),
        )

        self.transfer_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        )

        # Final classification head
        self.classifier = nn.Linear(hidden_dim, output_dim)

        self.distillation_loss = nn.KLDivLoss(reduction="batchmean")
        self.temperature = 3.0

    def forward(self, tokens):
        # tokens -> embeddings -> cnn -> logits
        embeddings = self.embedding(tokens)  # (batch, seq_len, emb_dim)
        z = embeddings.permute(0, 2, 1)  # (batch, emb_dim, seq_len)
        z = self.aggregator(z).squeeze(-1)  # (batch, 128)
        z = self.transfer_head(z)  # (batch, 128)
        logits = self.classifier(z)
        return logits

    def _step(self, batch):
        student_tokens, teacher_logits = batch
        student_logits = self.forward(student_tokens)

        # KL Divergence for soft label distillation
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)

        loss = (self.temperature**2) * self.distillation_loss(
            student_log_probs, teacher_probs
        )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


@app.command()
def train(
    prefix: str,
    batch_size: int = 32,
    num_workers: int = min(4, mp.cpu_count()),
    logit_dim: int = 10932,
    max_epochs: int = 10,
    resume: bool = typer.Option(
        False,
        "--resume/--no-resume",
        help="Resume training from last checkpoint if available.",
    ),
):
    scratch = Path("~/scratch/birdclef/2025").expanduser()
    root = scratch / "mel2vec-v1"

    wordvector_path = list(
        (root / "word2vec").glob("**/epochs=100/word2vec.wordvectors")
    )[0]
    data_path = scratch / "soundscape-token-perch"

    # Instantiate datasets and dataloaders
    train_ds = STGTEmbeddingDataset(data_path, wordvector_path, split="train")
    val_ds = STGTEmbeddingDataset(data_path, wordvector_path, split="val")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)

    # Instantiate the model
    model = LitStudentModel(
        embedding_dim=train_ds.embedding_dim,
        output_dim=logit_dim,
        wordvectors=train_ds.wordvectors,
    )

    print(model)

    # Output path for checkpoints and logs
    output_path = scratch / f"mel2vec-v1/student-teacher/{prefix}"
    output_path.mkdir(parents=True, exist_ok=True)

    # TensorBoard logger
    logger = TensorBoardLogger(save_dir=output_path, name="logs")

    # ModelCheckpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_path,
        filename="best",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )

    # Find last checkpoint if resume is enabled
    ckpt_path = None
    if resume:
        last_ckpt = output_path / "last.ckpt"
        if last_ckpt.exists():
            ckpt_path = str(last_ckpt)
            print(f"Resuming from checkpoint: {ckpt_path}")

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        callbacks=[checkpoint_callback],
        logger=logger,
        default_root_dir=output_path,
    )
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    app()
