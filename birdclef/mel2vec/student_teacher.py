import multiprocessing as mp
from functools import partial
from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import typer
from gensim.models import KeyedVectors
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from pacmap import PaCMAP
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
import json
from birdclef.config import colombia_species_list
from birdclef.mel2vec.loaders import get_index, get_pca, get_word_vectors, tokenize

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


# get the token dataset now
def get_token(mfcc, index_path, pca_path=None):
    index = get_index(index_path)
    pca = None
    if pca_path:
        pca = get_pca(pca_path)
    X = np.array(mfcc).reshape(1, -1)
    tokens = tokenize(X, index, pca=pca)
    return tokens[0]


def get_token_dataframe(
    df: pl.LazyFrame,
    tokenizer_path: str | Path,
    group_by=["part", "file", "start_time"],
    num_workers=8,
):
    index_path = Path(tokenizer_path) / "centroids.npy"
    pca_path: Path = Path(tokenizer_path) / "pca.bin"
    if not pca_path.exists():
        pca_path = None

    mfcc_series = df.select("mfcc").collect().get_column("mfcc")
    func = partial(get_token, index_path=index_path, pca_path=pca_path)
    with mp.Pool(num_workers) as pool:
        tokens = pool.map(func, tqdm.tqdm(mfcc_series, desc="Tokenizing MFCCs"))

    return (
        df.with_columns(
            (pl.col("timestamp") // 5 * 5).alias("start_time"),
            pl.Series(tokens).alias("tokens"),
            pl.col("file").str.split("/").list.last().alias("file"),
            "part",
        )
        .group_by(*group_by)
        .agg(pl.col("tokens").sort_by("timestamp").alias("tokens"))
        .sort("file", "start_time")
    )


@app.command()
def generate_token_perch(
    mfcc_path: Path = typer.Option(
        "~/scratch/birdclef/2025v2/mfcc-soundscape/data",
        help="Path to MFCC parquet data.",
    ),
    perch_path: Path = typer.Option(
        "~/shared/birdclef/2025/infer-soundscape/Perch/parts/predict/*.parquet",
        help="Glob path to Perch prediction parquet files.",
    ),
    tokenizer_path: Path = typer.Option(
        "~/scratch/birdclef/2025v2/mel2vec-v2/tokenizer_pca/n_clusters=16383",
        help="Path to centroids/pca",
    ),
    output_path: Path = typer.Option(
        "~/scratch/birdclef/2025v2/mel2vec-v2/soundscape-token-perch",
        help="Output directory for tokenized parquet.",
    ),
):
    mfcc_path = Path(mfcc_path).expanduser()
    perch_path = Path(perch_path).expanduser()
    tokenizer_path = Path(tokenizer_path).expanduser()
    output_path = Path(output_path).expanduser()

    perch = pl.scan_parquet(str(perch_path))
    columns = perch.collect_schema().names()
    perch = perch.select(
        pl.col("file").str.split("/").list.last().alias("file"),
        "start_time",
        "end_time",
        (pl.concat_list(columns[3:]).list.to_array(len(columns[3:])).alias("logits")),
    ).sort("file", "start_time")

    get_token_dataframe(
        df=pl.scan_parquet(str(mfcc_path)).sort("file", "timestamp"),
        tokenizer_path=str(tokenizer_path),
    ).join(
        perch,
        on=["file", "start_time"],
        how="left",
    ).select(
        "part",
        "file",
        "start_time",
        "tokens",
        "logits",
    ).collect().write_parquet(str(output_path), use_pyarrow=True, partition_by=["part"])


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
    root: Path = typer.Option(
        "~/scratch/birdclef/2025/mel2vec-v1",
        help="Root directory for the mel2vec dataset.",
    ),
    data_path: Path = typer.Option(
        "~/scratch/birdclef/2025/soundscape-token-perch",
    ),
):
    root = Path(root).expanduser().resolve()
    wordvector_path = list(
        (root / "word2vec").glob("**/epochs=100/word2vec.wordvectors")
    )[0]
    data_path = Path(data_path).expanduser().resolve()

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
    output_path = Path(f"{root}/{prefix}").expanduser().resolve()
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


def plot_pacmap(X, y, le, title):
    z = PaCMAP().fit_transform(X)
    for i, species in enumerate(le.classes_):
        plt.scatter(z[y == i, 0], z[y == i, 1], label=species, alpha=0.5, s=1)
    plt.legend()
    plt.xlabel("PaCMAP 1")
    plt.ylabel("PaCMAP 2")
    plt.title(title)


@app.command()
def evaluate(
    scratch: Path = typer.Option(
        "~/scratch/birdclef/2025v2",
        help="Scratch directory containing data and models.",
    ),
    wordvector_prefix_path: Path = typer.Option(
        "~/scratch/birdclef/2025v2/mel2vec-v2/word2vec",
        help="Path to the word2vec directory.",
    ),
    tokenizer_path: Path = typer.Option(
        "~/scratch/birdclef/2025v2/mel2vec-v2/tokenizer_pca/n_clusters=16383",
        help="Path to centroids/pca.",
    ),
    ckpt_dir: Path = typer.Option(
        "~/scratch/birdclef/2025v2/mel2vec-v2/student-teacher",
        help="Directory containing the student-teacher checkpoint.",
    ),
    num_workers: int = typer.Option(
        8,
        help="Number of workers for parallel processing.",
    ),
):
    scratch = Path(scratch).expanduser()
    tokenizer_path = Path(tokenizer_path).expanduser()
    ckpt_dir = Path(ckpt_dir).expanduser()

    mfcc_df = (
        pl.scan_parquet(f"{scratch}/mfcc-train/data", low_memory=True)
        .with_columns(
            pl.col("file").str.split("/").list.get(-2).alias("species"),
        )
        .filter(pl.col("species").is_in(colombia_species_list))
    )

    tokenized = get_token_dataframe(
        mfcc_df,
        tokenizer_path,
        group_by=["species", "file", "start_time"],
        num_workers=num_workers,
    ).select("species", "tokens")

    df = tokenized.collect().to_pandas()
    wordvector_path = list(
        Path(wordvector_prefix_path).glob("**/epochs=100/word2vec.wordvectors")
    )[0].as_posix()
    wordvectors = get_word_vectors(wordvector_path)

    model = LitStudentModel.load_from_checkpoint(
        (ckpt_dir / "best.ckpt").as_posix(),
        wordvectors=wordvectors,
    )

    # replace the classifier with an identity function
    model.classifier = nn.Identity()
    model.eval()

    clean = df[df.tokens.apply(len) == 40]
    tokens = np.stack(clean.tokens.values)
    with torch.no_grad():
        X = model(torch.from_numpy(tokens)).cpu().numpy()

    y = clean["species"].values
    le = LabelEncoder()
    y = le.fit_transform(y)

    # stratify the split by species
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    scores = {
        "f1_macro_score": f1_score(y_test, y_pred, average="macro"),
        "f1_micro_score": f1_score(y_test, y_pred, average="micro"),
        "accuracy": clf.score(X_test, y_test),
        # "roc_auc": roc_auc_score(
        #     y_test,
        #     clf.predict_proba(X_test),
        #     multi_class="ovr",
        #     labels=sorted(
        #         set(le.inverse_transform(y_pred)) | set(le.inverse_transform(y_test))
        #     ),
        # ),
    }
    output_root = ckpt_dir / "evaluation"
    output_root.mkdir(parents=True, exist_ok=True)
    print(scores)
    with open(output_root / "scores.json", "w") as f:
        json.dump(scores, f, indent=2)

    report = classification_report(
        y_test, y_pred, labels=np.arange(len(le.classes_)), target_names=le.classes_
    )
    print(report)
    with open(output_root / "classification_report.txt", "w") as f:
        f.write(report)

    plot_pacmap(X, y, le, "PaCMAP of Student-Teacher Embeddings")
    plt.savefig(output_root / "pacmap.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    app()
