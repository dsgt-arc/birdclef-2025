import json
from pathlib import Path

import faiss
import luigi
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import typer
from contexttimer import Timer
from gensim.models import Word2Vec
from pacmap import PaCMAP
from pyspark.sql import functions as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from birdclef.config import colombia_species_list
from birdclef.spark import get_spark

from . import loaders
from .callback import TqdmCallback

app = typer.Typer()


class OptionsMixin:
    input_root = luigi.Parameter(
        description="Directory containing audio files to process",
    )
    output_root = luigi.Parameter(
        description="Directory to save the output files",
    )


class BuildTokenizer(luigi.Task, OptionsMixin):
    input_dim = luigi.IntParameter(default=20)
    n_clusters = luigi.IntParameter(default=2**14 - 1)

    prefix = "tokenizer"

    def output(self):
        return {
            "centroids": luigi.LocalTarget(
                f"{self.output_root}/{self.prefix}/n_clusters={self.n_clusters}/centroids.npy"
            )
        }

    def _load_data(self):
        """Load the data from the input root directory."""
        df = (
            pl.scan_parquet(self.input_root)
            .filter(pl.col("part") < 80)
            .sort("file", "timestamp")
            .select("file", "timestamp", "mfcc")
        )
        return df

    def _prepare_matrix(self, df):
        """Prepare the matrix of MFCC features from the DataFrame."""
        X = np.stack(df.select("mfcc").collect().get_column("mfcc").to_numpy())
        X = X.astype(np.float32)
        return X

    def _save_centroids(self, cluster_faiss):
        """Save the centroids to the output directory."""
        output = Path(self.output()["centroids"].path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, cluster_faiss.centroids)

    def run(self):
        # use the first 80% of the data for training
        df = self._load_data()
        X = self._prepare_matrix(df)
        cluster_faiss = faiss.Kmeans(
            d=self.input_dim,
            k=self.n_clusters,
            niter=25,
            verbose=True,
        )
        cluster_faiss.train(X)
        self._save_centroids(cluster_faiss)


class BuildPCATokenizer(BuildTokenizer):
    prefix = "tokenizer_pca"

    def output(self):
        return {
            "centroids": luigi.LocalTarget(
                f"{self.output_root}/{self.prefix}/n_clusters={self.n_clusters}/centroids.npy"
            ),
            "pca": luigi.LocalTarget(
                f"{self.output_root}/{self.prefix}/n_clusters={self.n_clusters}pca.bin"
            ),
        }

    def _save_pca(self, pca):
        """Save the PCA model to the output directory."""
        output = Path(self.output()["pca"].path)
        output.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_VectorTransform(pca, output.as_posix())

    def run(self):
        # use the first 80% of the data for training
        df = self._load_data()
        X = self._prepare_matrix(df)

        pca = faiss.PCAMatrix(self.input_dim, self.input_dim)
        pca.train(X)

        cluster_faiss = faiss.Kmeans(
            d=self.input_dim,
            k=self.n_clusters,
            niter=25,
            verbose=True,
        )
        cluster_faiss.train(pca.apply(X))

        self._save_centroids(cluster_faiss)
        self._save_pca(pca)


class Word2VecOptionsMixin(OptionsMixin):
    vector_size = luigi.IntParameter(default=256)
    window = luigi.IntParameter(default=40)
    ns_exponent = luigi.FloatParameter(default=0.75)
    sample = luigi.FloatParameter(default=1e-3)
    workers = luigi.IntParameter(default=8)
    epochs = luigi.IntParameter(default=100)
    tokenizer = luigi.ChoiceParameter(
        default="tokenizer",
        choices=["tokenizer", "tokenizer_pca"],
        description="The tokenizer to use for training the Word2Vec model",
    )
    tokenizer_n_clusters = luigi.IntParameter(
        default=2**14 - 1,
        description="Number of clusters to use for the tokenizer",
    )


class Word2VecTask(luigi.Task, Word2VecOptionsMixin):
    """Task to train a Word2Vec model on a specific set of audio files."""

    def requires(self):
        return {
            "tokenizer": BuildTokenizer(
                input_root=self.input_root,
                output_root=self.output_root,
                n_clusters=self.tokenizer_n_clusters,
            ),
            "tokenizer_pca": BuildPCATokenizer(
                input_root=self.input_root,
                output_root=self.output_root,
                n_clusters=self.tokenizer_n_clusters,
            ),
        }[self.tokenizer]

    def output(self):
        prefix = "/".join(
            f"{k}={v}"
            for k, v in [
                ("tokenizer", self.tokenizer),
                ("tokenizer_n_clusters", self.tokenizer_n_clusters),
                ("vector_size", self.vector_size),
                ("window", self.window),
                ("ns_exponent", self.ns_exponent),
                ("sample", self.sample),
                ("epochs", self.epochs),
            ]
        )
        return {
            "model": luigi.LocalTarget(
                f"{self.output_root}/word2vec/{prefix}/word2vec.model"
            ),
            "wordvectors": luigi.LocalTarget(
                f"{self.output_root}/word2vec/{prefix}/word2vec.wordvectors"
            ),
            "timing": luigi.LocalTarget(
                f"{self.output_root}/word2vec/{prefix}/timing.json"
            ),
        }

    def token_generator(self, df, limit=-1):
        if limit > 0:
            df = df.filter(pl.col("part") < limit)
        for sub in df.collect().partition_by("file"):
            yield sub.sort("timestamp").get_column("token").to_list()

    def run(self):
        centroids = np.load(self.requires().output()["centroids"].path)
        index = faiss.IndexFlatL2(centroids.shape[1])
        index.add(centroids)

        df = (
            pl.scan_parquet(self.input_root)
            .filter(pl.col("part") < 80)
            .sort("file", "timestamp")
        )

        X = np.stack(df.select("mfcc").collect().get_column("mfcc").to_numpy())
        if self.tokenizer == "tokenizer_pca":
            pca = faiss.read_VectorTransform(self.requires().output()["pca"].path)
            X = pca.apply(X)
        X = X.astype(np.float32)
        _, indices = index.search(X, 1)
        ids = pl.Series("token", indices.flatten())
        token_df = df.with_columns(ids)

        with Timer() as t:
            model = Word2Vec(
                sentences=list(self.token_generator(token_df)),
                epochs=self.epochs,
                vector_size=self.vector_size,
                # 5 seconds, 8 frames per second = 40
                # can go to 10 seconds to have more context
                min_count=1,
                window=self.window,
                sg=1,
                negative=5,
                ns_exponent=self.ns_exponent,
                sample=self.sample,
                workers=self.workers,
                compute_loss=True,
                shrink_windows=True,
                callbacks=[TqdmCallback(total_epochs=self.epochs)],
            )
        # ensure folder
        output_dir = Path(self.output()["model"].path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save(self.output()["model"].path)
        model.wv.save(self.output()["wordvectors"].path)
        with open(self.output()["timing"].path, "w") as f:
            json.dump(
                {
                    "time": t.elapsed,
                    "epochs": self.epochs,
                    "vector_size": self.vector_size,
                    "window": self.window,
                    "ns_exponent": self.ns_exponent,
                    "sample": self.sample,
                    "workers": self.workers,
                },
                f,
            )


class EmbedWord2VecOptionsMixin(Word2VecOptionsMixin):
    input_root = luigi.Parameter(
        description="Directory containing audio files to process",
    )
    soundscape_root = luigi.Parameter(
        description="Directory containing soundscape files to train Word2Vec",
    )
    output_root = luigi.Parameter(
        description="Directory to save the output files",
    )
    output_prefix = luigi.Parameter(
        default="train",
        description="Prefix for the output files",
    )
    filter_species = luigi.ListParameter(
        default=[],
        description="List of species to filter on",
    )


class EmbedWord2VecTask(luigi.Task, EmbedWord2VecOptionsMixin):
    """Task to embed audio files using the trained Word2Vec model.

    We should be using the soundscape dataset to train the word2vec model.
    We'll want to embed the actual mfccs on the training dataset though.
    """

    def output(self):
        prefix = "/".join(
            f"{k}={v}"
            for k, v in [
                ("tokenizer", self.tokenizer),
                ("tokenizer_n_clusters", self.tokenizer_n_clusters),
                ("vector_size", self.vector_size),
                ("window", self.window),
                ("ns_exponent", self.ns_exponent),
                ("sample", self.sample),
                ("epochs", self.epochs),
            ]
        )
        return luigi.LocalTarget(
            f"{self.output_root}/embedding/{self.output_prefix}/{prefix}"
        )

    def requires(self):
        word2vec = Word2VecTask(
            input_root=self.soundscape_root,
            output_root=self.output_root,
            epochs=self.epochs,
            vector_size=self.vector_size,
            window=self.window,
            ns_exponent=self.ns_exponent,
            sample=self.sample,
            workers=self.workers,
            tokenizer=self.tokenizer,
            tokenizer_n_clusters=self.tokenizer_n_clusters,
        )
        return {
            "word2vec": word2vec,
            "tokenizer": word2vec.requires(),
        }

    def run(self):
        with get_spark() as spark:

            @F.udf(returnType="integer")
            def get_start_time(timestamp, interval=5) -> int:
                # up to but not including the value
                for i in range(0, 1000, interval):
                    if i <= timestamp < i + interval:
                        return i
                return -1

            # yeah kind of gross
            wv_path = self.requires()["word2vec"].output()["wordvectors"].path
            index_path = self.requires()["tokenizer"].output()["centroids"].path
            if self.tokenizer == "tokenizer_pca":
                pca_path = self.requires()["tokenizer"].output()["pca"].path
            else:
                pca_path = None

            @F.udf(returnType="array<float>")
            def mfcc_to_wv(
                mfcc: list,
                wv_path: str = wv_path,
                index_path: str = index_path,
                pca_path: str | None = pca_path,
            ) -> list:
                # convert mfcc to word vectors
                X = np.array(mfcc).reshape(1, -1)
                if self.tokenizer == "tokenizer_pca":
                    # unfortunately we can't serialize PCA so reread it from disk every time,
                    pca = loaders.get_pca(pca_path)
                    X = pca.apply(X)
                index = loaders.get_index(index_path)
                word_vectors = loaders.get_word_vectors(wv_path)
                _, indices = index.search(X, 1)
                return word_vectors[indices[0][0]].tolist()

            @F.udf(returnType="array<float>")
            def get_mfcc_stats(mfcc: list) -> list:
                X = np.stack(mfcc)
                return X.mean(axis=0).tolist() + X.std(axis=0).tolist()

            @F.udf(returnType="array<float>")
            def avg_vector(vectors: list) -> list:
                return np.mean(np.array(vectors), axis=0).tolist()

            df = spark.read.parquet(self.input_root)
            if "train" in self.input_root and self.filter_species:
                df = (
                    df.withColumn(
                        "species", F.udf(lambda x: Path(x).parts[-2], "string")("file")
                    )
                    .where(
                        F.col("species").isin([F.lit(x) for x in self.filter_species])
                    )
                    .drop("species")
                )

            (
                df.withColumn("start_time", get_start_time(F.col("timestamp")))
                .withColumn("word_vector", mfcc_to_wv(F.col("mfcc")))
                .groupBy("file", "start_time")
                .agg(
                    F.collect_list("mfcc").alias("mfcc"),
                    F.collect_list("word_vector").alias("word_vector"),
                )
                .where(F.col("start_time") >= 0)
                .withColumn("mfcc_stats", get_mfcc_stats(F.col("mfcc")))
                .withColumn("word_vector", avg_vector(F.col("word_vector")))
                .select("file", "start_time", "mfcc_stats", "word_vector")
                .repartition(20)
                .write.parquet(self.output().path, mode="overwrite"),
            )


class EvalWord2VecTask(luigi.Task, EmbedWord2VecOptionsMixin):
    """Task to evaluate word2vec model using logistic regression on the training set.

    On perch, we get a model that gets a f1 score of 0.85 on our list of species.
    """

    def requires(self):
        return {
            "embed": EmbedWord2VecTask(
                input_root=self.input_root,
                soundscape_root=self.soundscape_root,
                output_root=self.output_root,
                output_prefix="train",
                filter_species=self.filter_species,
                tokenizer=self.tokenizer,
                tokenizer_n_clusters=self.tokenizer_n_clusters,
                vector_size=self.vector_size,
                window=self.window,
                ns_exponent=self.ns_exponent,
                sample=self.sample,
                epochs=self.epochs,
            ),
        }

    def output(self):
        # get the output prefix from the embed task, and replace embedding with logistic
        output_root = Path(
            self.requires()["embed"].output().path.replace("embedding", "logistic", 1)
        )
        return {
            "scores": luigi.LocalTarget(f"{output_root}/scores.json"),
            "classification_report": luigi.LocalTarget(
                f"{output_root}/classification_report.txt"
            ),
            "plot": luigi.LocalTarget(f"{output_root}/plot.png"),
        }

    def run(self):
        # note, the mfcc one only needs to be run once, so here is really just the
        # word2vec performance
        embed_output_path = self.requires()["embed"].output().path
        df = (
            pl.read_parquet(f"{embed_output_path}/**/*.parquet")
            .sort("file", "start_time")
            .with_columns(species=pl.col("file").str.split("/").list.get(-2))
            .filter(pl.col("species").is_in(colombia_species_list))
            .select("species", pl.col("word_vector").alias("embedding"))
            .to_pandas()
        )
        X = np.stack(df["embedding"].values)
        y = df["species"].values
        le = LabelEncoder()
        y = le.fit_transform(y)
        # stratify the split by species
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        model = LogisticRegression(max_iter=1000, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores = {
            "f1_macro_score": f1_score(y_test, y_pred, average="macro"),
            "f1_micro_score": f1_score(y_test, y_pred, average="micro"),
            "accuracy": model.score(X_test, y_test),
        }
        output_root = Path(self.output()["scores"].path).parent
        output_root.mkdir(parents=True, exist_ok=True)
        with open(self.output()["scores"].path, "w") as f:
            json.dump(scores, f)

        report = classification_report(y_test, y_pred, target_names=le.classes_)
        with open(self.output()["classification_report"].path, "w") as f:
            f.write(report)

        # plot the first two components
        z = PaCMAP().fit_transform(X)
        for i, species in enumerate(le.classes_):
            plt.scatter(z[y == i, 0], z[y == i, 1], label=species, alpha=0.5, s=1)
        plt.legend()
        plt.xlabel("PaCMAP 1")
        plt.ylabel("PaCMAP 2")
        plt.title("PaCMAP projection of Word2Vec embeddings")
        plt.savefig(self.output()["plot"].path)


@app.command()
def run(
    train_root: str,
    soundscape_root: str,
    output_root: str,
    gensim_workers: int = 8,
    luigi_workers: int = 8,
):
    """Run the tokenizer building process.

    Note that the inputs for train and soundscape roots as of writing of this comment
    is for these to be the pre-computed MFCCs, and not the raw audio.
    """
    luigi.build(
        [
            EmbedWord2VecTask(
                input_root=input_root,
                soundscape_root=soundscape_root,
                output_root=output_root,
                output_prefix=output_prefix,
                workers=gensim_workers,
                tokenizer=tokenizer,
                **params,
            )
            for tokenizer in ["tokenizer", "tokenizer_pca"]
            for input_root, output_prefix in [
                (train_root, "train"),
                (soundscape_root, "soundscape"),
            ]
            for params in [
                {
                    "epochs": 100,
                    "vector_size": 256,
                    "window": 80,
                    "ns_exponent": 0.75,
                    "sample": 1e-4,
                }
            ]
        ],
        workers=luigi_workers,
        local_scheduler=True,
    )


@app.command()
def tune_tokenizer(
    train_root: str,
    soundscape_root: str,
    output_root: str,
    gensim_workers: int = 8,
    luigi_workers: int = 8,
):
    """Run the tokenizer building process.

    Note that the inputs for train and soundscape roots as of writing of this comment
    is for these to be the pre-computed MFCCs, and not the raw audio.
    """
    luigi.build(
        [
            EvalWord2VecTask(
                input_root=input_root,
                soundscape_root=soundscape_root,
                output_root=output_root,
                output_prefix=output_prefix,
                workers=gensim_workers,
                tokenizer=tokenizer,
                tokenizer_n_clusters=tokenizer_n_clusters,
                filter_species=colombia_species_list,
                **params,
            )
            for tokenizer in ["tokenizer"]
            # 4k, 8k, 16k, 32k, 64k
            for tokenizer_n_clusters in [2**12, 2**13, 2**14 - 1, 2**15 - 1, 2**16 - 1]
            for input_root, output_prefix in [(train_root, "train")]
            for params in [
                {
                    "epochs": 20,
                    "vector_size": 256,
                    "window": 80,
                    "ns_exponent": 0.75,
                    "sample": 1e-4,
                }
            ]
        ],
        workers=luigi_workers,
        local_scheduler=True,
    )


@app.command()
def tune_w2v(
    train_root: str,
    soundscape_root: str,
    output_root: str,
    gensim_workers: int = 8,
    luigi_workers: int = 8,
):
    """Tune the Word2Vec model parameters."""
    luigi.build(
        [
            EvalWord2VecTask(
                input_root=input_root,
                soundscape_root=soundscape_root,
                output_root=output_root,
                output_prefix=output_prefix,
                workers=gensim_workers,
                tokenizer=tokenizer,
                tokenizer_n_clusters=tokenizer_n_clusters,
                filter_species=colombia_species_list,
                epochs=20,
                **params,
            )
            for tokenizer in ["tokenizer"]
            for tokenizer_n_clusters in [2**14 - 1]
            for input_root, output_prefix in [(train_root, "train")]
            for params in [
                # Baseline Configuration
                {
                    "vector_size": 256,
                    "window": 80,
                    "ns_exponent": 0.75,
                    "sample": 1e-4,
                },
                # Varying Vector Size
                {
                    "vector_size": 128,
                    "window": 80,
                    "ns_exponent": 0.75,
                    "sample": 1e-4,
                },
                {
                    "vector_size": 384,
                    "window": 80,
                    "ns_exponent": 0.75,
                    "sample": 1e-4,
                },
                # Varying Window Size
                {
                    "vector_size": 256,
                    "window": 40,
                    "ns_exponent": 0.75,
                    "sample": 1e-4,
                },
                {
                    "vector_size": 256,
                    "window": 120,
                    "ns_exponent": 0.75,
                    "sample": 1e-4,
                },
                # Varying Negative Sampling Exponent (Key for imbalance)
                {
                    "vector_size": 256,
                    "window": 80,
                    "ns_exponent": 0.0,
                    "sample": 1e-4,
                },
                {
                    "vector_size": 256,
                    "window": 80,
                    "ns_exponent": -0.5,
                    "sample": 1e-4,
                },
                # Varying Downsampling Rate
                {
                    "vector_size": 256,
                    "window": 80,
                    "ns_exponent": 0.75,
                    "sample": 1e-5,
                },
                # Combination Experiment: Optimized for Rare Species
                {
                    "vector_size": 256,
                    "window": 120,
                    "ns_exponent": -0.5,
                    "sample": 1e-4,
                },
            ]
        ],
        workers=luigi_workers,
        local_scheduler=True,
    )


if __name__ == "__main__":
    import multiprocessing

    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    app()
