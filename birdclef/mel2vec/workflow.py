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
from sklearn.metrics import roc_auc_score, f1_score, classification_report
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


class BuildTokenizerOptionsMixin:
    input_dim = luigi.IntParameter(default=768)
    n_clusters = luigi.IntParameter(default=2**14 - 1)
    feature_column = luigi.Parameter(
        default="mfcc",
        description="The feature column to use for clustering",
    )
    kmeans_niter = luigi.IntParameter(
        default=10, description="Number of KMeans iterations"
    )
    pca_dim = luigi.IntParameter(
        default=128,
        description="Dimension of the PCA transformation to apply before clustering",
    )


class BuildTokenizer(luigi.Task, OptionsMixin, BuildTokenizerOptionsMixin):
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
            .select("file", "timestamp", self.feature_column)
        )
        return df

    def _prepare_matrix(self, df):
        """Prepare the matrix of spectrogram features from the DataFrame.

        The feature column can be MFCC or melspectrogram vectors.
        """
        X = np.stack(
            df.select(self.feature_column)
            .collect()
            .get_column(self.feature_column)
            .to_numpy()
        )
        return X

    def _normalize_matrix(self, X):
        """Normalize the matrix of spectrogram features."""
        # perform per-sample l2 normalization with small episilon for numerical stability
        # NOTE: this makes the tokenizer incompatible with previous versions
        X = X.astype(np.float32)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        return X

    def _save_centroids(self, cluster_faiss):
        """Save the centroids to the output directory."""
        output = Path(self.output()["centroids"].path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, cluster_faiss.centroids)

    def run(self):
        # use the first 80% of the data for training
        # The feature column can be MFCC or melspectrogram vectors.
        df = self._load_data()
        X = self._prepare_matrix(df)
        X = self._normalize_matrix(X)
        cluster_faiss = faiss.Kmeans(
            d=self.input_dim,
            k=self.n_clusters,
            niter=self.kmeans_niter,
            verbose=True,
        )
        cluster_faiss.train(X)
        self._save_centroids(cluster_faiss)


class BuildPCATokenizer(BuildTokenizer, BuildTokenizerOptionsMixin):
    prefix = "tokenizer_pca"

    def output(self):
        return {
            "centroids": luigi.LocalTarget(
                f"{self.output_root}/{self.prefix}/n_clusters={self.n_clusters}/centroids.npy"
            ),
            "pca": luigi.LocalTarget(
                f"{self.output_root}/{self.prefix}/n_clusters={self.n_clusters}/pca.bin"
            ),
        }

    def _save_pca(self, pca):
        """Save the PCA model to the output directory."""
        output = Path(self.output()["pca"].path)
        output.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_VectorTransform(pca, output.as_posix())

    def run(self):
        df = self._load_data()
        X = self._prepare_matrix(df)
        pca = faiss.PCAMatrix(self.input_dim, self.pca_dim)
        pca.train(X)

        cluster_faiss = faiss.Kmeans(
            d=self.pca_dim,
            k=self.n_clusters,
            niter=self.kmeans_niter,
            verbose=True,
        )
        X = pca.apply(X)
        X = self._normalize_matrix(X)
        cluster_faiss.train(X)

        self._save_centroids(cluster_faiss)
        self._save_pca(pca)


class Word2VecOptionsMixin(OptionsMixin, BuildTokenizerOptionsMixin):
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
    step = luigi.IntParameter(default=10, description="Number of epochs per checkpoint")


class Word2VecTask(luigi.Task, Word2VecOptionsMixin):
    """Task to train a Word2Vec model on a specific set of audio files.

    The feature column can be MFCC or melspectrogram vectors.
    """

    def requires(self):
        reqs = {
            "tokenizer": BuildTokenizer(
                input_root=self.input_root,
                output_root=self.output_root,
                n_clusters=self.n_clusters,
                kmeans_niter=self.kmeans_niter,
            ),
            "tokenizer_pca": BuildPCATokenizer(
                input_root=self.input_root,
                output_root=self.output_root,
                n_clusters=self.n_clusters,
                kmeans_niter=self.kmeans_niter,
            ),
        }
        # recursive dependency for checkpointing.
        # the base epoch size is the step size.
        if self.epochs > self.step:
            return {
                "tokenizer": reqs[self.tokenizer],
                "prev": self.clone(
                    epochs=self.epochs - self.step,
                ),
            }
        else:
            return {"tokenizer": reqs[self.tokenizer]}

    def output(self):
        prefix = "/".join(
            f"{k}={v}"
            for k, v in [
                ("tokenizer", self.tokenizer),
                # NOTE: this is no longer backwards compatible, and it also means that
                # the first job with n_iters is the first one we get.
                ("n_clusters", self.n_clusters),
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
        # Prepare centroids and index for tokenization
        index = loaders.get_index(
            self.requires()["tokenizer"].output()["centroids"].path
        )
        if self.tokenizer == "tokenizer_pca":
            pca = loaders.get_pca(self.requires()["tokenizer"].output()["pca"].path)
        else:
            pca = None

        if limit > 0:
            df = df.filter(pl.col("part") < limit)
        # Use lazy evaluation: process each partition/file as needed
        for sub in df.collect().partition_by("file"):
            features = np.stack(
                sub.sort("timestamp").get_column(self.feature_column).to_numpy()
            )
            yield loaders.tokenize(features, index, pca)

    def run(self):
        df = (
            pl.scan_parquet(self.input_root)
            .filter(pl.col("part") < 80)
            .sort("file", "timestamp")
        )

        # Remove eager tokenization, use token_generator for lazy tokenization
        with Timer() as t:
            if "prev" not in self.requires():
                model = Word2Vec(
                    sentences=list(self.token_generator(df)),
                    epochs=self.step,
                    vector_size=self.vector_size,
                    min_count=1,
                    window=self.window,
                    sg=1,
                    negative=5,
                    ns_exponent=self.ns_exponent,
                    sample=self.sample,
                    workers=self.workers,
                    compute_loss=True,
                    shrink_windows=True,
                    callbacks=[TqdmCallback(total_epochs=self.step)],
                )
            else:
                # continue training from previous checkpoint
                model = Word2Vec.load(self.requires()["prev"].output()["model"].path)
                model.train(
                    list(self.token_generator(df)),
                    total_examples=model.corpus_count,
                    epochs=self.step,
                    start_alpha=model.alpha,
                    end_alpha=model.min_alpha,
                    compute_loss=True,
                    callbacks=[TqdmCallback(total_epochs=self.step)],
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

    The feature column can be MFCC or melspectrogram vectors.
    We should be using the soundscape dataset to train the word2vec model.
    We'll want to embed the actual feature vectors on the training dataset though.
    """

    def output(self):
        prefix = "/".join(
            f"{k}={v}"
            for k, v in [
                ("tokenizer", self.tokenizer),
                ("tokenizer_n_clusters", self.n_clusters),
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
            n_clusters=self.n_clusters,
            kmeans_niter=self.kmeans_niter,
        )
        return {
            "word2vec": word2vec,
            "tokenizer": word2vec.requires()["tokenizer"],
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
                # convert feature vector (MFCC or melspectrogram) to word vectors
                X = np.array(mfcc).reshape(1, -1)

                index = loaders.get_index(index_path)
                pca = loaders.get_pca(pca_path) if pca_path else None
                token = loaders.tokenize(X, index, pca)[0]
                word_vectors = loaders.get_word_vectors(wv_path)
                if token not in word_vectors:
                    # if the token is not found, return a zero vector
                    return [0.0] * len(word_vectors[0])
                # return the word vector for the token
                return word_vectors[token].tolist()

            @F.udf(returnType="array<float>")
            def get_mfcc_stats(mfcc: list) -> list:
                # Compute mean and std of feature vectors (MFCC or melspectrogram)
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
                .withColumn("word_vector", mfcc_to_wv(F.col(self.feature_column)))
                .groupBy("file", "start_time")
                .agg(
                    F.collect_list(self.feature_column).alias(self.feature_column),
                    F.collect_list("word_vector").alias("word_vector"),
                )
                .where(F.col("start_time") >= 0)
                .withColumn("mfcc_stats", get_mfcc_stats(F.col(self.feature_column)))
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
                n_clusters=self.n_clusters,
                vector_size=self.vector_size,
                window=self.window,
                ns_exponent=self.ns_exponent,
                sample=self.sample,
                epochs=self.epochs,
                kmeans_niter=self.kmeans_niter,
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
            "roc_auc": roc_auc_score(
                y_test, model.predict_proba(X_test), multi_class="ovr"
            ),
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
def tune_tokenizer(
    train_root: str,
    soundscape_root: str,
    output_root: str,
    gensim_workers: int = 8,
    luigi_workers: int = 8,
):
    """Run the tokenizer building process.

    Note that the inputs for train and soundscape roots as of writing of this comment
    is for these to be the pre-computed MFCC or melspectrogram vectors, and not the raw audio.
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
                n_clusters=n_clusters,
                filter_species=colombia_species_list,
                **params,
            )
            for tokenizer in ["tokenizer", "tokenizer_pca"]
            # 4k, 8k, 16k, 32k
            for n_clusters in [2**12, 2**13, 2**14 - 1, 2**15 - 1]
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
    baseline = {
        "vector_size": 384,
        "window": 80,
        "ns_exponent": 1.5,
        "sample": 1e-5,
    }
    experiments = [baseline]
    for vector_size in [128, 256, 512, 1028]:
        experiments.append({**baseline, "vector_size": vector_size})
    for window in [40, 120]:
        experiments.append({**baseline, "window": window})
    for ns_exponent in [0.0, -0.5]:
        experiments.append({**baseline, "ns_exponent": ns_exponent})
    for sample in [1e-4, 1e-6]:
        experiments.append({**baseline, "sample": sample})
    # Combination experiment: Optimized for rare species
    experiments.append({**baseline, "window": 120, "ns_exponent": -0.5})
    luigi.build(
        [
            EvalWord2VecTask(
                input_root=input_root,
                soundscape_root=soundscape_root,
                output_root=output_root,
                output_prefix=output_prefix,
                workers=gensim_workers,
                tokenizer=tokenizer,
                n_clusters=n_clusters,
                filter_species=colombia_species_list,
                epochs=20,
                kmeans_niter=10,
                **params,
            )
            for tokenizer in ["tokenizer", "tokenizer_pca"]
            for n_clusters in [2**14 - 1]
            for input_root, output_prefix in [(train_root, "train")]
            for params in experiments
        ],
        workers=luigi_workers,
        local_scheduler=True,
    )


@app.command()
def tune_ns(
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
                n_clusters=n_clusters,
                filter_species=colombia_species_list,
                epochs=20,
                **params,
            )
            for tokenizer in ["tokenizer"]
            for n_clusters in [2**14 - 1]
            for input_root, output_prefix in [(train_root, "train")]
            for params in [
                {
                    "vector_size": 384,
                    "window": 80,
                    "ns_exponent": ns_exponent,
                    "sample": 1e-5,
                }
                for ns_exponent in [
                    -1.5,
                    -1.25,
                    -1.0,
                    -0.75,
                    -0.5,
                    -0.25,
                    0.0,
                    0.25,
                    0.5,
                    0.75,
                    1.0,
                    1.25,
                    1.5,
                    1.75,
                    2.0,
                    2.5,
                ]
            ]
        ],
        workers=luigi_workers,
        local_scheduler=True,
    )


@app.command()
def evaluate_strawman(
    train_root: str,
    soundscape_root: str,
    output_root: str,
    gensim_workers: int = 8,
    luigi_workers: int = 8,
):
    """Evaluate the model over 100 parameters.

    If things are done correctly, then we should be able to get a model that is
    checkpointed every 10 epochs.
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
                n_clusters=n_clusters,
                filter_species=colombia_species_list,
                epochs=epochs,
                **params,
            )
            for tokenizer in ["tokenizer"]
            for n_clusters in [2**14 - 1]
            for input_root, output_prefix in [(train_root, "train")]
            for params in [
                {
                    "vector_size": 256,
                    "window": 80,
                    "ns_exponent": 0.75,
                    "sample": 1e-4,
                }
            ]
            for epochs in [10 * i for i in range(1, 11)]
        ],
        workers=luigi_workers,
        local_scheduler=True,
    )


@app.command()
def evaluate_v1(
    train_root: str,
    soundscape_root: str,
    output_root: str,
    gensim_workers: int = 8,
    luigi_workers: int = 8,
):
    """Evaluate the model over 100 parameters.

    If things are done correctly, then we should be able to get a model that is
    checkpointed every 10 epochs.
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
                n_clusters=n_clusters,
                filter_species=colombia_species_list,
                epochs=epochs,
                **params,
            )
            for tokenizer in ["tokenizer"]
            for n_clusters in [2**14 - 1]
            for input_root, output_prefix in [(train_root, "train")]
            for params in [
                {
                    "vector_size": 384,
                    "window": 80,
                    "ns_exponent": 1.5,
                    "sample": 1e-5,
                }
            ]
            for epochs in [10 * i for i in range(1, 11)]
        ]
        + [
            EmbedWord2VecTask(
                input_root=input_root,
                soundscape_root=soundscape_root,
                output_root=output_root,
                output_prefix=output_prefix,
                workers=gensim_workers,
                tokenizer=tokenizer,
                n_clusters=n_clusters,
                **params,
            )
            for tokenizer in ["tokenizer"]
            for n_clusters in [2**14 - 1]
            for input_root, output_prefix in [
                (train_root, "train_all"),
                (soundscape_root, "soundscape_all"),
            ]
            for params in [
                {
                    "epochs": 100,
                    "vector_size": 384,
                    "window": 80,
                    "ns_exponent": 1.5,
                    "sample": 1e-5,
                }
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
