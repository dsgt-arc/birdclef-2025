import luigi
import faiss
import typer
import numpy as np
import polars as pl
from pathlib import Path
from .callback import TqdmCallback
from gensim.models import Word2Vec, KeyedVectors
from functools import cache

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
                f"{self.output_root}/{self.prefix}/centroids.npy"
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
                f"{self.output_root}/{self.prefix}/centroids.npy"
            ),
            "pca": luigi.LocalTarget(f"{self.output_root}/{self.prefix}/pca.bin"),
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


class Word2VecTask(luigi.Task, Word2VecOptionsMixin):
    def requires(self):
        return {
            "tokenizer": BuildTokenizer(
                input_root=self.input_root,
                output_root=self.output_root,
            ),
            "tokenizer_pca": BuildPCATokenizer(
                input_root=self.input_root,
                output_root=self.output_root,
            ),
        }[self.tokenizer]

    def output(self):
        prefix = "/".join(
            f"{k}={v}"
            for k, v in [
                ("tokenizer", self.tokenizer),
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


class EmbedWord2VecTask(luigi.Task, Word2VecOptionsMixin):
    """Task to embed audio files using the trained Word2Vec model."""

    def output(self):
        prefix = "/".join(
            f"{k}={v}"
            for k, v in [
                ("tokenizer", self.tokenizer),
                ("vector_size", self.vector_size),
                ("window", self.window),
                ("ns_exponent", self.ns_exponent),
                ("sample", self.sample),
                ("epochs", self.epochs),
            ]
        )
        return luigi.LocalTarget(f"{self.output_root}/embedding/{prefix}")

    def requires(self):
        word2vec = Word2VecTask(
            input_root=self.input_root,
            output_root=self.output_root,
            epochs=self.epochs,
            vector_size=self.vector_size,
            window=self.window,
            ns_exponent=self.ns_exponent,
            sample=self.sample,
            workers=self.workers,
            tokenizer=self.tokenizer,
        )
        return {
            "word2vec": word2vec,
            "tokenizer": word2vec.requires(),
        }

    @cache
    def get_index(self):
        """Get the FAISS index for the centroids."""
        centroids = np.load(self.requires()["tokenizer"].output()["centroids"].path)
        index = faiss.IndexFlatL2(centroids.shape[1])
        index.add(centroids)
        return index

    @cache
    def get_word_vectors(self):
        """Get the word vectors from the Word2Vec model."""
        word_vectors = KeyedVectors.load(
            self.requires()["word2vec"].output()["wordvectors"].path,
            mmap="r",
        )
        return word_vectors

    def get_start_time(self, timestamp, interval=5) -> int:
        # up to but not including the value
        for i in range(0, 100, interval):
            if i <= timestamp < i + interval:
                return i
        return -1

    def mfcc_to_wv(self, mfcc: list) -> list:
        # convert mfcc to word vectors
        X = np.array(mfcc).reshape(1, -1)
        _, indices = self.get_index().search(X, 1)  # get the closest centroid
        return self.get_word_vectors()[indices[0][0]].tolist()

    def aggregate_mfcc(self, group: pl.DataFrame) -> pl.DataFrame:
        X_mfcc = np.stack(group.get_column("mfcc").to_numpy())
        X_w2v = np.stack(group.get_column("word_vector").to_numpy())
        return pl.DataFrame(
            {
                "file": group.get_column("file").to_numpy()[0],
                "start_time": group.get_column("start_time").to_numpy()[0],
                "mfcc_stats": [
                    X_mfcc.mean(axis=0).tolist() + X_mfcc.std(axis=0).tolist()
                ],
                "word_vector": [X_w2v.mean(axis=0).tolist()],
            }
        )

    def run(self):
        mfcc = pl.scan_parquet(self.input_root).with_columns(
            pl.col("timestamp")
            .map_elements(self.get_start_time, return_dtype=pl.Int64)
            .alias("start_time")
        )
        processed = (
            mfcc.with_columns(
                pl.col("mfcc")
                .map_elements(self.mfcc_to_wv, return_dtype=pl.List(pl.Float64))
                .alias("word_vector")
            )
            .group_by("file", "start_time")
            .map_groups(
                self.aggregate_mfcc,
                schema=pl.Schema(
                    {
                        "file": pl.Utf8,
                        "start_time": pl.Int64,
                        "mfcc_stats": pl.List(pl.Float64),
                        "word_vector": pl.List(pl.Float64),
                    }
                ),
            )
            .sort("file", "start_time")
        )
        processed.sink_parquet(self.output().path, compression="zstd")


@app.command()
def run(
    input_root: str, output_root: str, gensim_workers: int = 8, luigi_workers: int = 8
):
    """Run the tokenizer building process."""
    luigi.build(
        [
            EmbedWord2VecTask(
                input_root=input_root,
                output_root=output_root,
                epochs=100,
                vector_size=256,
                window=80,
                ns_exponent=0.75,
                sample=1e-4,
                workers=gensim_workers,
                tokenizer=tokenizer,
            )
            for tokenizer in ["tokenizer", "tokenizer_pca"]
        ],
        workers=luigi_workers,
        local_scheduler=True,
    )


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn")
    app()
