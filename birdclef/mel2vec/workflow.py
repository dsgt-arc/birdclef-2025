import luigi
import faiss
import typer
import numpy as np
import polars as pl
from pathlib import Path
from .callback import TqdmCallback
from gensim.models import Word2Vec

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


class Word2VecTask(luigi.Task, OptionsMixin):
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


@app.command()
def run(
    input_root: str, output_root: str, gensim_workers: int = 8, luigi_workers: int = 8
):
    """Run the tokenizer building process."""
    luigi.build(
        [
            Word2VecTask(
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
