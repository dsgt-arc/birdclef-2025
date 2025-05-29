import typer
import pickle
import numpy as np
import pandas as pd

from .learner import Learner
from xgboost import XGBClassifier
from typing_extensions import Annotated
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from skopt import BayesSearchCV

app = typer.Typer()


def get_param_grid():
    return {
        "model__objective": ["multi:softmax"],
        "model__n_estimators": [400],  # np.arange(50, 200, 50),
        "model__max_depth": [3, 4, 5],  # np.arange(3, 10),
        # "model__min_child_weight": np.arange(1, 6),
        # "model__gamma": np.linspace(0, 0.6, 5),
        # "model__subsample": np.linspace(0.5, 1.0, 6),
        # "model__colsample_bytree": np.linspace(0.5, 1.0, 6),
    }


def preprocess_data(input_path: str) -> pd.DataFrame:
    df = pd.read_parquet(input_path)
    # concatenate all embeddings into a single DataFrame
    df["species_name"] = df["file"].apply(
        lambda x: x.split("train_audio/")[1].split("/")[0]
    )
    embed_cols = list(map(str, range(1280)))
    df["embeddings"] = df[embed_cols].values.tolist()
    df_embs = df[["species_name", "embeddings"]].copy()
    print(f"DataFrame shape: {df_embs.shape}")
    print(f"Embedding size: {len(df_embs['embeddings'].iloc[0])}")
    return df_embs


def perform_train_test_split(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple:
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        np.stack(df["embeddings"]),
        df["species_name"],
        test_size=test_size,
        stratify=df["species_name"],
    )

    # data shape
    print(f"X_train, X_test shape: {X_train.shape, X_test.shape}")
    print(f"y_train, y_test shape: {y_train.shape, y_test.shape}")
    return X_train, X_test, y_train, y_test


def train_model(
    X_train: pd.Series,
    X_test: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
    search_method: str = "random",
) -> Learner:
    # create a label encoder object
    le = LabelEncoder()

    # fit and transform the target with label encoder
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # XGBoost pipeline
    xgb_pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", XGBClassifier(seed=42)),
        ]
    )
    # GridSearchCV params
    xgb_param_grid = get_param_grid()

    # init learners
    xgb = Learner(pipe=xgb_pipe, params=xgb_param_grid)
    # fit model
    if search_method == "grid":
        search_func = GridSearchCV()
    elif search_method == "bayesian":
        search_func = BayesSearchCV(
            n_iter=10,
            scoring="f1",
            verbose=4,
            cv=3,
            n_points=1,
            n_jobs=-1,
        )
    elif search_method == "random":
        search_func = RandomizedSearchCV()
    xgb.fit_gridsearch(search_func, X_train, y_train_enc, verbose=2)
    # get model scores
    xgb.get_scores(X_train, X_test, y_train_enc, y_test_enc, average="macro")
    # print scores
    xgb.evaluate_learner()
    return xgb


def save_model(model: Learner, output_path: str):
    # save model
    with open(output_path, "wb") as f:
        pickle.dump(model.clf.best_estimator_, f)
    print(f"Model saved to {output_path}")


@app.command()
def main(
    input_path: Annotated[str, typer.Argument(help="Input root directory")],
    output_path: Annotated[str, typer.Argument(help="Output root directory")],
    search_method: Annotated[
        str,
        typer.Option(
            help="Search method for hyperparameter tuning ('grid', 'random', 'bayesian')",
            default="random",
        ),
    ],
):
    # preprocess data
    df_embs = preprocess_data(input_path)

    # train/test split
    X_train, X_test, y_train, y_test = perform_train_test_split(df_embs)

    # train model
    xgb = train_model(X_train, X_test, y_train, y_test, search_method)

    # save model
    save_model(xgb, output_path)
    print("Training completed successfully!")


if __name__ == "__main__":
    app()
