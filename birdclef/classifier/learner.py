import time
import numpy as np

from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    make_scorer,
    classification_report,
)


class Learner:
    def __init__(self, pipe, params):
        self.pipe = pipe
        self.params = params
        self.clf = None
        self.scores = None
        self.average = "macro"
        self.search_name = None
        self.class_report = None
        self.dataset_name = None
        self.learning_curve = {}
        self.validation_curve = {}
        self.cv = StratifiedKFold(n_splits=5, shuffle=True)
        self.name = str(self.pipe["model"].__class__.__name__)

    def fit_gridsearch(self, search_func, X_train, y_train, verbose=False):
        """
        Method to train the model using a search algorithm.

        search_func: GridSearchCV, RandomizedSearchCV from sklearn.
        X_train: training features dataset.
        y_train: training labels dataset.
        verbose: int() Controls the verbosity: the higher, the more messages (1, 2, or 3).
        """
        np.random.seed(42)

        # estimate class weights for unbalanced datasets
        weights = class_weight.compute_sample_weight(class_weight="balanced", y=y_train)

        # train learner
        self.clf = search_func(
            self.pipe,
            self.params,
            scoring={
                "accuracy": make_scorer(accuracy_score),
                "precision": make_scorer(precision_score, average=self.average),
                "recall": make_scorer(recall_score, average=self.average),
                "f1": make_scorer(f1_score, average=self.average),
            },
            refit="f1",
            cv=self.cv,
            verbose=verbose,
            n_jobs=1,
        )
        # fit the model
        self.clf.fit(X_train, y_train, **{"model__sample_weight": weights})
        self.search_name = str(self.clf.__class__.__name__)

    def get_scores(self, X_train, X_test, y_train, y_test, average=None):
        """
        Method to get model scores.

        X_train: training features dataset.
        X_test: test features dataset.
        y_train: training labels dataset.
        y_test: test labels dataset.
        """
        if self.search_name == "Benchmark":
            best_estimator = self.clf
        else:
            best_estimator = self.clf.best_estimator_

        np.random.seed(42)
        # score on training data
        start_time = time.time()
        best_estimator.fit(X_train, y_train)
        end_time = time.time()
        wall_clock_fit = end_time - start_time
        train_score = best_estimator.score(X_train, y_train)

        # score on test data
        start_time = time.time()
        y_pred = best_estimator.predict(X_test)
        end_time = time.time()
        wall_clock_pred = end_time - start_time
        test_score = best_estimator.score(X_test, y_test)

        # metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average=average)
        recall = recall_score(y_test, y_pred, average=average)
        f1 = f1_score(y_test, y_pred, average=average)

        # classification report
        self.class_report = classification_report(y_test, y_pred)

        self.scores = {
            "train_score": round(train_score, 3),
            "test_score": round(test_score, 3),
            "accuracy": round(accuracy, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "wall_clock_fit": wall_clock_fit,
            "wall_clock_pred": wall_clock_pred,
        }

    # evaluate Learner class
    def evaluate_learner(self, file_path: str = None):
        """
        Print model scores to console and optionally write to a .txt file.
        """
        lines = []
        lines.append(f"{'#################################' * 2}")
        lines.append(f"{self.search_name}:\t  {self.name}")
        lines.append(f"Train score:     {round(self.scores['train_score'], 3)}")
        lines.append(f"Test score:      {round(self.scores['test_score'], 3)}")
        lines.append(f"Accuracy score:  {round(self.scores['accuracy'], 3)}")
        lines.append(f"Precision score: {round(self.scores['precision'], 3)}")
        lines.append(f"Recall score:    {round(self.scores['recall'], 3)}")
        lines.append(f"F1 score:        {round(self.scores['f1'], 3)}")
        lines.append(f"Wall Clock Fit:  {round(self.scores['wall_clock_fit'], 3)}")
        lines.append(f"Wall Clock Pred: {round(self.scores['wall_clock_pred'], 3)}")
        lines.append(f"\nClassification report:\n{self.class_report}")
        lines.append(f"Best score: {round(self.clf.best_score_, 3)}")
        lines.append("Best params:")
        for param in self.clf.best_params_.items():
            lines.append(f"\t{param}")
        lines.append("")

        report_str = "\n".join(lines)
        print(report_str)

        if file_path:
            with open(file_path, "w") as f:
                f.write(report_str)
