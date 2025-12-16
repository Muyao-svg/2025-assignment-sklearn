"""Assignment - making a sklearn estimator and cv splitter.

The goal of this assignment is to implement by yourself:

- a scikit-learn estimator for the KNearestNeighbors for classification
  tasks and check that it is working properly.
- a scikit-learn CV splitter where the splits are based on a Pandas
  DateTimeIndex.

Detailed instructions for question 1:
The nearest neighbor classifier predicts for a point X_i the target y_k of
the training sample X_k which is the closest to X_i. We measure proximity with
the Euclidean distance. The model will be evaluated with the accuracy (average
number of samples corectly classified). You need to implement the `fit`,
`predict` and `score` methods for this class. The code you write should pass
the test we implemented. You can run the tests by calling at the root of the
repo `pytest test_sklearn_questions.py`. Note that to be fully valid, a
scikit-learn estimator needs to check that the input given to `fit` and
`predict` are correct using the `validate_data, check_is_fitted` functions
imported in this file.
You can find more information on how they should be used in the following doc:
https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator.
Make sure to use them to pass `test_nearest_neighbor_check_estimator`.


Detailed instructions for question 2:
The data to split should contain the index or one column in
datatime format. Then the aim is to split the data between train and test
sets when for each pair of successive months, we learn on the first and
predict of the following. For example if you have data distributed from
november 2020 to march 2021, you have have 4 splits. The first split
will allow to learn on november data and predict on december data, the
second split to learn december and predict on january etc.

We also ask you to respect the pep8 convention: https://pep8.org. This will be
enforced with `flake8`. You can check that there is no flake8 errors by
calling `flake8` at the root of the repo.

Finally, you need to write docstrings for the methods you code and for the
class. The docstring will be checked using `pydocstyle` that you can also
call at the root of the repo.

Hints
-----
- You can use the function:

from sklearn.metrics.pairwise import pairwise_distances

to compute distances between 2 sets of samples.
"""
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets


class KNearestNeighbors(ClassifierMixin, BaseEstimator):
    """KNearestNeighbors classifier."""

    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Fitting function.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data to train the model.
        y : ndarray, shape (n_samples,)
            Labels associated with the training data.

        Returns
        -------
        self : instance of KNearestNeighbors
            The current instance of the classifier
        """
        X, y = validate_data(
            self,
            X,
            y,
            ensure_2d=True,
            dtype=np.float64,
            y_numeric=False,
            multi_output=False,
        )
        check_classification_targets(y)

        self.X_train_ = X
        self.y_train_ = y
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        """Predict function.

        Parameters
        ----------
        X : ndarray, shape (n_test_samples, n_features)
            Data to predict on.

        Returns
        -------
        y : ndarray, shape (n_test_samples,)
            Predicted class labels for each test data sample.
        """
        check_is_fitted(self, ["X_train_", "y_train_", "classes_"])

        X = validate_data(
            self,
            X,
            ensure_2d=True,
            dtype=np.float64,
            reset=False,
        )

        n_test = X.shape[0]
        y_pred = np.empty(n_test, dtype=self.y_train_.dtype)
        k = int(self.n_neighbors)

        class_order = {c: i for i, c in enumerate(self.classes_)}

        for i in range(n_test):
            dists = np.linalg.norm(self.X_train_ - X[i], axis=1)
            nn_idx = np.argsort(dists)[:k]
            neigh = self.y_train_[nn_idx]

            labels, counts = np.unique(neigh, return_counts=True)
            max_count = counts.max()
            candidates = labels[counts == max_count]

            y_pred[i] = min(candidates, key=lambda c: class_order[c])

        return y_pred

    def score(self, X, y):
        """Compute the accuracy of the classifier."""
        check_is_fitted(self, ["X_train_", "y_train_", "classes_"])

        X, y = validate_data(
            self,
            X,
            y,
            ensure_2d=True,
            dtype=np.float64,
            y_numeric=False,
            multi_output=False,
            reset=False,
        )

        return float(np.mean(self.predict(X) == y))


class MonthlySplit(BaseCrossValidator):
    """CrossValidator based on monthly split.

    Split data based on the given `time_col` (or default to index). Each split
    corresponds to one month of data for the training and the next month of
    data for the test.

    Parameters
    ----------
    time_col : str, defaults to 'index'
        Column of the input DataFrame that will be used to split the data. This
        column should be of type datetime. If split is called with a DataFrame
        for which this column is not a datetime, it will raise a ValueError.
        To use the index as column just set `time_col` to `'index'`.
    """
    def __init__(self, time_col="index"):
        self.time_col = time_col

    def __repr__(self):
        return f"MonthlySplit(time_col='{self.time_col}')"

    def _get_datetime_index(self, X):
        # 支持 DataFrame / Series
        if not isinstance(X, (pd.DataFrame, pd.Series)):
            raise ValueError(
                "Input X should be a pandas DataFrame to use MonthlySplit."
                )

        if self.time_col == "index":
            time_vals = X.index
            if not pd.api.types.is_datetime64_any_dtype(time_vals):
                raise ValueError(
                    f"The column {self.time_col} is not of datetime type."
                    )

            return pd.DatetimeIndex(time_vals)

        # time_col != 'index' 时必须是 DataFrame
        if isinstance(X, pd.Series):
            raise ValueError(
                "Input X should be a pandas DataFrame to use MonthlySplit."
                )

        col = X[self.time_col]
        if not pd.api.types.is_datetime64_any_dtype(col):
            raise ValueError(
                f"The column {self.time_col} is not of datetime type."
                )

        return pd.DatetimeIndex(col.to_numpy())

    def get_n_splits(self, X, y=None, groups=None):
        """Return the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            The number of splits.
        """
        dt_index = self._get_datetime_index(X)
        months = dt_index.to_period("M").unique()
        return max(len(months) - 1, 0)

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        Yields
        ------
        idx_train : ndarray
            The training set indices for that split.
        idx_test : ndarray
            The testing set indices for that split.
        """
        dt_index = self._get_datetime_index(X)
        periods = dt_index.to_period("M")
        months = pd.PeriodIndex(periods.unique()).sort_values()

        for i in range(len(months) - 1):
            train_idx = np.where(periods == months[i])[0]
            test_idx = np.where(periods == months[i + 1])[0]
            yield train_idx, test_idx
