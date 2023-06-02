from .base import BaseRegressor
from sklearn.ensemble import GradientBoostingRegressor


class GradientBoostingRegressorModel(BaseRegressor, GradientBoostingRegressor):
    """
    GradientBoostingRegressorModel is a regression model based on the Gradient Boosting algorithm.

    Parameters
    ----------
        n_estimators :int
            The number of boosting stages. Default is 100.
        learning_rate :float
            The learning rate shrinks the contribution of each tree. Default is 0.1.
        max_depth : int or None
            The maximum depth of the tree. Default is 3.
        min_samples_split :int
            The minimum number of samples required to split an internal node. Default is 2.
        min_samples_leaf :int
            The minimum number of samples required to be at a leaf node. Default is 1.
        max_features : int, float, string or None
            The number of features to consider when looking for the best split. Default is None.
    """

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
    ):
        GradientBoostingRegressor.__init__(
            self,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
        )
        self.name = "GradientBoostingRegressorModel"
        BaseRegressor.__init__(self)

    def tune_model(self, X_train, X_val, y_train, y_val):
        """
        Tunes the hyperparameters of the GradientBoostingRegressorModel using grid search and cross-validation.

        Parameters
        ----------
            X_train : numpy.ndarray
                The training features.
            X_val : numpy.ndarray
                The validation features.
            y_train : numpy.ndarray
                The training target.
            y_val : numpy.ndarray
                The validation target.

        Returns
        -------
            tuple
                A tuple containing the best estimator and the best parameters found during tuning.
        """
        param_grid = {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 10, 15, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 5],
            "max_features": ["sqrt", "log2", None],
        }

        return super().tune_model(X_train, X_val, y_train, y_val, param_grid, self)

    def get_params(self, deep=True):
        """
        Returns the current hyperparameters of the GradientBoostingRegressorModel.

        Parameters
        ----------
            deep :bool
                If True, return the parameters of all sub-objects that are estimators. If False, return only the top-level parameters. Default is True.

        Returns
        -------
            dict
                The current hyperparameters of the GradientBoostingRegressorModel.
        """
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
        }
