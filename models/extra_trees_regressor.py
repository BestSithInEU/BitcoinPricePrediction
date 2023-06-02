from .base import BaseRegressor
from sklearn.ensemble import ExtraTreesRegressor


class ExtraTreesRegressorModel(BaseRegressor, ExtraTreesRegressor):
    """
    ExtraTreesRegressorModel is a regression model based on the Extra Trees algorithm.

    Parameters
    ----------
        n_estimators : int
            The number of trees in the forest. Default is 100.
        max_depth : int or None
            The maximum depth of the tree. Default is None.
        min_samples_split : int
            The minimum number of samples required to split an internal node. Default is 2.
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
    ):
        ExtraTreesRegressor.__init__(
            self,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
        )
        self.name = "ExtraTreesRegressorModel"
        BaseRegressor.__init__(self)

    def tune_model(self, X_train, X_val, y_train, y_val):
        """
        Tunes the hyperparameters of the ExtraTreesRegressorModel using grid search and cross-validation.

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
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
        }
        return super().tune_model(X_train, X_val, y_train, y_val, param_grid, self)

    def get_params(self, deep=True):
        """
        Returns the current hyperparameters of the ExtraTreesRegressorModel.

        Parameters
        ----------
            deep : bool
                If True, return the parameters of all sub-objects that are estimators. If False, return only the top-level parameters. Default is True.

        Returns
        -------
            dict
                The current hyperparameters of the ExtraTreesRegressorModel.
        """
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
        }
