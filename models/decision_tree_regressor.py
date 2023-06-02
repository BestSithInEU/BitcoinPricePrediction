from .base import BaseRegressor
from sklearn.tree import DecisionTreeRegressor


class DecisionTreeRegressorModel(BaseRegressor, DecisionTreeRegressor):
    """
    DecisionTreeRegressorModel is a regression model based on the decision tree algorithm.

    Parameters
    ----------
        max_depth : int or None
            The maximum depth of the tree. Default is None.
        min_samples_split : int
            The minimum number of samples required to split an internal node. Default is 2.
        min_samples_leaf : int
            The minimum number of samples required to be at a leaf node. Default is 1.
        max_features : int, float, string or None
            The number of features to consider when looking for the best split. Default is None.
    """

    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
    ):
        DecisionTreeRegressor.__init__(
            self,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
        )
        self.name = "DecisionTreeRegressorModel"
        BaseRegressor.__init__(self)

    def tune_model(self, X_train, X_val, y_train, y_val):
        """
        Tunes the hyperparameters of the DecisionTreeRegressorModel using grid search and cross-validation.

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
            "max_depth": [5, 10, 15, 20, 25, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 5],
            "max_features": ["sqrt", "log2", None],
        }
        return super().tune_model(X_train, X_val, y_train, y_val, param_grid, self)

    def get_params(self, deep=True):
        """
        Returns the current hyperparameters of the DecisionTreeRegressorModel.

        Parameters
        ----------
            deep : bool
                If True, return the parameters of all sub-objects that are estimators. If False, return only the top-level parameters. Default is True.

        Returns
        -------
            dict
                The current hyperparameters of the DecisionTreeRegressorModel.
        """
        return {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
        }
